import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import scipy.sparse as sp
import joblib
import os
import gc
import concurrent.futures
import trafilatura 
import warnings

# Tắt cảnh báo để terminal sạch sẽ
warnings.filterwarnings('ignore')

# ==========================================
# 0. HỆ THỐNG CACHE VÀ BẮT LINK CHUYÊN SÂU
# ==========================================
URL_CACHE = {}

def extract_and_format_urls(text):
    """Bắt link cực mạnh, bất chấp không có http/https/www"""
    if not isinstance(text, str): return []
    
    # Regex bắt domain có đuôi .com, .net, .vn,... và các đường dẫn theo sau
    url_pattern = r'(?i)\b(?:(?:https?://|www\.)[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)\b'
    raw_urls = re.findall(url_pattern, text)
    
    clean_urls = []
    for url in raw_urls:
        # Dọn dẹp dấu câu vô tình dính vào cuối link
        url = url.rstrip('.,;!"\'()[]{}')
        
        # Bơm thêm https:// cho các link trần trụi (như vnexpress.net)
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        clean_urls.append(url)
        
    return list(set(clean_urls))

def get_text_metadata(text):
    """Trích xuất 100% các chỉ số NLP có thể đếm được từ văn bản thuần"""
    text = str(text) if pd.notna(text) else ""
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    
    return {
        'Word_Count': word_count,
        'Char_Count': char_count,
        'Caps_Ratio': sum(1 for c in text if c.isupper()) / (char_count + 1e-5),
        'Exclamation_Density': text.count('!') / (char_count + 1e-5),
        'Question_Density': text.count('?') / (char_count + 1e-5),
        'Punctuation_Density': len(re.findall(r'[.,;:]', text)) / (char_count + 1e-5),
        'Avg_Word_Length': np.mean([len(w) for w in words]) if word_count > 0 else 0
    }

# ==========================================
# 1. CỖ MÁY CÀO WEB (CÀO SẠCH NỘI DUNG LINK)
# ==========================================
def fetch_url_content(url):
    """Cào nội dung thực tế của bài báo đằng sau link"""
    if url in URL_CACHE: return URL_CACHE[url]
    text = ""
    try:
        # Ưu tiên dùng trafilatura để bóc tách bài báo chuyên nghiệp
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            
        # Fallback: Nếu web chống cào, dùng BS4 bóc thẻ <p>
        if not text:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                paragraphs = soup.find_all(['p'])
                text = " ".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])
    except: 
        pass
        
    URL_CACHE[url] = text
    return text

def process_pipeline(args):
    """Phân loại luồng dữ liệu cho Model 1 và Model 2"""
    source_text, react_text = args
    source_text = str(source_text).strip() if pd.notna(source_text) else ""
    react_text = str(react_text).strip() if pd.notna(react_text) else ""
    
    # TEXT MODEL 1: Chỉ học từ văn bản nội tại (Bài gốc + Cảm xúc đám đông bình luận)
    text_model_1 = source_text
    if react_text:
        text_model_1 += f" [CROWD_COMMENTS] {react_text}"
        
    # Chuẩn bị cho Model 2
    source_urls = extract_and_format_urls(source_text)
    react_urls = extract_and_format_urls(react_text)
    
    # TEXT MODEL 2: Bơm thêm toàn bộ nội dung web cào được
    text_model_2 = text_model_1
    
    for url in source_urls:
        article = fetch_url_content(url)
        if article: text_model_2 += f" [SOURCE_LINK_CONTENT] {article} "
            
    for url in react_urls:
        article = fetch_url_content(url)
        if article: text_model_2 += f" [CROWD_LINK_CONTENT] {article} "

    return (text_model_1, text_model_2, len(source_urls), len(react_urls))

# ==========================================
# 2. XỬ LÝ LOGIC ĐÁM ĐÔNG (GẮN ĐÚNG THREAD_ID)
# ==========================================
def build_features(df):
    print("   -> Đang trích xuất Meta-features NLP thuần...")
    meta_df = df['Content_Snippet'].apply(get_text_metadata).apply(pd.Series)
    df = pd.concat([df, meta_df], axis=1)

    df['Thread_ID'] = df['Thread_ID'].astype(str).str.strip()
    
    # Tách chủ - tớ
    df_source = df[df['Is_Source'] == 1].copy()
    df_react = df[df['Is_Source'] == 0].copy()
    
    # Gom thông tin tương tác theo Thread_ID
    react_agg = df_react.groupby('Thread_ID').agg(
        React_Count=('Thread_ID', 'count'),
        React_Avg_Neg=('Negative_Score', 'mean'),
        React_Avg_Pos=('Positive_Score', 'mean'),
        React_Max_Neg=('Negative_Score', 'max')
    ).reset_index()
    
    # Gom chữ của đám đông
    react_text_agg = df_react.groupby('Thread_ID')['Content_Snippet'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    react_text_agg.rename(columns={'Content_Snippet': 'All_Crowd_Snippets'}, inplace=True)
    
    # Nối vào bài Gốc
    df_merged = pd.merge(df_source, react_agg, on='Thread_ID', how='left')
    df_merged = pd.merge(df_merged, react_text_agg, on='Thread_ID', how='left')
    
    # Xử lý các bài 0 tương tác
    fill_cols = ['React_Count', 'React_Avg_Neg', 'React_Avg_Pos', 'React_Max_Neg']
    df_merged[fill_cols] = df_merged[fill_cols].fillna(0)
    df_merged['All_Crowd_Snippets'] = df_merged['All_Crowd_Snippets'].fillna("")
    
    # Sinh đặc trưng Toán học: Độ lệch cảm xúc
    df_merged['Neg_Divergence'] = df_merged['React_Avg_Neg'] - df_merged['Negative_Score']
    df_merged['Pos_Divergence'] = df_merged['React_Avg_Pos'] - df_merged['Positive_Score']
    df_merged['Source_Sentiment_Gap'] = df_merged['Positive_Score'] - df_merged['Negative_Score']
    
    return df_merged

# ==========================================
# 3. HUẤN LUYỆN SIÊU MÔ HÌNH
# ==========================================
def train_lightgbm_dual_models():
    print("\n" + "="*80)
    print("🚀 BẮT ĐẦU TRAINING HỆ THỐNG KÉP (ULTIMATE NLP & WEB MINING EDITION)")
    print("="*80)

    file_path = 'input_for_model.csv'
    if not os.path.exists(file_path): 
        print(f"[-] Lỗi: Không tìm thấy {file_path}")
        return

    df = pd.read_csv(file_path, sep=r'\s*\|\s*', engine='python', skiprows=[1], on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    
    # Chuẩn hóa Data
    df['Is_Source'] = pd.to_numeric(df['Is_Source'], errors='coerce').fillna(0)
    df['Label_Num'] = df['Label'].str.strip().str.lower().map({'fake': 1, 'real': 0})
    df = df.dropna(subset=['Label_Num'])
    
    num_cols = ['Negative_Score', 'Positive_Score', 'Clickbait_Score', 'Depth', 'Spread', 'Verified', 'Followers', 'Engagement']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    print("[1/5] Xây dựng Cây đặc trưng (Tính toán độ lệch Nguồn - Đám đông)...")
    df_final = build_features(df)
    del df; gc.collect()

    print(f"[2/5] Khởi chạy 32 Luồng Cào Web ngầm ({len(df_final)} bài gốc)... Vui lòng đợi...")
    args_list = list(zip(df_final['Content_Snippet'], df_final['All_Crowd_Snippets']))
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for res in executor.map(process_pipeline, args_list):
            results.append(res)
            
    df_final['Text_Model_1'] = [str(res[0]) for res in results]
    df_final['Text_Model_2'] = [str(res[1]) for res in results]
    df_final['Source_Link_Count'] = [res[2] for res in results]
    df_final['React_Link_Count'] = [res[3] for res in results]

    print("[3/5] Mã hóa Văn bản Mật độ cao (Word + Char N-Grams)...")
    # TF-IDF Word (Bắt từ/cụm từ ý nghĩa)
    tfidf_word_1 = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english', min_df=2)
    tfidf_word_2 = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), sublinear_tf=True, stop_words='english', min_df=2)
    
    # TF-IDF Char (Bắt thói quen gõ phím, sai chính tả, ký tự lạ của Fake News)
    tfidf_char_1 = TfidfVectorizer(max_features=5000, analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True)
    tfidf_char_2 = TfidfVectorizer(max_features=10000, analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True)
    
    X_word_1 = tfidf_word_1.fit_transform(df_final['Text_Model_1'])
    X_char_1 = tfidf_char_1.fit_transform(df_final['Text_Model_1'])
    X_text_1 = sp.hstack([X_word_1, X_char_1]) # Ghép Word và Char lại
    
    X_word_2 = tfidf_word_2.fit_transform(df_final['Text_Model_2'])
    X_char_2 = tfidf_char_2.fit_transform(df_final['Text_Model_2'])
    X_text_2 = sp.hstack([X_word_2, X_char_2])

    print("[4/5] Hợp nhất Ma trận Dữ liệu...")
    # MODEL 1: Chỉ thuần NLP + Chỉ số Cảm xúc
    cols_1 = ['Negative_Score', 'Positive_Score', 'Clickbait_Score', 'Source_Sentiment_Gap', 
              'Word_Count', 'Char_Count', 'Caps_Ratio', 'Exclamation_Density', 'Question_Density', 
              'Punctuation_Density', 'Avg_Word_Length',
              'React_Count', 'React_Avg_Neg', 'React_Avg_Pos', 'React_Max_Neg', 
              'Neg_Divergence', 'Pos_Divergence']
              
    # MODEL 2: Đầy đủ Model 1 + Network + Thông số Link
    cols_2 = cols_1 + ['Depth', 'Spread', 'Verified', 'Followers', 'Engagement', 'Source_Link_Count', 'React_Link_Count']
    
    scaler_1 = RobustScaler()
    X_meta_1 = scaler_1.fit_transform(np.nan_to_num(df_final[cols_1].values.astype(np.float32)))
    
    scaler_2 = RobustScaler()
    X_meta_2 = scaler_2.fit_transform(np.nan_to_num(df_final[cols_2].values.astype(np.float32)))
    
    X_1_combined = sp.hstack((X_text_1, sp.csr_matrix(X_meta_1)), format='csr')
    X_2_combined = sp.hstack((X_text_2, sp.csr_matrix(X_meta_2)), format='csr')

    y = df_final['Label_Num'].values
    
    print("[5/5] Ép xung Mô hình LightGBM...")
    # Bộ Tham Số Hạng Nặng: Tìm kiếm mẫu cực sâu nhưng hạn chế Overfit
    lgb_params = {
        'n_estimators': 3000,        # Học rất lâu và kỹ
        'learning_rate': 0.01,       # Tốc độ học chậm lại để tối ưu hóa
        'num_leaves': 255,           # Tăng max độ phức tạp của cây
        'max_depth': 12,             # Cho phép cây đào sâu hơn
        'min_child_samples': 15,     # Tránh các lá nhiễu
        'subsample': 0.8,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.5,            # L1 Regularization (Loại bỏ feature rác)
        'reg_lambda': 2.0,           # L2 Regularization (Mịn hóa trọng số)
        'class_weight': 'balanced',  # Bắt buộc CÂN BẰNG DATA
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }

    X1_train, X1_test, y1_train, y1_test = train_test_split(X_1_combined, y, test_size=0.2, random_state=42, stratify=y)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X_2_combined, y, test_size=0.2, random_state=42, stratify=y)

    model_1 = lgb.LGBMClassifier(**lgb_params)
    model_1.fit(X1_train, y1_train)

    model_2 = lgb.LGBMClassifier(**lgb_params)
    model_2.fit(X2_train, y2_train)

    print("\n" + "="*60)
    print("🏆 BÁO CÁO KẾT QUẢ ĐỘ TAY (V9.0 ULTIMATE)")
    print("="*60)

    y1_pred = model_1.predict(X1_test)
    print(f"\n[*] MODEL 1 (VĂN BẢN THUẦN + CẢM XÚC) ACCURACY: {accuracy_score(y1_test, y1_pred):.2%}")
    print(classification_report(y1_test, y1_pred, target_names=['Real', 'Fake']))

    y2_pred = model_2.predict(X2_test)
    print(f"\n[*] MODEL 2 (FULL WEB NLP + ĐÁM ĐÔNG + TƯƠNG TÁC) ACCURACY: {accuracy_score(y2_test, y2_pred):.2%}")
    print(classification_report(y2_test, y2_pred, target_names=['Real', 'Fake']))

    # LƯU TRỮ TOÀN BỘ PIPELINE
    joblib.dump(model_1, 'lgbm_model_text.pkl')
    joblib.dump(model_2, 'lgbm_model_link.pkl')
    joblib.dump({'word': tfidf_word_1, 'char': tfidf_char_1}, 'tfidf_model_text.pkl')
    joblib.dump({'word': tfidf_word_2, 'char': tfidf_char_2}, 'tfidf_model_link.pkl')
    joblib.dump(scaler_1, 'scaler_text.pkl')
    joblib.dump(scaler_2, 'scaler_link.pkl')
    
    print("\n[+] Đã lưu tất cả model và pipeline chuẩn chỉ!")

if __name__ == "__main__":
    train_lightgbm_dual_models()