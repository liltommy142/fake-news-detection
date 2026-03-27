import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import scipy.sparse as sp
import joblib
import os
import gc
import concurrent.futures
import trafilatura 

# ==========================================
# 1. HÀM CÀO DỮ LIỆU SIÊU TỐC & CHỐNG BLOCK
# ==========================================
def extract_and_scrape(text):
    # [FIX BUG]: Ép toàn bộ dữ liệu lỗi/trống thành chuỗi rỗng "" thay vì NaN
    if pd.isna(text) or not isinstance(text, str) or str(text).strip() == '':
        return "", 0
    
    text = str(text).strip()
    
    # Nhận diện mọi loại link (http, www, hoặc các domain phổ biến)
    urls = re.findall(r'(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.(com|net|vn|org|edu|gov))', text)
    
    if not urls:
        return text, 0 # Không có link, trả về text gốc và cờ Has_Link = 0
    
    target_url = urls[0][0] if isinstance(urls[0], tuple) else urls[0]
    
    if not target_url.startswith('http'):
        target_url = 'https://' + target_url

    try:
        downloaded = trafilatura.fetch_url(target_url)
        if downloaded:
            article_text = trafilatura.extract(downloaded)
            if article_text:
                return text + " [ARTICLE] " + article_text[:3000], 1
                
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}
        response = requests.get(target_url, headers=headers, timeout=3)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all(['p', 'h1', 'h2'])
            article_text = " ".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
            return text + " [ARTICLE] " + article_text[:3000], 1
            
    except Exception:
        pass 

    return text, 1 

def process_row(row_text):
    return extract_and_scrape(row_text)

# ==========================================
# 2. HÀM TẠO ĐẶC TRƯNG HÀNH VI 
# ==========================================
def inject_smart_features(df):
    df['Sentiment_Gap'] = df['Pos_Scr'] - df['Neg_Scr']
    df['Eng_Ratio'] = df['Engage'] / (df['Follow'] + 1)
    df['Viral_Factor'] = df['Spread'] / (df['Depth'] + 1)
    df['Trust_Score'] = (df['Ver'] * 2) + df['Src']
    return df

# ==========================================
# 3. HÀM HUẤN LUYỆN CHÍNH
# ==========================================
def train_ultimate_dual_model():
    print("\n" + "="*70)
    print("🚀 TRAINING ULTIMATE DUAL-BRAIN ARCHITECTURE (GPU RTX 5060 ACCELERATED)")
    print("="*70)

    file_path = 'input_for_model.csv'
    if not os.path.exists(file_path):
        print(f"[-] LỖI: Không tìm thấy file '{file_path}'. Vui lòng chạy file C++ trước!")
        return

    df = pd.read_csv(file_path, sep=r'\s*\|\s*', engine='python', skiprows=[1], on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]

    df['Label_Num'] = df['Label'].str.strip().str.lower().map({'fake': 1, 'real': 0})
    df = df.dropna(subset=['Label_Num'])
    
    num_cols = ['Neg_Scr', 'Pos_Scr', 'ClickBt', 'Caps_Rt', 'Punct_D', 'Avg_W_L', 'Depth', 'Spread', 'Ver', 'Follow', 'AccAge', 'Engage', 'Src']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    print("[1/6] Injecting Smart Features (Engineering)...")
    df = inject_smart_features(df)

    print("[2/6] Scraping URLs for Context Enrichment (Using 24 CPU Threads)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        results = list(executor.map(process_row, df['Content_Snippet'].tolist()))
    
    # [FIX BUG 2 LỚP]: Ép kiểu dữ liệu về String thêm 1 lần nữa cho chắc chắn
    df['Full_Article_Text'] = [str(res[0]) for res in results]
    df['Has_Link'] = [res[1] for res in results]

    print("[3/6] Vectorizing Text (N-grams 1-3, Sublinear TF)...")
    tfidf = TfidfVectorizer(
        max_features=12000, 
        ngram_range=(1, 3), 
        sublinear_tf=True, 
        stop_words='english'
    )
    X_text = tfidf.fit_transform(df['Full_Article_Text'])

    print("[4/6] Preparing Dual Feature Sets with RobustScaler...")
    nlp_cols = ['Neg_Scr', 'Pos_Scr', 'ClickBt', 'Caps_Rt', 'Punct_D', 'Avg_W_L', 'Sentiment_Gap', 'Has_Link']
    
    X_meta_nlp = np.nan_to_num(df[nlp_cols].values.astype(np.float32))
    scaler_nlp = RobustScaler()
    X_meta_nlp_scaled = scaler_nlp.fit_transform(X_meta_nlp)
    X_combined_text_only = sp.hstack((X_text, sp.csr_matrix(X_meta_nlp_scaled)), format='csr')

    full_cols = nlp_cols + ['Depth', 'Spread', 'Ver', 'Follow', 'AccAge', 'Engage', 'Src', 'Eng_Ratio', 'Viral_Factor', 'Trust_Score']
    X_meta_full = np.nan_to_num(df[full_cols].values.astype(np.float32))
    scaler_full = RobustScaler()
    X_meta_full_scaled = scaler_full.fit_transform(X_meta_full)
    X_combined_full = sp.hstack((X_text, sp.csr_matrix(X_meta_full_scaled)), format='csr')

    y = df['Label_Num'].values
    ratio = (float(np.sum(y == 0)) / np.sum(y == 1)) * 1.5 

    del df, X_meta_nlp, X_meta_full, X_meta_nlp_scaled, X_meta_full_scaled
    gc.collect()

    print("[5/6] Training Advanced Dual XGBoost Models (GPU CUDA)...")
    xgb_params = {
        'n_estimators': 3000, 
        'learning_rate': 0.02, 
        'max_depth': 8,
        'min_child_weight': 2, 
        'gamma': 0.1, 
        'subsample': 0.85,
        'colsample_bytree': 0.85, 
        'scale_pos_weight': ratio,
        'tree_method': 'hist', 
        'device': 'cuda', 
        'random_state': 42,
        'early_stopping_rounds': 100 
    }

    print("      -> Tuning Brain A (Text-Only Expert)...")
    Xa_train, Xa_test, ya_train, ya_test = train_test_split(X_combined_text_only, y, test_size=0.15, random_state=42, stratify=y)
    model_text = XGBClassifier(**xgb_params)
    model_text.fit(Xa_train, ya_train, eval_set=[(Xa_test, ya_test)], verbose=False)

    print("      -> Tuning Brain B (Full-Context Expert)...")
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_combined_full, y, test_size=0.15, random_state=42, stratify=y)
    model_full = XGBClassifier(**xgb_params)
    model_full.fit(Xb_train, yb_train, eval_set=[(Xb_test, yb_test)], verbose=False)

    print("\n" + "="*50)
    print("🏆 FINAL PERFORMANCE REPORTS ")
    print("="*50)

    ya_pred = model_text.predict(Xa_test)
    print(f"\n[*] BRAIN A (TEXT + NLP ONLY) ACCURACY: {accuracy_score(ya_test, ya_pred):.2%}")
    rep_a = classification_report(ya_test, ya_pred, target_names=['Real', 'Fake'], output_dict=True)
    print(f"    - Fake News F1-Score: {rep_a['Fake']['f1-score']:.2%}")

    yb_pred = model_full.predict(Xb_test)
    print(f"\n[*] BRAIN B (FULL CONTEXT) ACCURACY: {accuracy_score(yb_test, yb_pred):.2%}")
    rep_b = classification_report(yb_test, yb_pred, target_names=['Real', 'Fake'], output_dict=True)
    print(f"    - Fake News F1-Score: {rep_b['Fake']['f1-score']:.2%}")

    print("\n[6/6] Saving Ultimate Dual-Brains...")
    joblib.dump(model_text, 'xgboost_text_only.pkl')
    joblib.dump(model_full, 'xgboost_full_context.pkl')
    joblib.dump(tfidf, 'tfidf_dual_vectorizer.pkl')
    joblib.dump(scaler_nlp, 'scaler_nlp_only.pkl')
    joblib.dump(scaler_full, 'scaler_full_context.pkl')
    print("[+] All models saved successfully! Ready for app.py")

if __name__ == "__main__":
    train_ultimate_dual_model()