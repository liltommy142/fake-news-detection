import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import scipy.sparse as sp
import joblib
import os
import gc # Garbage Collector để giải phóng RAM

def train_fast_hybrid_model():
    print("\n" + "="*70)
    print(" PHASE 7: HIGH-SPEED HYBRID MODEL TRAINING (XGBOOST + NLP) ")
    print("="*70)

    file_path = 'input_for_model.csv'
    if not os.path.exists(file_path):
        print("[!] Error: CSV file not found!")
        return

    # ==========================================
    # 1. SIÊU TỐC ĐỌC DATA (Bỏ qua engine chậm chạp)
    # ==========================================
    print("[1/5] Loading Data...")
    
    # Định nghĩa sẵn kiểu dữ liệu để Pandas không phải đoán (tiết kiệm 50% thời gian)
    dtypes = {
        'Thread_ID': str, 'Author': str, 'Score': np.float32, 
        'Depth': np.int16, 'Spread': np.int16, 'Ver': np.int8, 
        'Follow': np.int32, 'AccAge': np.int16, 'Engage': np.float32, 
        'Src': np.int8, 'Content_Snippet': str, 'Label': str
    }
    
    # Dùng regex delimiter mạnh hơn để dọn sạch khoảng trắng thừa
    df = pd.read_csv(file_path, sep=r'\s*\|\s*', engine='python', skiprows=[1], dtype=dtypes, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]

    # ==========================================
    # 2. XỬ LÝ NHÃN VÀ DỮ LIỆU RỖNG TRONG 1 NỐT NHẠC
    # ==========================================
    print("[2/5] Cleaning Data...")
    df['Label'] = df['Label'].str.strip()
    # Loại bỏ ngay những dòng có Label tào lao (do lỗi cắt chữ từ C++)
    df = df[df['Label'].isin(['fake', 'real'])]
    df['Label_Num'] = df['Label'].map({'fake': 1, 'real': 0}).astype(np.int8)
    
    df['Content_Snippet'] = df['Content_Snippet'].fillna('')

    # Gom các cột số liệu (Loại bỏ `to_numeric` chậm chạp)
    meta_cols = ['Score', 'Depth', 'Spread', 'Ver', 'Follow', 'AccAge', 'Engage']
    
    # Chuyển đổi và điền NaN cực nhanh bằng numpy
    X_meta = np.nan_to_num(df[meta_cols].values.astype(np.float32))

    print(f"      -> Retained {len(df)} pristine samples.")

    # ==========================================
    # 3. CHUẨN HÓA VÀ TRÍCH XUẤT ĐẶC TRƯNG NLP
    # ==========================================
    print("[3/5] Extracting Features (TF-IDF & Meta Scaling)...")
    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)

    # Tối ưu TF-IDF: Chặn document frequency quá thấp/cao để giảm noise
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english', min_df=3, max_df=0.85)
    X_text = tfidf.fit_transform(df['Content_Snippet'])

    # Ghép 2 ma trận bằng Scipy Sparse (Rất nhẹ RAM)
    X_combined = sp.hstack((X_text, sp.csr_matrix(X_meta_scaled)), format='csr')
    y = df['Label_Num'].values
    
    # Giải phóng dataframe cũ khỏi RAM
    del df
    del X_meta
    del X_meta_scaled
    gc.collect() 

    # ==========================================
    # 4. CHIA TẬP TRAIN/TEST VÀ ÉP XUNG XGBOOST
    # ==========================================
    print("[4/5] Training XGBoost (Hist-Tree Method)...")
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

    # Xử lý mất cân bằng dữ liệu
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.08,        # Giảm nhẹ LR để bắt nhịp với số cây nhiều hơn
        max_depth=8,               # Giữ nguyên độ sâu
        scale_pos_weight=ratio,
        tree_method='hist',        # THUẬT TOÁN QUAN TRỌNG NHẤT CHO HIỆU SUẤT
        n_jobs=-1,                 # Full CPU Cores
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=30   # Dừng sớm nếu 30 vòng không cải thiện (giảm từ 50 để chạy nhanh hơn)
    )

    # Chạy Train Model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print(f"      -> Converged at tree number: {model.best_iteration}")

    # ==========================================
    # 5. ĐÁNH GIÁ VÀ LƯU TRỮ
    # ==========================================
    print("\n[5/5] Evaluation & Saving...")
    y_pred = model.predict(X_test)
    
    print(f"\n[!!!] FINAL ACCURACY: {accuracy_score(y_test, y_pred):.2%}\n")
    print(classification_report(y_test, y_pred, target_names=['Real News', 'Fake News']))

    joblib.dump(model, 'max_samples_hybrid_model.pkl')
    joblib.dump(tfidf, 'tfidf_max_vectorizer.pkl')
    joblib.dump(scaler, 'scaler_max.pkl')
    print("[+] Assets saved! Ready for inference.")

if __name__ == "__main__":
    train_fast_hybrid_model()