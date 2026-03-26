import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

def train_improved_model():
    print("\n" + "="*50)
    print(" PHASE 2: TRAINING IMPROVED AI MODEL ")
    print("="*50)

    # 1. Đọc dữ liệu
    file_path = 'input_for_model.csv'
    if not os.path.exists(file_path):
        print("[!] Error: CSV file not found!")
        return

    df = pd.read_csv(file_path, sep='|', skiprows=[1], skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    # 2. Tiền xử lý
    # Chuyển nhãn thành số và xử lý khoảng trắng
    df['Label_Num'] = df['Label'].str.strip().map({'fake': 1, 'real': 0})
    df = df.dropna(subset=['Label_Num'])

    X = df[['Score', 'Depth', 'Spread']]
    y = df['Label_Num']

    # Chia dữ liệu 70/30 để kiểm tra khắt khe hơn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Cấu hình Model cải tiến
    # - n_estimators=500: Dùng 500 cây quyết định thay vì 100
    # - class_weight='balanced': Ép AI phải học kỹ nhóm Fake News (nhóm ít hơn)
    # - max_depth=15: Cho phép cây học sâu hơn các quy luật phức tạp
    model = RandomForestClassifier(
        n_estimators=500, 
        max_depth=15, 
        min_samples_split=5,
        class_weight='balanced', 
        random_state=42
    )

    print("[*] AI is deep learning with Balanced Weights...")
    model.fit(X_train, y_train)

    # 4. Đánh giá
    y_pred = model.predict(X_test)
    
    print(f"\n[*] NEW ACCURACY: {accuracy_score(y_test, y_pred):.2%}")
    print("\n[!] Detailed Report:")
    print(classification_report(y_test, y_pred, target_names=['Real News', 'Fake News']))

    # 5. Xem "độ quan trọng" của các chỉ số
    print("-" * 30)
    print("Feature Importance (Cái nào quan trọng nhất?):")
    for name, importance in zip(X.columns, model.feature_importances_):
        print(f" - {name}: {importance:.4f}")
    print("-" * 30)

    # 6. Lưu model
    joblib.dump(model, 'fake_news_model_v2.pkl')
    print("[+] Improved model saved as 'fake_news_model_v2.pkl'")

if __name__ == "__main__":
    train_improved_model()