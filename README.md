# 🕵️‍♂️ Fake News Detection: Graph-based Spread Analysis
**Course**: Data Structures and Algorithms (DSA) - HCMUS  
**Team**: 4lg0r1thm  

## 📌 Overview
Dự án tập trung vào việc nhận diện tin giả (rumors) trên nền tảng mạng xã hội Twitter bằng cách kết hợp giữa **Cấu trúc dữ liệu Đồ thị/Cây** và **Machine Learning**. Thay vì chỉ phân tích nội dung văn bản, hệ thống này tập trung vào **cấu trúc lan truyền** (propagation structure) của các luồng hội thoại (conversation threads).



## 🏗️ DSA Core Implementation
Điểm khác biệt của đồ án này là việc mô hình hóa dữ liệu Twitter dưới dạng **Tree Structure**.

*   **Tree Modeling**: Mỗi Tweet gốc là `Root`, các lượt Reply là `Children`.
*   **Traversals (Duyệt cây)**: 
    *   Sử dụng **BFS (Breadth-First Search)** để tính toán độ rộng lan truyền theo thời gian.
    *   Sử dụng **DFS (Depth-First Search)** để xác định độ sâu của cuộc hội thoại (Conversation Depth).
*   **Structural Features**: Trích xuất các thuộc tính từ cấu trúc cây để đưa vào Model:
    *   *Depth*: Độ sâu tối đa của luồng phản hồi.
    *   *Breadth*: Số lượng phản hồi trung bình tại mỗi tầng.
    *   *Structural Virality*: Chỉ số lan truyền dựa trên khoảng cách trung bình giữa các node.

## 📊 Dataset: PHEME
*   Sử dụng tập dữ liệu **PHEME** bao gồm các sự kiện lớn (Ferguson, Charlie Hebdo, Germanwings crash, v.v.)
*   Dữ liệu được tổ chức dưới dạng các file JSON đại diện cho các nhánh hội thoại (threads).

## 📂 Project Structure
```bash
.
├── data/
│   ├── raw/            # Dataset gốc (PHEME)
│   └── processed/      # Dữ liệu sau khi dựng cây và trích xuất đặc trưng
├── notebooks/          # EDA và thử nghiệm thuật toán
├── src/
│   ├── full_preprocessing.py     # Tải dữ liệu, xây dựng cây và làm sạch
│   ├── feature_engineering.py    # Chuyển đổi cấu trúc cây thành Vector số học
│   ├── mini_preprocessing.py     # Xử lý dữ liệu nhỏ
│   ├── utils.py                  # Các hàm tiện ích và TreeNode
│   └── requirements.txt           # Dependencies cụ thể
├── requirements.txt    # Danh sách thư viện chính
└── .gitignore          # Đã loại bỏ .venv, .exe, và dữ liệu nặng
```

## 🚀 Quick Start
1. **Clone & Setup**:
   ```bash
   git clone <repository-url>
   cd fake-news-detection
   python -m venv .venv
   source .venv/bin/activate  # Hoặc .venv\Scripts\activate trên Windows
   pip install -r requirements.txt
   ```

2. **Run Preprocessing Pipeline**:
   ```bash
   # Xử lý dữ liệu và xây dựng cây từ PHEME dataset
   python src/full_preprocessing.py
   
   # Trích xuất đặc trưng từ cấu trúc cây
   python src/feature_engineering.py
   ```

3. **View Results**:
   - Dữ liệu xử lý được lưu trong `data/processed/`
   - Kết quả đánh giá mô hình trong `results/metrics.txt`
   - Biểu đồ trực quan trong `results/figures/`

## 👥 Contributors (Team 4lg0r1thm)
* **Lil'Tommy**
* **Loc**
* **Thien Khanh**
