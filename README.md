# 🕵️‍♂️ Fake News Detection: Tree-based Propagation Analysis

**Course**: Data Structures and Algorithms (DSA) - HCMUS  
**Team**: 4lg0r1thm  

## 📌 Overview

Dự án tập trung vào việc nhận diện tin giả (rumors) trên nền tảng mạng xã hội Twitter bằng cách kết hợp giữa **Cấu trúc dữ liệu Cây (Tree)** và **Machine Learning**. Thay vì chỉ phân tích nội dung văn bản, hệ thống này tập trung vào **cấu trúc lan truyền** (propagation structure) của các luồng hội thoại (conversation threads).

Mỗi luồng Twitter được mô hình hóa dưới dạng một cây (Tree), nơi Tweet gốc là Root và các phản hồi (replies) là các node con. Thông qua các thuật toán DSA như DFS, BFS, và Hash Map, ta trích xuất các đặc trưng cấu trúc để dự đoán tính chân thực của thông tin.

## 🔄 Mô hình Vận hành

**Input**: Luồng hội thoại Twitter (từ PHEME dataset)  
**Output**: Phân loại (Tin Thật/Tin Giả) + Chỉ số cấu trúc lan truyền

### Quy Trình Xử Lý (DSA)

1. **Hash Map Indexing** - Mapping tweet_id → TreeNode để xây dựng cây O(n)
2. **Cấu trúc Cây + DFS** - Tính toán:
   - *Depth*: Độ sâu tối đa của luồng phản hồi
   - *Size*: Tổng số node (tweet) trong luồng
3. **BFS (Breadth-First Search)** - Phân tích từng mức độ lan truyền:
   - *Max Width*: Số lượng phản hồi nhiều nhất tại một mức
   - *Avg Branching*: Hệ số phân nhánh trung bình (số reply/tweet trung bình)
   - *Leaf Nodes*: Số lượng tweet cuối cùng (không có phản hồi)
4. **Text Heuristics** - Trích xuất đặc trưng từ nội dung:
   - Từ "sensational" (urgent, breaking, shocking, etc.)
   - Dấu câu (?, !, ...)
   - Độ dài text

### 🌳 Lớp TreeNode

```text
TreeNode - Lớp cơ sở mô hình hóa một tweet trong luồng hội thoại:
  Attributes:
    - tweet_id: Mã định danh Tweet (unique)
    - text: Nội dung văn bản của Tweet
    - user: Tên tài khoản người đăng
    - timestamp: Thời điểm đăng tweet
    - label: Nhãn chân thực (true, false, unverified)
    - children: Danh sách các phản hồi trực tiếp (node con)
  
  Methods:
    - get_depth(): Tính độ sâu cây sử dụng DFS O(n)
    - get_size(): Tính tổng số node trong cây O(n)
    - traverse_preorder(): Duyệt tiền thứ tự (Pre-order) O(n)
```

## 📊 Dataset: PHEME

- Sử dụng tập dữ liệu **PHEME** bao gồm các sự kiện lớn (Ferguson, Charlie Hebdo, Germanwings crash, v.v.)
- Dữ liệu được tổ chức dưới dạng các file JSON đại diện cho các nhánh hội thoại (threads).

## 📥 Dataset Installation

1. **Download PHEME dataset** từ nguồn chính thức:
   - [All-in-one: Multi-task Learning for Rumour Verification](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)
   - Trong repo này cũng đã tải sẵn tại folder: `/references`

2. **Giải nén và đặt** vào thư mục `data/raw/`

3. **Cấu trúc thư mục expected:**

   ```plaintext
   data/raw/all-rnr-annotated-threads/
   ├── charliehebdo-all-rnr-threads/
   │   ├── rumours/
   │   │   ├── 552784600502915072/
   │   │   │   ├── annotation.json
   │   │   │   ├── source-tweets/
   │   │   │   └── reactions/
   │   │   └── ...
   │   └── non-rumours/
   ├── ebola-essien-all-rnr-threads/
   ├── ferguson-all-rnr-threads/
   ├── germanwings-crash-all-rnr-threads/
   └── ... (các event khác)
   ```

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

## 🧩 Dependencies

**Core Libraries:**

- `numpy` - Xử lý dữ liệu số học
- `pandas` - Xử lý DataFrame & CSV
- `scikit-learn` - Thuật toán Machine Learning (Classification)
- `matplotlib` - Vẽ biểu đồ và trực quan hóa

**Python Built-in:**

- `json` - Xử lý file JSON
- `pickle` - Serialize/deserialize dữ liệu Python
- `collections.deque` - Queue implementation cho BFS
- `typing` - Type hints

## 🚀 Quick Start

1. **Clone & Setup**:

   ```bash
   git clone <repository-url>
   cd fake-news-detection
   python -m venv .venv
   source .venv/bin/activate  # Hoặc .venv\Scripts\activate trên Windows
   pip install -r requirements.txt
   ```

2. **Dataset Setup** (xem phần Dataset Installation ở trên)

3. **Run Preprocessing Pipeline**:

   ```bash
   # Bước 1: Tải dữ liệu PHEME và xây dựng cây từ JSON
   python src/full_preprocessing.py
   
   # Bước 2: Trích xuất đặc trưng từ cấu trúc cây
   # (Sử dụng DFS, BFS, Text Heuristics)
   python src/feature_engineering.py
   ```

4. **Alternative - Test với dữ liệu nhỏ**:

   ```bash
   # Để test xử lý nhanh mà không cần toàn bộ dataset
   python src/mini_preprocessing.py
   ```

## 📤 Expected Output

**Sau khi chạy:** `python src/full_preprocessing.py`

- `data/processed/trees_data.pkl` - Dữ liệu cây được serialize (các TreeNode objects)
- `data/processed/threads_compact.json` - Luồng Twitter dạng JSON compact

**Sau khi chạy:** `python src/feature_engineering.py`

- `data/processed/extracted_features.csv` - Feature vectors (Depth, Size, Max Width, Avg Branching, Text features, Binary Label)
- `data/processed/pheme_features.csv` - Features đầy đủ + Labels

**CSV Output Columns:**

- `tweet_id` - ID của Tweet gốc
- `tree_depth` - Độ sâu tối đa của cây (DFS)
- `tree_size` - Tổng số tweet trong luồng
- `max_width` - Số lượng phản hồi tầng sâu nhất
- `avg_branching` - Hệ số phân nhánh trung bình
- `leaf_nodes` - Số tweet không có phản hồi
- `word_count` - Số từ trong tweet gốc
- `sensational_count` - Số từ sensational
- `has_question_mark` - Có dấu ? (0/1)
- `has_exclamation` - Có dấu ! (0/1)
- `binary_label` - Nhãn (0=true/non-rumor, 1=false/rumor)

## ⚡ Troubleshooting

| Lỗi                               | Giải pháp                                                             |
| --------------------------------- | --------------------------------------------------------------------- |
| `FileNotFoundError: data/raw/...` | Đảm bảo dataset PHEME được đặt đúng vị trí (xem Dataset Installation) |
| `ModuleNotFoundError: utils`      | Chạy script từ thư mục `src/` hoặc thêm `src/` vào PYTHONPATH         |
| `MemoryError` khi xử lý data      | Sử dụng `python src/mini_preprocessing.py` để test với dữ liệu nhỏ    |
| `pickle.UnpicklingError`          | File `.pkl` có thể bị corrupted, chạy lại preprocessing               |
| `KeyError` khi xử lý JSON         | Một số file JSON có cấu trúc không chuẩn, script tự skip chúng        |

## 📄 License

Dự án được cấp phép theo MIT License - xem [LICENSE](LICENSE) để biết chi tiết.

## 📖 References

- **PHEME Dataset**:  [All-in-one: Multi-task Learning for Rumour Verification](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)
- **Tree Data Structures**: Cormen et al., "Introduction to Algorithms" (Chapter 10-13)
- **Graph/Tree Traversals**: DFS & BFS algorithms

## 👥 Contributors (Team 4lg0r1thm)

- **Lil'Tommy**
- **Loc**
- **Thien Khanh**
