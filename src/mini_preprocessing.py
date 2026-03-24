from utils import TreeNode, build_thread_tree, ensure_dir
import os
import json
import pickle
import sys

# Đảm bảo Python tìm thấy file utils.py trong thư mục src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def convert_annotations(annotation):
    """
    Hàm chuyển đổi annotation của PHEME sang nhãn văn bản.
    """
    if 'misinformation' in annotation and 'true' in annotation:
        m = int(annotation['misinformation'])
        t = int(annotation['true'])
        if m == 0 and t == 0:
            return "unverified"
        if m == 0 and t == 1:
            return "true"
        if m == 1 and t == 0:
            return "false"
    elif 'misinformation' in annotation:
        return "false" if int(annotation['misinformation']) == 1 else "unverified"
    return "unverified"


def preprocess_text(text: str) -> str:
    """Làm sạch văn bản cơ bản (lowercase, xóa xuống dòng)."""
    if not text:
        return ""
    return text.lower().replace("\n", " ").strip()


def load_pheme_pipeline(data_dir: str):
    """
    Pipeline: Đọc raw -> Trích xuất -> Dán nhãn -> Trả về list dict thu gọn.
    """
    all_threads_data = []

    # Lấy danh sách các sự kiện (ebola, ferguson, etc.)
    events = [e for e in os.listdir(data_dir) if not e.startswith(
        '.') and os.path.isdir(os.path.join(data_dir, e))]

    for event in events:
        print(f"📂 Processing event: {event}...")
        # PHEME có 2 thư mục con: rumours và non-rumours
        for folder_type in ['rumours', 'non-rumours']:
            label_path = os.path.join(data_dir, event, folder_type)
            if not os.path.exists(label_path):
                continue

            thread_ids = [t for t in os.listdir(
                label_path) if not t.startswith('.')]

            for tid in thread_ids:
                t_path = os.path.join(label_path, tid)

                try:
                    # 1. Xác định nhãn (Label)
                    if folder_type == 'non-rumours':
                        label = "true"
                    else:
                        anno_file = os.path.join(t_path, 'annotation.json')
                        with open(anno_file, 'r', encoding='utf-8') as f:
                            label = convert_annotations(json.load(f))

                    # 2. Đọc Tweet gốc (Source)
                    src_dir = os.path.join(t_path, 'source-tweets')
                    src_files = [f for f in os.listdir(src_dir) if f.endswith(
                        '.json') and not f.startswith('.')]
                    with open(os.path.join(src_dir, src_files[0]), 'r', encoding='utf-8') as f:
                        s = json.load(f)

                    compact_thread = {
                        "source": {
                            "id_str": s['id_str'],
                            "text": preprocess_text(s['text']),
                            "user": s['user']['screen_name'],
                            "created_at": s['created_at']
                        },
                        "label": label,
                        "reactions": []
                    }

                    # 3. Đọc các phản hồi (Reactions)
                    reac_dir = os.path.join(t_path, 'reactions')
                    if os.path.exists(reac_dir):
                        for rf in os.listdir(reac_dir):
                            if not rf.endswith('.json') or rf.startswith('.'):
                                continue
                            with open(os.path.join(reac_dir, rf), 'r', encoding='utf-8') as f:
                                r = json.load(f)
                                compact_thread["reactions"].append({
                                    "id_str": r['id_str'],
                                    "text": preprocess_text(r['text']),
                                    "user": r['user']['screen_name'],
                                    "in_reply_to_status_id_str": r.get('in_reply_to_status_id_str'),
                                    "created_at": r['created_at']
                                })

                    all_threads_data.append(compact_thread)
                except Exception as e:
                    # Bỏ qua nếu folder thiếu file hoặc lỗi định dạng
                    continue

    return all_threads_data


if __name__ == "__main__":
    # 1. Cấu hình đường dẫn
    RAW_DATA_DIR = "../data/raw/all-rnr-annotated-threads"
    PROCESSED_DIR = "../data/processed"
    ensure_dir(PROCESSED_DIR)

    print("--- 🚀 STARTING PREPROCESSING PIPELINE ---")

    # 2. Chạy pipeline load dữ liệu thu gọn
    raw_threads = load_pheme_pipeline(RAW_DATA_DIR)
    print(f"✅ Loaded {len(raw_threads)} threads from raw files.")

    # 3. Chuyển đổi sang đối tượng TreeNode (DSA Structure)
    print("🌳 Building Tree objects with labels...")
    tree_objects = [build_thread_tree(t) for t in raw_threads]

    # 4. Lưu trữ
    # Lưu JSON thu gọn (để kiểm tra mắt thường)
    with open(os.path.join(PROCESSED_DIR, "threads_compact.json"), 'w', encoding='utf-8') as f:
        json.dump(raw_threads, f, indent=4, ensure_ascii=False)

    # Lưu Pickle (Lưu thẳng List các Object TreeNode có thuộc tính .label)
    with open(os.path.join(PROCESSED_DIR, "trees_data.pkl"), 'wb') as f:
        pickle.dump(tree_objects, f)

    print(f"💾 Success! Processed data saved to {PROCESSED_DIR}")
    print(f"💡 Final Tree Count: {len(tree_objects)}")
