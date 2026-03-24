import os
import json
import pickle
from typing import List, Dict, Any
# Đảm bảo utils.py nằm cùng thư mục hoặc trong PYTHONPATH
from utils import TreeNode, build_thread_tree


def convert_annotations(annotation, string=True):
    """Chuyển đổi file annotation.json thành nhãn văn bản."""
    if 'misinformation' in annotation.keys() and 'true' in annotation.keys():
        if int(annotation['misinformation']) == 0 and int(annotation['true']) == 0:
            label = "unverified" if string else 2
        elif int(annotation['misinformation']) == 0 and int(annotation['true']) == 1:
            label = "true" if string else 1
        elif int(annotation['misinformation']) == 1 and int(annotation['true']) == 0:
            label = "false" if string else 0
        else:
            label = "unverified"  # Trường hợp cả hai bằng 1 (hiếm)
    elif 'misinformation' in annotation.keys():
        label = "unverified" if int(
            annotation['misinformation']) == 0 else "false"
    else:
        label = "unverified"
    return label


def load_pheme_compact(data_dir: str) -> List[Dict[str, Any]]:
    """Tải dữ liệu PHEME và thu gọn để giảm dung lượng."""
    threads = []
    if not os.path.exists(data_dir):
        print(f"Lỗi: Không tìm thấy thư mục {data_dir}")
        return []

    for event in os.listdir(data_dir):
        event_path = os.path.join(data_dir, event)
        if event.startswith('.') or not os.path.isdir(event_path):
            continue

        # PHEME chia làm rumours và non-rumours
        for folder_type in ['rumours', 'non-rumours']:
            label_path = os.path.join(event_path, folder_type)
            if not os.path.exists(label_path):
                continue

            for thread_id in os.listdir(label_path):
                if thread_id.startswith('.'):
                    continue

                thread_full_path = os.path.join(label_path, thread_id)

                # 1. Đọc nhãn (Nếu là non-rumours thì mặc định là true, nếu là rumours thì đọc file)
                label = "true" if folder_type == 'non-rumours' else "unknown"
                anno_path = os.path.join(thread_full_path, 'annotation.json')
                if folder_type == 'rumours' and os.path.exists(anno_path):
                    with open(anno_path, 'r', encoding='utf-8') as f:
                        label = convert_annotations(json.load(f))

                # 2. Đọc Tweet gốc
                src_dir = os.path.join(thread_full_path, 'source-tweets')
                if not os.path.exists(src_dir):
                    continue
                src_files = [f for f in os.listdir(src_dir) if f.endswith(
                    '.json') and not f.startswith('.')]
                if not src_files:
                    continue

                with open(os.path.join(src_dir, src_files[0]), 'r', encoding='utf-8') as f:
                    src_data = json.load(f)

                compact_thread = {
                    "source": {
                        "id_str": src_data['id_str'],
                        "text": preprocess_text(src_data['text']),
                        "user": src_data['user']['screen_name'],
                        "created_at": src_data['created_at']
                    },
                    "label": label,
                    "reactions": []
                }

                # 3. Đọc phản hồi
                react_dir = os.path.join(thread_full_path, 'reactions')
                if os.path.exists(react_dir):
                    for r_file in os.listdir(react_dir):
                        if not r_file.endswith('.json') or r_file.startswith('.'):
                            continue
                        with open(os.path.join(react_dir, r_file), 'r', encoding='utf-8') as f:
                            r_data = json.load(f)
                            compact_thread["reactions"].append({
                                "id_str": r_data['id_str'],
                                "text": preprocess_text(r_data['text']),
                                "user": r_data['user']['screen_name'],
                                "in_reply_to_status_id_str": r_data.get('in_reply_to_status_id_str'),
                                "created_at": r_data['created_at']
                            })
                threads.append(compact_thread)
    return threads


def preprocess_text(text: str) -> str:
    """Làm sạch văn bản cơ bản."""
    import re
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text,
                  flags=re.MULTILINE)  # Xóa link cho nhẹ
    return text.strip()


if __name__ == "__main__":
    # Cập nhật đường dẫn chuẩn của bạn
    data_dir = "../data/raw/all-rnr-annotated-threads"

    print("--- Bắt đầu pipeline xử lý dữ liệu ---")

    # 1. Load và nén dữ liệu
    raw_threads = load_pheme_compact(data_dir)
    print(f"1. Đã tải {len(raw_threads)} luồng hội thoại.")

    # 2. Xây dựng cây (Sử dụng hàm từ utils)
    print("2. Đang xây dựng cấu trúc cây DSA...")
    trees = [build_thread_tree(t) for t in raw_threads]
    print(f"   Hoàn thành dựng {len(trees)} cây.")

    # 3. Lưu dữ liệu
    processed_dir = "../data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Lưu bản JSON thu gọn (để đọc mắt thường)
    with open(os.path.join(processed_dir, "threads_compact.json"), 'w', encoding='utf-8') as f:
        json.dump(raw_threads, f, ensure_ascii=False, indent=4)

    # Lưu bản Pickle (Lưu cả Object TreeNode - cực nhanh và nhẹ cho DSA)
    with open(os.path.join(processed_dir, "trees_data.pkl"), 'wb') as f:
        pickle.dump(trees, f)

    print(f"3. Đã lưu dữ liệu tại {processed_dir}")
    print("--- Pipeline hoàn tất! ---")
