import os
import sys
import json
import pickle
import argparse
from typing import List, Dict, Any

# Đảm bảo Python tìm thấy utils.py khi chạy từ root hoặc từ thư mục src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils import TreeNode, build_thread_tree, ensure_dir  # type: ignore

# ====================== UTILITIES ======================

def convert_annotations(annotation: Dict[str, Any], string: bool = True):
    """
    Chuyển file annotation.json thành nhãn:
      - true / false / unverified
    """
    if 'misinformation' in annotation and 'true' in annotation:
        m = int(annotation['misinformation'])
        t = int(annotation['true'])
        if m == 0 and t == 0:
            label = "unverified" if string else 2
        elif m == 0 and t == 1:
            label = "true" if string else 1
        elif m == 1 and t == 0:
            label = "false" if string else 0
        else:
            label = "unverified"
    elif 'misinformation' in annotation:
        m = int(annotation['misinformation'])
        label = "unverified" if m == 0 else "false"
    else:
        label = "unverified"
    return label


def preprocess_text(text: str) -> str:
    """
    Làm sạch văn bản:
      - lower
      - bỏ xuống dòng
      - bỏ link
    """
    if not text:
        return ""
    import re

    text = text.lower().replace("\n", " ")
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    return text.strip()


# ====================== CORE LOADER ======================

def load_pheme_threads(
    data_dir: str,
    max_events: int | None = None,
    max_threads_per_event: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Đọc dataset PHEME và trả về list các thread thu gọn.
    Có thể giới hạn:
      - max_events: số event tối đa (None = full)
      - max_threads_per_event: số thread / event (None = full)
    """
    all_threads: List[Dict[str, Any]] = []

    if not os.path.exists(data_dir):
        print(f"❌ Không tìm thấy thư mục dữ liệu: {data_dir}")
        return []

    # Lấy danh sách event (ebola, ferguson, ...)
    events = [
        e for e in os.listdir(data_dir)
        if not e.startswith(".") and os.path.isdir(os.path.join(data_dir, e))
    ]
    events.sort()
    if max_events is not None:
        events = events[:max_events]

    print(f"📂 Tìm thấy {len(events)} event trong PHEME.")
    if max_events is not None:
        print(
            f"   → Đang giới hạn còn {len(events)} event (max_events={max_events}).")

    for event_idx, event in enumerate(events, start=1):
        event_path = os.path.join(data_dir, event)
        print(f"\n📁 [{event_idx}/{len(events)}] Processing event: {event}")

        # rumours / non-rumours
        for folder_type in ["rumours", "non-rumours"]:
            label_path = os.path.join(event_path, folder_type)
            if not os.path.exists(label_path):
                continue

            thread_ids = [
                t for t in os.listdir(label_path)
                if not t.startswith(".") and os.path.isdir(os.path.join(label_path, t))
            ]
            thread_ids.sort()

            threads_processed = 0

            for tid in thread_ids:
                if max_threads_per_event is not None and \
                        threads_processed >= max_threads_per_event:
                    break

                t_path = os.path.join(label_path, tid)

                try:
                    # 1. Nhãn
                    if folder_type == "non-rumours":
                        label = "true"
                    else:
                        anno_file = os.path.join(t_path, "annotation.json")
                        if os.path.exists(anno_file):
                            with open(anno_file, "r", encoding="utf-8") as f:
                                label = convert_annotations(json.load(f))
                        else:
                            label = "unverified"

                    # 2. Source tweet
                    src_dir = os.path.join(t_path, "source-tweets")
                    if not os.path.exists(src_dir):
                        continue
                    src_files = [
                        f for f in os.listdir(src_dir)
                        if f.endswith(".json") and not f.startswith(".")
                    ]
                    if not src_files:
                        continue

                    with open(
                        os.path.join(src_dir, src_files[0]),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        s = json.load(f)

                    compact_thread = {
                        "source": {
                            "id_str": s["id_str"],
                            "text": preprocess_text(s.get("text", "")),
                            "user": s["user"]["screen_name"],
                            "created_at": s["created_at"],
                        },
                        "label": label,
                        "reactions": [],
                    }

                    # 3. Reactions
                    reac_dir = os.path.join(t_path, "reactions")
                    if os.path.exists(reac_dir):
                        for rf in os.listdir(reac_dir):
                            if not rf.endswith(".json") or rf.startswith("."):
                                continue
                            with open(
                                os.path.join(reac_dir, rf),
                                "r",
                                encoding="utf-8",
                            ) as f:
                                r = json.load(f)
                            compact_thread["reactions"].append(
                                {
                                    "id_str": r["id_str"],
                                    "text": preprocess_text(r.get("text", "")),
                                    "user": r["user"]["screen_name"],
                                    "in_reply_to_status_id_str": r.get(
                                        "in_reply_to_status_id_str"
                                    ),
                                    "created_at": r["created_at"],
                                }
                            )

                    all_threads.append(compact_thread)
                    threads_processed += 1

                except Exception:
                    # Skip thread lỗi cấu trúc / thiếu file
                    continue

            print(
                f"   - {folder_type}: {threads_processed} threads "
                f"(limit={max_threads_per_event or '∞'})"
            )

    print(f"\n✅ Tổng số thread thu được: {len(all_threads)}")
    return all_threads


# ====================== MAIN PIPELINE ======================

def main():
    parser = argparse.ArgumentParser(
        description="PHEME preprocessing: build trees & compact threads"
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/all-rnr-annotated-threads",
        help="Thư mục chứa PHEME (mặc định: ../data/raw/all-rnr-annotated-threads)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Thư mục output (mặc định: ../data/processed)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Giới hạn số event (None = full dataset)",
    )
    parser.add_argument(
        "--max-threads-per-event",
        type=int,
        default=None,
        help="Giới hạn số thread mỗi event (None = full)",
    )

    args = parser.parse_args()

    raw_data_dir = args.data_dir
    processed_dir = args.out_dir
    max_events = args.max_events
    max_threads = args.max_threads_per_event

    print("=== 🚀 PREPROCESSING PIPELINE START ===")
    print(f"📁 RAW_DATA_DIR  = {raw_data_dir}")
    print(f"💾 PROCESSED_DIR = {processed_dir}")
    print(f"⚙️  max_events           = {max_events}")
    print(f"⚙️  max_threads_per_event = {max_threads}")

    ensure_dir(processed_dir)

    # 1. Load & compact
    raw_threads = load_pheme_threads(
        raw_data_dir,
        max_events=max_events,
        max_threads_per_event=max_threads,
    )
    print(f"\n📊 Loaded {len(raw_threads)} compact threads.")

    # 2. Build trees
    print("🌳 Building Tree objects with labels...")
    trees: List[TreeNode] = [build_thread_tree(t) for t in raw_threads]
    print(f"   ✓ Built {len(trees)} trees.")

    # 3. Save
    threads_json_path = os.path.join(processed_dir, "threads_compact.json")
    trees_pkl_path = os.path.join(processed_dir, "trees_data.pkl")

    with open(threads_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_threads, f, indent=4, ensure_ascii=False)

    with open(trees_pkl_path, "wb") as f:
        pickle.dump(trees, f)

    print(f"\n💾 Saved JSON  → {threads_json_path}")
    print(f"💾 Saved PKL   → {trees_pkl_path}")
    print(f"💡 Final Tree Count: {len(trees)}")
    print("=== ✅ PREPROCESSING PIPELINE DONE ===")


if __name__ == "__main__":
    main()
