import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List


class TreeNode:
    """
    Cấu trúc dữ liệu Cây (Tree) để đại diện cho một luồng hội thoại Twitter.
    """

    def __init__(self, tweet_id: str, text: str, user: str, timestamp: str, label: str = "unknown"):
        self.tweet_id = tweet_id
        self.text = text
        self.user = user
        self.timestamp = timestamp
        self.label = label  # Nhãn của toàn bộ luồng (true, false, unverified)
        self.children: List['TreeNode'] = []

    def add_child(self, child: 'TreeNode'):
        """Thêm một node con (phản hồi) vào node hiện tại."""
        self.children.append(child)

    def get_depth(self) -> int:
        """Tính độ sâu của cây bằng giải thuật đệ quy."""
        if not self.children:
            return 1
        return 1 + max(child.get_depth() for child in self.children)

    def get_size(self) -> int:
        """Tính tổng số node trong cây (bao gồm cả gốc và tất cả phản hồi)."""
        return 1 + sum(child.get_size() for child in self.children)

    def display(self, level=0, prefix="Root: "):
        """Trực quan hóa cấu trúc cây dưới dạng ASCII (Duyệt DFS)."""
        indent = "    " * level
        # Rút gọn nội dung tweet để in không bị quá dài
        display_text = (self.text[:60] +
                        '...') if len(self.text) > 60 else self.text
        print(
            f"{indent}{prefix}[ID: {self.tweet_id}] @{self.user}: {display_text}")

        for i, child in enumerate(self.children):
            # Tạo ký hiệu nhánh cây cho đẹp
            connector = "└── " if i == len(self.children) - 1 else "├── "
            child.display(level + 1, prefix=connector)

    def traverse_preorder(self) -> list:
        """Duyệt tiền thứ tự (Pre-order Traversal)."""
        result = [self.tweet_id]
        for child in self.children:
            result.extend(child.traverse_preorder())
        return result


def build_thread_tree(thread_data: Dict[str, Any]) -> TreeNode:
    """
    Giải thuật dựng cây từ danh sách phẳng (Flat List) sử dụng Hash Map.
    Độ phức tạp: O(n) - Tối ưu nhất cho DSA.
    """
    source = thread_data.get('source', {})
    reactions = thread_data.get('reactions', [])
    label = thread_data.get('label', 'unknown')

    # 1. Tạo node gốc (Source Tweet)
    root = TreeNode(
        tweet_id=source.get('id_str', ''),
        text=source.get('text', ''),
        user=source.get('user', ''),
        timestamp=source.get('created_at', ''),
        label=label
    )

    # 2. Khởi tạo Hash Map (node_map) để truy xuất Node nhanh theo ID
    node_map = {root.tweet_id: root}

    # Tạo trước tất cả các đối tượng TreeNode cho phần phản hồi
    for r in reactions:
        tid = r.get('id_str', '')
        node_map[tid] = TreeNode(
            tweet_id=tid,
            text=r.get('text', ''),
            user=r.get('user', 'unknown'),
            timestamp=r.get('created_at', ''),
            label=label
        )

    # 3. Duyệt lại lần nữa để nối "Cha-Con" dựa trên parent_id
    for r in reactions:
        tid = r.get('id_str', '')
        pid = r.get('in_reply_to_status_id_str')

        current_node = node_map[tid]

        # Nếu tìm thấy ID cha trong map thì gắn vào,
        # nếu không (do tweet cha bị xóa/lỗi) thì gắn trực tiếp vào Root
        if pid and pid in node_map:
            node_map[pid].add_child(current_node)
        else:
            root.add_child(current_node)

    return root


def ensure_dir(path: str):
    """Đảm bảo thư mục tồn tại."""
    os.makedirs(path, exist_ok=True)


def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """Vẽ biểu đồ Confusion Matrix cho báo cáo."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    save_path = "../results/figures/confusion_matrix.png"
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.show()
