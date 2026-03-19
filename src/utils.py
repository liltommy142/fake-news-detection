import os
import matplotlib.pyplot as plt
import numpy as np

class TreeNode:
    """
    Tree node for representing Twitter thread structure.
    """
    def __init__(self, tweet_id: str, text: str, user: str, timestamp: str):
        self.tweet_id = tweet_id
        self.text = text
        self.user = user
        self.timestamp = timestamp
        self.children = []  # List of replies (TreeNode)

    def add_child(self, child: 'TreeNode'):
        """Add a reply to this tweet."""
        self.children.append(child)

    def get_depth(self) -> int:
        """Calculate the depth of the subtree rooted at this node."""
        if not self.children:
            return 1
        return 1 + max(child.get_depth() for child in self.children)

    def get_size(self) -> int:
        """Calculate the number of nodes in the subtree."""
        return 1 + sum(child.get_size() for child in self.children)

    def traverse_inorder(self) -> list:
        """Inorder traversal of the tree."""
        result = []
        for child in self.children:
            result.extend(child.traverse_inorder())
        result.append(self.tweet_id)
        return result

    def traverse_preorder(self) -> list:
        """Preorder traversal of the tree."""
        result = [self.tweet_id]
        for child in self.children:
            result.extend(child.traverse_preorder())
        return result

def build_thread_tree(thread_data: dict) -> TreeNode:
    """
    Build a tree from thread data using DSA tree structure.

    Args:
        thread_data: Dictionary containing thread information.

    Returns:
        Root TreeNode of the thread.
    """
    # Assuming thread_data has 'root' and 'replies' structure
    root_data = thread_data.get('root', {})
    root = TreeNode(
        tweet_id=root_data.get('id', ''),
        text=root_data.get('text', ''),
        user=root_data.get('user', ''),
        timestamp=root_data.get('timestamp', '')
    )

    replies = thread_data.get('replies', [])
    for reply in replies:
        child = build_reply_tree(reply)
        root.add_child(child)

    return root

def build_reply_tree(reply_data: dict) -> TreeNode:
    """
    Recursively build tree from reply data.

    Args:
        reply_data: Reply data.

    Returns:
        TreeNode.
    """
    node = TreeNode(
        tweet_id=reply_data.get('id', ''),
        text=reply_data.get('text', ''),
        user=reply_data.get('user', ''),
        timestamp=reply_data.get('timestamp', '')
    )
    for sub_reply in reply_data.get('replies', []):
        child = build_reply_tree(sub_reply)
        node.add_child(child)
    return node

def ensure_dir(path: str):
    """
    Ensure directory exists.

    Args:
        path: Directory path.
    """
    os.makedirs(path, exist_ok=True)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix.
        classes: Class labels.
        title: Plot title.
    """
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("../results/figures/confusion_matrix.png")
    plt.show()

# Add more utilities as needed