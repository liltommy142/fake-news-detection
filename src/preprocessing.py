import os
import json
import pandas as pd
from typing import List, Dict, Any
from utils import TreeNode, build_thread_tree

def load_pheme_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load PHEME dataset from JSON files.

    Args:
        data_dir: Directory containing the raw PHEME data.

    Returns:
        List of conversation threads.
    """
    threads = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    thread = json.load(f)
                    threads.append(thread)
    return threads

def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing.

    Args:
        text: Raw text.

    Returns:
        Preprocessed text.
    """
    # Add more preprocessing steps as needed (lowercasing, removing URLs, etc.)
    return text.lower().strip()

def build_thread_tree(thread: Dict[str, Any]) -> TreeNode:
    """
    Build a tree structure from the thread data using DSA TreeNode.

    Args:
        thread: Thread data from JSON.

    Returns:
        Root TreeNode of the thread.
    """
    return build_thread_tree(thread)  # Use the function from utils

if __name__ == "__main__":
    data_dir = "../data/raw"
    threads = load_pheme_data(data_dir)
    print(f"Loaded {len(threads)} threads")
    # Build trees
    trees = [build_thread_tree(thread) for thread in threads]
    print(f"Built {len(trees)} thread trees")
    # Save processed data
    processed_dir = "../data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    with open(os.path.join(processed_dir, "threads.json"), 'w') as f:
        json.dump(threads, f)