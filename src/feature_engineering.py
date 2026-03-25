import os
import pickle
import pandas as pd
from collections import deque
import sys

# Ensure Python finds utils.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import TreeNode

def extract_bfs_metrics(root: TreeNode):
    """
    Uses Breadth-First Search (BFS) to analyze the propagation tree.
    Time Complexity: O(V + E) where V is tweets and E is replies.
    Space Complexity: O(W) where W is the maximum width of the tree.
    """
    if not root:
        return 0, 0

    max_width = 0
    total_nodes = 0
    leaf_nodes = 0
    
    # Standard BFS Queue
    queue = deque([root])
    
    while queue:
        level_width = len(queue)
        max_width = max(max_width, level_width)
        
        for _ in range(level_width):
            current_node = queue.popleft()
            total_nodes += 1
            
            if not current_node.children:
                leaf_nodes += 1
            else:
                for child in current_node.children:
                    queue.append(child)
                    
    # Average branching factor (how many replies a tweet gets on average)
    # E / V where E = total_nodes - 1
    avg_branching = (total_nodes - 1) / (total_nodes - leaf_nodes) if (total_nodes - leaf_nodes) > 0 else 0
                    
    return max_width, avg_branching, leaf_nodes

def extract_text_heuristics(root: TreeNode):
    """
    Baseline text feature extraction. O(L) where L is text length.
    """
    sensational_words = {'urgent', 'breaking', 'shocking', 'wtf', 'omg', 'fake'}
    text = root.text.lower()
    
    word_count = len(text.split())
    sensational_count = sum(1 for word in sensational_words if word in text)
    
    has_question_mark = 1 if '?' in text else 0
    has_exclamation = 1 if '!' in text else 0
    
    return word_count, sensational_count, has_question_mark, has_exclamation

if __name__ == "__main__":
    PROCESSED_DIR = "../data/processed"
    TREE_FILE = os.path.join(PROCESSED_DIR, "trees_data.pkl")
    
    print("--- 🚀 STARTING FEATURE ENGINEERING ---")
    
    # 1. Load the Trees built by preprocessing.py
    if not os.path.exists(TREE_FILE):
        print("❌ Error: trees_data.pkl not found. Run preprocessing.py first.")
        sys.exit(1)
        
    with open(TREE_FILE, 'rb') as f:
        trees = pickle.load(f)
        
    print(f"✅ Loaded {len(trees)} trees for analysis.")
    
    # 2. Extract Features
    features_list = []
    
    for tree in trees:
        # Skip unverified data to focus purely on binary classification (True/False)
        if tree.label == 'unverified':
            continue 
            
        # Graph Features (DSA focused)
        depth = tree.get_depth()  # From your teammate's utils.py
        size = tree.get_size()    # From your teammate's utils.py
        max_width, avg_branching, leaves = extract_bfs_metrics(tree)
        
        # Text Features (Baseline)
        words, sens_count, has_q, has_exc = extract_text_heuristics(tree)
        
        # Convert label to binary: 1 for false (rumor), 0 for true (non-rumor)
        binary_label = 1 if tree.label == 'false' else 0
        
        features_list.append({
            'tweet_id': tree.tweet_id,
            'tree_depth': depth,
            'tree_size': size,
            'max_width': max_width,
            'avg_branching_factor': avg_branching,
            'leaf_count': leaves,
            'word_count': words,
            'sensational_word_count': sens_count,
            'has_question_mark': has_q,
            'has_exclamation': has_exc,
            'label': binary_label
        })

    # 3. Save to a format the Machine Learning model can easily read
    df = pd.DataFrame(features_list)
    output_path = os.path.join(PROCESSED_DIR, "extracted_features.csv")
    df.to_csv(output_path, index=False)
    
    print(f"💾 Success! Extracted features for {len(df)} threads.")
    print(f"📊 File saved to {output_path}")