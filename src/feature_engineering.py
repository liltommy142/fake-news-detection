import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import Dict, List, Any
from utils import TreeNode

# Download NLTK data if needed
nltk.download('vader_lexicon')

def extract_text_features(text: str) -> Dict[str, float]:
    """
    Extract text-based features.

    Args:
        text: Preprocessed text.

    Returns:
        Dictionary of features.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)

    features = {
        'sentiment_neg': sentiment['neg'],
        'sentiment_neu': sentiment['neu'],
        'sentiment_pos': sentiment['pos'],
        'sentiment_compound': sentiment['compound'],
        'text_length': len(text),
        'word_count': len(text.split()),
    }
    return features

def extract_structural_features(tree: TreeNode) -> Dict[str, float]:
    """
    Extract structural features from the thread tree using DSA.

    Args:
        tree: Root TreeNode of the thread.

    Returns:
        Dictionary of features.
    """
    depth = tree.get_depth()
    size = tree.get_size()
    breadth = len(tree.children)

    # Additional DSA-based features
    preorder_traversal = tree.traverse_preorder()
    inorder_traversal = tree.traverse_inorder()

    features = {
        'tree_depth': depth,
        'tree_size': size,
        'tree_breadth': breadth,
        'preorder_length': len(preorder_traversal),
        'inorder_length': len(inorder_traversal),
    }
    return features

def vectorize_text(texts: List[str]) -> np.ndarray:
    """
    Vectorize texts using TF-IDF.

    Args:
        texts: List of texts.

    Returns:
        TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    return vectorizer.fit_transform(texts).toarray()

if __name__ == "__main__":
    # Example usage
    sample_text = "This is a sample tweet."
    features = extract_text_features(sample_text)
    print(features)