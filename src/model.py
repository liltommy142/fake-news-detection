import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def load_features_and_labels() -> tuple:
    """
    Load features and labels for training.

    Returns:
        Tuple of (X, y)
    """
    # Placeholder: Load from processed data
    # Assume features are in a CSV or similar
    # For now, dummy data
    np.random.seed(42)
    X = np.random.rand(100, 10)  # Dummy features
    y = np.random.randint(0, 2, 100)  # Dummy labels (0: real, 1: fake)
    return X, y

def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """
    Train the classification model.

    Args:
        X: Feature matrix.
        y: Labels.

    Returns:
        Trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Save metrics
    with open("../results/metrics.txt", 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")

    return model

def save_model(model: RandomForestClassifier, path: str):
    """
    Save the trained model.

    Args:
        model: Trained model.
        path: Save path.
    """
    joblib.dump(model, path)

if __name__ == "__main__":
    X, y = load_features_and_labels()
    model = train_model(X, y)
    save_model(model, "../results/model.pkl")