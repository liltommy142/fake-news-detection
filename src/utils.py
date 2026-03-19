import os
import matplotlib.pyplot as plt

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