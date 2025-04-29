import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report
import seaborn as sns

def visualise_results(y_test, y_preds, encoder, param):
    """
    y-test: encoded labels
    y_preds: predicted labels (column in dataframe)
    encoder: encoder to decode labels
    """
    cfm_raw = confusion_matrix(y_test, y_preds)
    cfm_norm = confusion_matrix(y_test, y_preds, normalize='true')
    bal_acc = balanced_accuracy_score(y_test, y_preds)

    fig = plt.figure(figsize=(5, 5))
    sns.heatmap(cfm_norm, fmt="d", annot=cfm_raw, cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_, norm=Normalize(0, 1))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{param} | bal. acc. = {bal_acc:.3f}')
    return fig

def print_classification_report(y_test, y_preds, encoder):
    """
    Print classification report for the model predictions.
    """
    return classification_report(y_test, y_preds, target_names=encoder.classes_)

def visualise_results_with_report(y_test, y_preds, encoder, param):
    cfm_raw = confusion_matrix(y_test, y_preds)
    cfm_norm = confusion_matrix(y_test, y_preds, normalize='true')
    bal_acc = balanced_accuracy_score(y_test, y_preds)
    class_report = classification_report(y_test, y_preds, target_names=encoder.classes_)

    # Create two vertically stacked subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 8), 
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Plot the heatmap
    sns.heatmap(
        cfm_norm, 
        annot=cfm_raw, 
        fmt="d", 
        cmap='Blues', 
        xticklabels=encoder.classes_, 
        yticklabels=encoder.classes_,
        ax=ax1
    )
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title(f'{param} | bal. acc. = {bal_acc:.3f}')

    # Add classification report as text in the second subplot
    ax2.axis('off')  # Hide the axis
    ax2.text(
        0.01, 0.99, class_report, 
        fontsize=10, 
        va='top', 
        ha='left', 
        family='monospace'
    )

    plt.tight_layout()
    return fig