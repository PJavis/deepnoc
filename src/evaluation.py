"""
Evaluation utilities: confusion matrix, accuracy plots, threshold analysis.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support,
)


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix",
                          save_path=None, normalize=False):
    """Plot and optionally save a confusion matrix."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    display_cm = cm_norm if normalize else cm
    fmt = '.2f' if normalize else 'd'
    
    sns.heatmap(display_cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Known NoC')
    ax.set_ylabel('Predicted NoC')
    ax.set_title(title)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return cm


def plot_training_history(history, title="Training History", save_path=None):
    """Plot training/test accuracy and loss over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(len(history['train_acc']))
    
    # Accuracy
    ax1.plot(epochs, history['train_acc'], 'r-', label='Training', alpha=0.7)
    ax1.plot(epochs, history['test_acc'], 'k-', label='Test', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(epochs, history['train_loss'], 'r-', label='Training', alpha=0.7)
    ax2.plot(epochs, history['test_loss'], 'k-', label='Test', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def compute_per_class_metrics(y_true, y_pred, class_labels=None):
    """
    Compute accuracy, precision, recall, F1 per class.
    Reproduces Table 2 from the paper.
    """
    if class_labels is None:
        class_labels = sorted(set(y_true) | set(y_pred))
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, zero_division=0
    )
    
    results = {}
    for i, label in enumerate(class_labels):
        mask = y_true == label
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
        else:
            acc = 0.0
        
        results[label] = {
            'accuracy': acc,
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i]),
        }
    
    # Overall
    results['overall'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'total': len(y_true),
    }
    
    return results


def print_results_table(metrics, title="Results"):
    """Print metrics in a formatted table (similar to Table 2 in paper)."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"{'NoC':>6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'N':>6}")
    print(f"{'-'*60}")
    
    for key, vals in metrics.items():
        if key == 'overall':
            continue
        print(f"{key:>6} {vals['accuracy']:>10.3f} {vals['precision']:>10.3f} "
              f"{vals['recall']:>10.3f} {vals['f1']:>10.3f} {vals['support']:>6d}")
    
    if 'overall' in metrics:
        print(f"{'-'*60}")
        print(f"{'ALL':>6} {metrics['overall']['accuracy']:>10.3f} "
              f"{'':>10} {'':>10} {'':>10} {metrics['overall']['total']:>6d}")
    print(f"{'='*60}")


def plot_threshold_analysis(y_true, y_probs, save_path=None):
    """
    Reproduce Figure 7: accuracy vs proportion classified at different thresholds.
    
    Args:
        y_true: [N] true labels (1-indexed)
        y_probs: [N, num_classes] probability matrix
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    accuracies = []
    proportions = []
    
    y_pred = y_probs.argmax(axis=1) + 1  # 1-indexed
    max_probs = y_probs.max(axis=1)
    
    for thresh in thresholds:
        mask = max_probs >= thresh
        n_classified = mask.sum()
        
        if n_classified > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            prop = n_classified / len(y_true)
        else:
            acc = 1.0  # No misclassifications if nothing classified
            prop = 0.0
        
        accuracies.append(acc)
        proportions.append(prop)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, accuracies, 'k-', label='Correctly assigned', linewidth=2)
    ax.plot(thresholds, proportions, 'r-', label='Classified', linewidth=2)
    ax.set_xlabel('Probability threshold')
    ax.set_ylabel('Proportion')
    ax.set_title('Assignment Accuracy vs Classification Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.05)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def full_evaluation(y_true, y_pred, y_probs=None, class_labels=None,
                    title="Model", save_dir="results"):
    """Run complete evaluation and save all plots/metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    if class_labels is None:
        class_labels = sorted(set(y_true))
    
    # Confusion matrix
    cm = plot_confusion_matrix(
        y_true, y_pred, labels=class_labels,
        title=f'Confusion Matrix - {title}',
        save_path=os.path.join(save_dir, f'confusion_matrix_{title.lower().replace(" ", "_")}.png'),
    )
    
    # Per-class metrics
    metrics = compute_per_class_metrics(y_true, y_pred, class_labels)
    print_results_table(metrics, title)
    
    # Threshold analysis (if probabilities available)
    if y_probs is not None:
        plot_threshold_analysis(
            y_true, y_probs,
            save_path=os.path.join(save_dir, f'threshold_{title.lower().replace(" ", "_")}.png'),
        )
    
    # Save metrics as JSON
    metrics_json = {str(k): v for k, v in metrics.items()}
    with open(os.path.join(save_dir, f'metrics_{title.lower().replace(" ", "_")}.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    return metrics, cm