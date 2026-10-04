"""
Evaluation module.
Computes metrics, plots confusion matrices, and compares baselines.
"""

import logging
import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict[str, Any]:
    """
    Compute various classification metrics.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        labels (List[int]): List of unique label classes.

    Returns:
        Dict[str, Any]: Computed metrics dictionary.
    """
    acc = accuracy_score(y_true, y_pred)
    
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_m = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    
    per_class_prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    
    per_class = {}
    for i, label in enumerate(labels):
        per_class[str(label)] = {
            "precision": float(per_class_prec[i]),
            "recall": float(per_class_rec[i]),
            "f1": float(per_class_f1[i])
        }
        
    return {
        "accuracy": float(acc),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
        "per_class": per_class,
        "confusion_matrix": cm
    }

def plot_confusion_matrix(cm: List[List[int]], labels: List[str], output_path: str) -> None:
    """
    Plot and save confusion matrix.

    Args:
        cm (List[List[int]]): Confusion matrix.
        labels (List[str]): List of class names.
        output_path (str): File path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        logging.info(f"Saved confusion matrix to {output_path}")
    except IOError as e:
        logging.error(f"Failed to save plot to {output_path}: {e}")
    plt.close()

def plot_feature_importance(model: Any, feature_names: List[str], output_path: str) -> None:
    """
    Plot and save feature importances.

    Args:
        model (Any): Trained model with feature_importances_ attribute.
        feature_names (List[str]): Names of the features.
        output_path (str): File path to save the plot.
    """
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), sorted_importances, align="center")
        plt.xticks(range(len(importances)), sorted_features, rotation=90)
        plt.title("Feature Importances")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path)
        logging.info(f"Saved feature importance plot to {output_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to plot feature importance: {e}")

def plot_class_distribution(y: np.ndarray, output_path: str) -> None:
    """
    Plot class distribution.

    Args:
        y (np.ndarray): Labels.
        output_path (str): File path to save the plot.
    """
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=[str(u) for u in unique], autopct='%1.1f%%', startangle=140)
    plt.title("Class Distribution")
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path)
        logging.info(f"Saved class distribution to {output_path}")
    except IOError as e:
        logging.error(f"Failed to save plot to {output_path}: {e}")
    plt.close()

def generate_report(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Generate a markdown evaluation report.

    Args:
        metrics (Dict[str, Any]): Metrics dictionary.
        output_path (str): File path to save the report.
    """
    report = "# Evaluation Report\n\n"
    report += f"- **Accuracy**: {metrics.get('accuracy', 0):.4f}\n"
    report += f"- **Weighted F1**: {metrics.get('f1_weighted', 0):.4f}\n\n"
    report += "## Per-Class Metrics\n"
    
    per_class = metrics.get("per_class", {})
    for cls, cmets in per_class.items():
        report += f"### Class {cls}\n"
        for m, val in cmets.items():
            report += f"- {m}: {val:.4f}\n"
            
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logging.info(f"Saved report to {output_path}")
    except IOError as e:
        logging.error(f"Failed to write report to {output_path}: {e}")

def compare_baselines(results_dict: Dict[str, float], output_path: str) -> None:
    """
    Compare baseline models with a bar chart.

    Args:
        results_dict (Dict[str, float]): Model names mapped to their evaluation score.
        output_path (str): File path to save the plot.
    """
    models = list(results_dict.keys())
    scores = list(results_dict.values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(models, scores, color=['blue', 'orange', 'green'])
    plt.ylabel('F1 Score')
    plt.title('Baseline Model Comparison')
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path)
        logging.info(f"Saved baseline comparison to {output_path}")
    except IOError as e:
        logging.error(f"Failed to save baseline comparison: {e}")
    plt.close()

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 0, 2])
    
    metrics = compute_metrics(y_true, y_pred, labels=[0, 1, 2])
    print("Metrics Computed.")
    
    class DummyModel:
        def __init__(self):
            self.feature_importances_ = np.random.rand(5)
            
    dummy = DummyModel()
    
    plot_confusion_matrix(metrics["confusion_matrix"], ["Scanner", "Amateur", "Advanced"], "figures/confusion_matrix.png")
    plot_feature_importance(dummy, ["f1", "f2", "f3", "f4", "f5"], "figures/feature_importance.png")
    plot_class_distribution(y_true, "figures/class_distribution.png")
    generate_report(metrics, "figures/report.md")
    compare_baselines({"Rule-based": 0.65, "Random Forest": 0.82, "XGBoost": 0.88}, "figures/baseline_comparison.png")
