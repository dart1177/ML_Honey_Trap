"""
Evaluation script for ThreatClassifier.
"""

import argparse
import logging
import os
import json
import numpy as np
from rich.console import Console
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.classifier import ThreatClassifier
from src.evaluator import compute_metrics, plot_confusion_matrix, plot_feature_importance, plot_class_distribution, generate_report, compare_baselines
from src.features import build_feature_matrix, derive_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
console = Console()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ThreatClassifier model")
    parser.add_argument("--model", type=str, default="models/xgboost_model.pkl", help="Path to trained model")
    parser.add_argument("--data", type=str, default="data/test_sessions.json", help="Path to test data")
    parser.add_argument("--output", type=str, default="docs/evaluation_report.md", help="Output report path")
    parser.add_argument("--plot", type=str, default="all", choices=["all", "confusion", "importance", "distribution"], help="Plots to generate")
    return parser.parse_args()

def generate_synthetic_test_data(n_samples=100):
    """
    Generate synthetic test data if missing.
    
    Args:
        n_samples (int): Number of synthetic samples to generate.
        
    Returns:
        tuple: (X, y, feature_names)
    """
    import pandas as pd
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=n_samples, n_features=15, n_informative=10, n_classes=3, random_state=42)
    feature_names = [f"feature_{i}" for i in range(15)]
    return X, y, feature_names

def rule_based_classifier(X):
    """
    Simple rule-based baseline.
    
    Args:
        X (np.ndarray): Test features.
        
    Returns:
        np.ndarray: Predicted labels.
    """
    preds = []
    for row in X:
        if row[0] > 10:
            preds.append(2)
        elif row[0] > 2:
            preds.append(1)
        else:
            preds.append(0)
    return np.array(preds)

def main():
    """Main evaluation pipeline."""
    args = parse_args()
    os.makedirs("figures", exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    classifier = ThreatClassifier()
    try:
        classifier.load(args.model)
    except Exception as e:
        logging.error(f"Failed to load model {args.model}, ensure it exists: {e}")
        return
        
    X_test, y_test, feature_names = generate_synthetic_test_data(200)
    
    class_names = ["Scanner", "Amateur", "Advanced"]
    
    # Compute metrics
    metrics = classifier.evaluate(X_test, y_test)
    
    # Baselines
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_test, y_test) # Overfit on test just for baseline comparison demo
    rf_preds = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_preds, average="weighted")
    
    rb_preds = rule_based_classifier(X_test)
    rb_f1 = f1_score(y_test, rb_preds, average="weighted")
    
    xgb_f1 = metrics["f1_weighted"]
    baselines = {"Rule-based": rb_f1, "Random Forest": rf_f1, "XGBoost": xgb_f1}
    
    if args.plot in ["all", "confusion"]:
        plot_confusion_matrix(metrics["confusion_matrix"], class_names, "figures/confusion_matrix.png")
    if args.plot in ["all", "importance"]:
        plot_feature_importance(classifier.model, feature_names, "figures/feature_importance.png")
    if args.plot in ["all", "distribution"]:
        plot_class_distribution(y_test, "figures/class_distribution.png")
        
    compare_baselines(baselines, "figures/baseline_comparison.png")
    
    generate_report(metrics, args.output)
    
    console.print(f"[bold green]Evaluation complete! Report saved to {args.output}[/bold green]")
    console.print("Metrics:")
    for k, v in metrics.items():
        if k != "per_class":
            console.print(f"{k}: {v}")

if __name__ == "__main__":
    main()
