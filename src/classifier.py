"""
Threat classification module using XGBoost.
Handles model training, inference, saving, and loading.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import yaml
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ThreatClassifier:
    """XGBoost-based threat classifier."""
    
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Initialize the classifier.

        Args:
            config_path (str): Path to configuration YAML file.
        """
        self.config_path = config_path
        self.model = None
        self.config: Dict[str, Any] = {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        except IOError as e:
            logging.error(f"Failed to read config file {config_path}: {e}")
        
        xgb_params = self.config.get("xgboost", {})
        self.model = XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 100),
            max_depth=xgb_params.get("max_depth", 6),
            learning_rate=xgb_params.get("learning_rate", 0.1),
            use_label_encoder=False,
            objective="multi:softmax",
            num_class=3
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Train the XGBoost model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (Optional[np.ndarray]): Validation features.
            y_val (Optional[np.ndarray]): Validation labels.

        Returns:
            Dict[str, float]: Training metrics.
        """
        logging.info("Starting model training...")
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        logging.info("Model training completed.")
        
        preds = self.model.predict(X_train)
        return {"train_accuracy": float(accuracy_score(y_train, preds))}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict threat classes for given features.

        Args:
            X (np.ndarray): Features to classify.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X (np.ndarray): Features to classify.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path (str): File path to save the model.
        """
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            joblib.dump(self.model, path)
            logging.info(f"Model saved to {path}")
        except IOError as e:
            logging.error(f"Failed to save model: {e}")

    def load(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path (str): File path to load the model from.
        """
        try:
            self.model = joblib.load(path)
            logging.info(f"Model loaded from {path}")
        except IOError as e:
            logging.error(f"Failed to load model: {e}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on a test set.

        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.

        Returns:
            Dict[str, Any]: Evaluation metrics.
        """
        preds = self.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec_w = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec_w = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1_w = f1_score(y_test, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, preds).tolist()
        
        prec_pc = precision_score(y_test, preds, average=None, zero_division=0)
        rec_pc = recall_score(y_test, preds, average=None, zero_division=0)
        f1_pc = f1_score(y_test, preds, average=None, zero_division=0)
        
        per_class = {
            i: {"precision": prec_pc[i], "recall": rec_pc[i], "f1": f1_pc[i]}
            for i in range(len(prec_pc))
        }
        
        return {
            "accuracy": float(acc),
            "precision_weighted": float(prec_w),
            "recall_weighted": float(rec_w),
            "f1_weighted": float(f1_w),
            "per_class": per_class,
            "confusion_matrix": cm
        }
        
    @property
    def feature_importances(self) -> np.ndarray:
        """Get feature importances from the model."""
        if self.model is None:
            raise ValueError("Model is not trained.")
        return self.model.feature_importances_

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Create dummy config
    os.makedirs("config", exist_ok=True)
    try:
        with open("config/config.yaml", "w", encoding="utf-8") as f:
            yaml.dump({"xgboost": {"n_estimators": 50, "max_depth": 3}}, f)
    except IOError as e:
        logging.error(f"Failed to create dummy config: {e}")

    # Generate synthetic 3-class dataset
    X, y = make_classification(
        n_samples=300, n_features=15, n_informative=10, 
        n_classes=3, random_state=42
    )
    
    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]
    
    classifier = ThreatClassifier("config/config.yaml")
    train_metrics = classifier.train(X_train, y_train)
    logging.info(f"Train metrics: {train_metrics}")
    
    eval_metrics = classifier.evaluate(X_test, y_test)
    print("Evaluation Metrics:")
    for k, v in eval_metrics.items():
        if k != "per_class":
            print(f"{k}: {v}")
        else:
            print("per_class:")
            for cls_idx, mets in v.items():
                print(f"  Class {cls_idx}: {mets}")
