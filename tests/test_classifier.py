"""
Tests for src/classifier.py
"""

import os
import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.classifier import ThreatClassifier

@pytest.fixture
def synthetic_data():
    X, y = make_classification(n_samples=300, n_features=15, n_classes=3, n_informative=10, random_state=42)
    return X[:200], y[:200], X[200:], y[200:]

@pytest.fixture
def classifier(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("xgboost: {n_estimators: 10, max_depth: 3}")
    return ThreatClassifier(str(config_path))

def test_classifier_init(classifier):
    assert classifier is not None
    assert classifier.model is not None

def test_train_returns_metrics(classifier, synthetic_data):
    X_train, y_train, _, _ = synthetic_data
    metrics = classifier.train(X_train, y_train)
    assert "train_accuracy" in metrics
    assert isinstance(metrics["train_accuracy"], float)

def test_predict_shape(classifier, synthetic_data):
    X_train, y_train, X_test, _ = synthetic_data
    classifier.train(X_train, y_train)
    preds = classifier.predict(X_test)
    assert preds.shape == (len(X_test),)

def test_predict_proba_sums_to_one(classifier, synthetic_data):
    X_train, y_train, X_test, _ = synthetic_data
    classifier.train(X_train, y_train)
    probs = classifier.predict_proba(X_test)
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_save_load_roundtrip(classifier, synthetic_data, tmp_path):
    X_train, y_train, X_test, _ = synthetic_data
    classifier.train(X_train, y_train)
    
    preds_before = classifier.predict(X_test)
    
    model_path = tmp_path / "model.pkl"
    classifier.save(str(model_path))
    assert os.path.exists(model_path)
    
    new_classifier = ThreatClassifier()
    new_classifier.load(str(model_path))
    preds_after = new_classifier.predict(X_test)
    
    np.testing.assert_array_equal(preds_before, preds_after)

def test_evaluate_returns_all_metrics(classifier, synthetic_data):
    X_train, y_train, X_test, y_test = synthetic_data
    classifier.train(X_train, y_train)
    metrics = classifier.evaluate(X_test, y_test)
    
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics

def test_feature_importances(classifier, synthetic_data):
    X_train, y_train, _, _ = synthetic_data
    classifier.train(X_train, y_train)
    importances = classifier.feature_importances
    assert len(importances) == 15
