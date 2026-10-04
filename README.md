# ML Honey Trap

An ML-powered honeypot designed for an IEEE paper. This repository contains the complete pipeline from parsing raw Cowrie honeypot logs to feature extraction, threat classification (using XGBoost), dynamic response generation, and real-time visualization via Streamlit.

## Setup Instructions

1. **Install Dependencies**
   Make sure you have Python 3.10+ installed. Install the required packages via:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Dataset (Optional)**
   You can download the Shell Attack Evolution dataset from HuggingFace to train on real data. If you skip this, the training script will automatically generate synthetic fallback data.
   ```bash
   python scripts/download_dataset.py
   ```

## Pipeline Execution

Follow these steps to run the complete machine learning pipeline:

### 1. Train the Model
Run the training script to parse data, extract features, and train the XGBoost classifier.
```bash
python scripts/train.py --config config/config.yaml --output models/xgboost_model.pkl
```
*This will output the training metrics and save the model to the `models/` directory.*

### 2. Evaluate the Model
Evaluate the trained model against a test dataset and baseline models (Random Forest, Rule-based).
```bash
python scripts/evaluate.py --model models/xgboost_model.pkl
```
*This will generate a classification report, confusion matrix, feature importance graph, and baseline comparison in the `figures/` and `docs/` directories.*

### 3. Run the Dashboard
To visualize real-time sessions, threat levels, and extracted Indicators of Compromise (IOCs), launch the Streamlit dashboard:
```bash
streamlit run src/dashboard.py
```
*The dashboard will automatically open in your default web browser.*

## Running Unit Tests
The project includes comprehensive unit tests built with `pytest` covering the parser, feature extraction, and classifier.
```bash
pytest tests/ -v
```

## Core Modules
- `src/parser.py`: Parses JSON logs from Cowrie into structured sessions.
- `src/features.py`: Extracts 15 distinct session-level and payload-level features (entropy, privileges, etc.).
- `src/classifier.py`: The XGBoost-based threat classifier.
- `src/responder.py`: Adaptive response engine determining mitigations (e.g., monitor, tarpit, fake_flag).
- `src/evaluator.py`: Utility functions for metrics and plotting.
