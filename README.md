# Credit Card Fraud Detection

Detects fraudulent credit card transactions using a supervised machine learning model
and serves predictions through a Streamlit web interface.

**Dataset:** [Kaggle: Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
284,807 transactions, 492 frauds (0.172%). Features V1-V28 are anonymised PCA components;
`Time`, `Amount` and `Class` are the only original columns.

## Project structure

```
Credit-Card-Fraud-Detection/
├── data/
│   └── creditcard.csv          # download from Kaggle (not committed)
├── notebooks/
│   └── fraud_detection.ipynb   # EDA, training, evaluation
├── models/
│   ├── fraud_model.pkl         # trained model (saved from the notebook)
│   └── scaler.pkl              # fitted scaler
├── app.py                      # Streamlit interface
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

1. Download `creditcard.csv` from Kaggle and place it in `data/`.
2. Run `notebooks/fraud_detection.ipynb` top to bottom to train and save the model into `models/`.
3. Launch the app:

```bash
streamlit run app.py
```

## Evaluation approach

- Metric: PR-AUC (average precision), plus recall at a fixed precision. Accuracy is not used.
- Stratified train/test split performed **before** any resampling.
- Decision threshold tuned on the precision-recall curve, not left at 0.5.

## Status

- [x] Step 1: Project skeleton
- [ ] Step 2: Data loading and EDA
- [ ] Step 3: Preprocessing and split
- [ ] Step 4: Baseline model
- [ ] Step 5: Advanced models and comparison
- [ ] Step 6: Threshold tuning and evaluation
- [ ] Step 7: Save model artefacts
- [ ] Step 8: Streamlit app
- [ ] Step 9: Testing and documentation
