"""Credit card fraud detection app.

Run from the project root:
    streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
DATA_PATH = ROOT / "data" / "creditcard.csv"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
SCALED_FEATURES = ["Time", "Amount"]


@st.cache_resource
def load_artifacts():
    """Load the Isolation Forest and the RobustScaler saved by the notebook."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def score(frame: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """Return a fraud flag and an anomaly score for each row. Higher means more unusual."""
    X = frame[FEATURES].astype(float).copy()
    X[SCALED_FEATURES] = scaler.transform(X[SCALED_FEATURES])
    scores = -model.score_samples(X)
    threshold = float(model.fraud_threshold_)
    return pd.DataFrame(
        {
            "label": np.where(scores >= threshold, "Fraud", "Normal"),
            "anomaly_score": np.round(scores, 6),
            "threshold": threshold,
        },
        index=frame.index,
    )


def fill_with_sample(is_fraud: int) -> None:
    row = load_dataset().loc[lambda d: d["Class"] == is_fraud].sample(1).iloc[0]
    for name in FEATURES:
        st.session_state[name] = float(row[name])
    st.session_state["sample_label"] = "Fraud" if is_fraud else "Normal"


st.set_page_config(page_title="Fraud Detection", page_icon=":credit_card:", layout="centered")
st.title("Credit Card Fraud Detection")
st.caption("Isolation Forest anomaly detection. A higher score means the transaction looks less like normal spending.")

artifacts = load_artifacts()
if artifacts is None:
    st.error(
        "Trained files were not found in `models/`. "
        "Open `notebooks/fraud_detection.ipynb` and run all cells, then refresh this page."
    )
    st.stop()

model, scaler = artifacts

st.subheader("Check a transaction")
for name in FEATURES:
    st.session_state.setdefault(name, 0.0)

if DATA_PATH.exists():
    c1, c2 = st.columns(2)
    c1.button("Load a normal sample", on_click=fill_with_sample, args=(0,), width="stretch")
    c2.button("Load a fraud sample", on_click=fill_with_sample, args=(1,), width="stretch")
    if "sample_label" in st.session_state:
        st.caption(f"Loaded a real transaction. True label: **{st.session_state['sample_label']}**")

c1, c2 = st.columns(2)
c1.number_input("Time (seconds since first transaction)", min_value=0.0, key="Time")
c2.number_input("Amount", min_value=0.0, key="Amount")
with st.expander("PCA features V1 to V28"):
    cols = st.columns(4)
    for i in range(1, 29):
        cols[(i - 1) % 4].number_input(f"V{i}", key=f"V{i}", format="%.4f")

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{name: st.session_state[name] for name in FEATURES}])
    result = score(row, model, scaler).iloc[0]
    message = (
        f"{result['label'].upper()}: anomaly score {result['anomaly_score']:.4f} "
        f"(threshold {result['threshold']:.4f})"
    )
    st.error(message) if result["label"] == "Fraud" else st.success(message)

st.divider()
st.subheader("Check a CSV file")
st.caption("Columns needed: Time, V1 to V28, and Amount.")
uploaded = st.file_uploader("Upload transactions CSV", type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
    elif df[FEATURES].isna().any().any():
        st.error("The file has empty values in the feature columns.")
    elif df.empty:
        st.error("The file has no rows.")
    else:
        result = score(df, model, scaler)
        table = pd.concat([result[["label", "anomaly_score"]], df], axis=1)
        flagged = int((result["label"] == "Fraud").sum())
        st.write(f"**{flagged}** of {len(result)} transactions flagged as fraud.")
        st.dataframe(table.sort_values("anomaly_score", ascending=False), hide_index=True)
