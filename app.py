import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)

st.set_page_config(page_title="Credit Default Classifier", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib",
}

TARGET_COL = "default"


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


st.title("Credit Card Default Prediction")
st.write("UCI Default of Credit Card Clients — comparison of five classifiers.")

# ---------- Sidebar: model selection ----------
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))

# ---------- Dataset upload ----------
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded is None:
    st.info("Upload test_data.csv to see model performance.")
    st.stop()

data = load_csv(uploaded)

if TARGET_COL not in data.columns:
    st.error(f"Uploaded file must contain a '{TARGET_COL}' column.")
    st.stop()

st.subheader("Uploaded data")
st.write("Shape:", data.shape)
st.dataframe(data.head())

X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

# ---------- Prediction ----------
pipe = load_model(MODEL_FILES[model_name])
y_pred = pipe.predict(X)
y_proba = pipe.predict_proba(X)[:, 1]

# ---------- YOUR CODE 1: evaluation metrics ----------
st.subheader(f"Evaluation metrics — {model_name}")

metrics = {
    "Metric": [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ],
    "Score": [
        accuracy_score(y, y_pred),
        roc_auc_score(y, y_proba),
        precision_score(y, y_pred, zero_division=0),
        recall_score(y, y_pred, zero_division=0),
        f1_score(y, y_pred, zero_division=0),
        matthews_corrcoef(y, y_pred),
    ],
}
metrics_df = pd.DataFrame(metrics).set_index("Metric")
st.dataframe(metrics_df, use_container_width=True)

# ---------- YOUR CODE 2: confusion matrix ----------
st.subheader("Confusion matrix")

cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)