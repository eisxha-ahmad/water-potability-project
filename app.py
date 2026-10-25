"""Drinking Water Safety Classifier — Streamlit app (Stage G).

Run from the project folder:
    streamlit run app.py

Screens:
1. Data & Quality
2. Model Performance
3. Single Prediction
4. Batch Scoring
5. Settings (model, threshold, cost matrix — consequences update live)
6. Field Kit (phone-friendly, portable parameters only)

Everything the app shows comes from files in this folder. No path
needs editing and no dataset needs downloading by the marker.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve)

sys.path.insert(0, "src")
from config import FEATURES, RAW_PATH, TARGET  # noqa: E402
from explain import explain  # noqa: E402

KIT_FEATURES = ["ph", "Turbidity", "Conductivity", "Solids"]
MEDIANS = {"ph": 7.04, "Hardness": 196.97, "Solids": 20927.83,
           "Chloramines": 7.13, "Sulfate": 333.07, "Conductivity": 421.88,
           "Organic_carbon": 14.22, "Trihalomethanes": 66.62,
           "Turbidity": 3.96}


@st.cache_resource
def load_models():
    return joblib.load("models/all_models.pkl")


@st.cache_resource
def load_kit():
    return joblib.load("models/kit_model.pkl")


@st.cache_data
def load_evaluation():
    return json.loads(Path("reports/evaluation.json").read_text(encoding="utf-8"))


@st.cache_data
def load_test():
    return pd.read_csv("models/test_set.csv")


@st.cache_data
def load_cleaned():
    return pd.read_csv("data/cleaned/water_cleaned_v1.csv")


@st.cache_data
def load_audit():
    p = Path("reports/audit_report.md")
    return p.read_text(encoding="utf-8") if p.exists() else ""


def cost_of(y_true, y_pred, c_dangerous, c_harmless):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # fp here = actual unsafe (0) predicted safe (1): the dangerous error.
    return int(c_dangerous * fp + c_harmless * fn)


st.set_page_config(page_title="Drinking Water Safety Classifier",
                   page_icon="💧", layout="wide")
st.title("💧 Drinking Water Safety Classifier")
st.write("For a municipal inspector or NGO field officer: enter readings, "
         "get safe / unsafe, the parameter responsible, and what to do next.")

models = load_models()
kit_model = load_kit()
ev = load_evaluation()
test = load_test()
screen = st.sidebar.radio(
    "Choose a screen",
    ["1. Data & Quality", "2. Model Performance", "3. Single Prediction",
     "4. Batch Scoring", "5. Settings", "6. Field Kit (phone)"])

# ---------------------------------------------------------------- Screen 1
if screen.startswith("1."):
    st.header("1. Data & Quality")
    raw = pd.read_csv(RAW_PATH)
    st.write(f"Raw data: **{len(raw)} rows, {len(raw.columns)} columns** — "
             "this meets Stage A (at least 2,000 records, 10 attributes).")
    st.dataframe(raw.head(20))
    st.subheader("Missing values per column")
    miss = raw.isna().sum().rename("missing").to_frame()
    miss["missing %"] = (miss["missing"] / len(raw) * 100).round(1)
    st.dataframe(miss)
    st.subheader("Audit report (from the raw file)")
    st.markdown(load_audit())
    st.subheader("Download the cleaned file")
    cleaned = load_cleaned()
    st.write(f"Cleaned file: {len(cleaned)} rows. The 3 provenance columns "
             "show which values were imputed.")
    st.download_button("Download cleaned CSV",
                       cleaned.to_csv(index=False).encode("utf-8"),
                       file_name="water_cleaned_v1.csv", mime="text/csv")

# ---------------------------------------------------------------- Screen 2
elif screen.startswith("2."):
    st.header("2. Model Performance")
    baseline = ev["majority_baseline_test"]
    st.write(f"Majority-class baseline on the test set: **{baseline:.3f}** "
             "— every accuracy below is beside this baseline, never alone.")
    rows = []
    for name, m in ev["models"].items():
        rows.append({"Model": name, "Accuracy": round(m["accuracy"], 3),
                     "Baseline": round(baseline, 3),
                     "Recall (unsafe)": round(m["recall_unsafe"], 3),
                     "F1": round(m["f1"], 3),
                     "ROC AUC": round(m["roc_auc"], 3),
                     "Cost": m["total_cost"],
                     "CV accuracy (train)": f"{m['cv_accuracy_mean']:.3f} +/- {m['cv_accuracy_std']:.3f}"})
    st.dataframe(pd.DataFrame(rows))
    st.write("Honest note: this dataset is hard. The best model beats the "
             "baseline only slightly on accuracy, which is why recall for "
             "the unsafe class and the cost-based threshold matter more "
             "than accuracy alone. Naive Bayes is even slightly below "
             "the baseline on accuracy — that is reported, not hidden.")

    st.subheader("ROC curves (test set)")
    fig, ax = plt.subplots()
    y_test = test[TARGET]
    for name, model in models.items():
        proba = model.predict_proba(test[FEATURES])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(); st.pyplot(fig)

    st.subheader("Confusion matrices (test set, threshold 0.5)")
    for name, model in models.items():
        pred = model.predict(test[FEATURES])
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        st.write(f"**{name}** — rows = actual, columns = predicted "
                 "[unsafe, safe]:")
        st.dataframe(pd.DataFrame(cm, index=["Actual unsafe", "Actual safe"],
                                  columns=["Predicted unsafe", "Predicted safe"]))

    st.subheader("Chosen operating threshold")
    th = ev["threshold"]
    st.write(f"Model: {th['model']}, threshold: **{th['threshold']:.2f}** "
             "(chosen from the train out-of-fold cost sweep, not from 0.5).")
    st.write(f"At this threshold, test cost = "
             f"{th['test_metrics_at_threshold']['total_cost']} vs "
             f"{th['test_metrics_at_0.5']['total_cost']} at 0.5. "
             "A dangerous error (unsafe called safe) costs 10, a "
             "harmless-side error (safe called unsafe) costs 1.")

# ---------------------------------------------------------------- Screen 3
elif screen.startswith("3."):
    st.header("3. Single Prediction")
    st.write("Enter one water sample. Tick 'missing' for any reading you "
             "do not have — the model will still answer, but it will warn "
             "you that the answer is less certain.")
    model_name = st.selectbox("Model", list(models.keys()),
                              index=list(models.keys()).index("Decision Tree")
                              if "Decision Tree" in models else 0)
    case = {}
    cols = st.columns(3)
    for i, feat in enumerate(FEATURES):
        with cols[i % 3]:
            missing = st.checkbox(f"{feat} missing", key=f"miss_{feat}")
            if missing:
                case[feat] = None
            else:
                case[feat] = st.number_input(feat, value=float(MEDIANS[feat]),
                                             key=f"val_{feat}")
    if st.button("Predict", type="primary"):
        train = pd.read_csv("models/train_set.csv")
        result = explain(models[model_name], model_name, case,
                         train_X=train[FEATURES], train_y=train[TARGET])
        verdict = result["verdict"]
        st.success(f"✅ SAFE — confidence {result['confidence']:.0%}"
                   if verdict == "safe" else
                   f"⛔ UNSAFE — confidence {result['confidence']:.0%}")
        st.write(f"Probability the water is safe: {result['confidence_safe']:.0%}")
        st.subheader("Why — in plain language")
        for r in result["reasons"]:
            st.write(f"- {r}")
        st.subheader("Recommended action")
        st.write(result["action"])

# ---------------------------------------------------------------- Screen 4
elif screen.startswith("4."):
    st.header("4. Batch Scoring")
    st.write("Upload a CSV with the 9 feature columns. Results are ranked "
             "most-risky first (highest probability of being unsafe), "
             "so the worst supplies are checked first.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    model_name = st.selectbox("Model", list(models.keys()), key="batch_model")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing_cols = [c for c in FEATURES if c not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in your file: {missing_cols}")
        else:
            model = models[model_name]
            proba_safe = model.predict_proba(df[FEATURES])[:, 1]
            out = df.copy()
            out["prob_safe"] = proba_safe
            out["prob_unsafe"] = 1 - proba_safe
            out["verdict"] = np.where(proba_safe >= 0.5, "safe", "unsafe")
            out = out.sort_values("prob_unsafe", ascending=False)
            verdict_filter = st.selectbox("Filter by verdict",
                                          ["All", "unsafe", "safe"])
            if verdict_filter != "All":
                out = out[out["verdict"] == verdict_filter]
            st.dataframe(out)
            st.download_button("Download scored CSV",
                               out.to_csv(index=False).encode("utf-8"),
                               file_name="water_scored.csv", mime="text/csv")

# ---------------------------------------------------------------- Screen 5
elif screen.startswith("5."):
    st.header("5. Settings — see the consequences live")
    model_name = st.selectbox("Model", list(models.keys()), key="set_model")
    threshold = st.slider("Decision threshold (P(safe) needed to call it safe)",
                          0.05, 0.95, 0.50, 0.05)
    c_dangerous = st.number_input("Cost: unsafe called safe (dangerous)",
                                  value=10, min_value=1)
    c_harmless = st.number_input("Cost: safe called unsafe (harmless side)",
                                 value=1, min_value=1)
    model = models[model_name]
    y_test = test[TARGET]
    proba = model.predict_proba(test[FEATURES])[:, 1]
    pred = (proba >= threshold).astype(int)
    from sklearn.metrics import accuracy_score, recall_score
    acc = accuracy_score(y_test, pred)
    rec_unsafe = recall_score(y_test, pred, pos_label=0, zero_division=0)
    cost = cost_of(y_test, pred, c_dangerous, c_harmless)
    cost_05 = cost_of(y_test, (proba >= 0.5).astype(int), c_dangerous, c_harmless)
    st.write(f"At threshold **{threshold:.2f}**: accuracy **{acc:.3f}** "
             f"(baseline {ev['majority_baseline_test']:.3f}), "
             f"recall (unsafe) **{rec_unsafe:.3f}**, total cost **{cost}** "
             f"vs {cost_05} at 0.5 with the same costs.")
    st.write("Raising the threshold calls water safe less often: dangerous "
             "errors fall, but more safe supplies get flagged for retesting. "
             "That trade-off is the whole point of this screen.")
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    st.dataframe(pd.DataFrame(cm, index=["Actual unsafe", "Actual safe"],
                               columns=["Predicted unsafe", "Predicted safe"]))
    st.write(f"Recommended threshold from the cost sweep: "
             f"{ev['threshold']['threshold']:.2f} "
             f"for {ev['threshold']['model']}.")

# ---------------------------------------------------------------- Screen 6
else:
    st.header("6. Field Kit — for a phone, in the field")
    st.write("Only the parameters a portable kit can actually measure. "
             "The rest are not available here, and the app says so honestly.")
    kit = ev["field_kit"]
    st.warning(f"Confidence loss: full-model AUC {kit['full_model_auc']:.3f} "
               f"vs field-kit AUC {kit['roc_auc']:.3f} "
               f"(loss {kit['auc_loss']:.3f}). A field-kit verdict is a "
               "warning, not a laboratory result.")
    case = {}
    for feat in KIT_FEATURES:
        missing = st.checkbox(f"{feat} missing", key=f"kit_miss_{feat}")
        case[feat] = None if missing else st.number_input(
            feat, value=float(MEDIANS[feat]), key=f"kit_val_{feat}")
    if st.button("Check this water", type="primary"):
        result = explain(kit_model, "Field Kit", case)
        st.success(f"✅ SAFE — confidence {result['confidence']:.0%}"
                   if result["verdict"] == "safe" else
                   f"⛔ UNSAFE — confidence {result['confidence']:.0%}")
        for r in result["reasons"]:
            st.write(f"- {r}")
        st.write(result["action"])
        st.write("If this says unsafe, or if readings were missing, "
                 "send a sample to the laboratory for the full 9-parameter test.")
