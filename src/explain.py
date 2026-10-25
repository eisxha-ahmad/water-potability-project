"""Stage F — Explanations in the user's language, not the model's.

Every prediction the product shows comes with:
- verdict (safe / unsafe), confidence, the parameter(s) responsible,
  and a recommended action.
- Never "node 14, gini 0.31". Always "pH is 5.1, below the safe
  range 6.5-8.5, so do not drink this water".

How each model explains itself:
- Decision Tree: the decision path, translated rule by rule.
- k-Nearest Neighbours: the nearest training cases and what most
  of them were.
- Gaussian Naive Bayes: the features whose evidence moved the
  posterior most, from the Gaussian log-likelihood difference.

WHO / common safe ranges used only for wording, never as a second
model: pH 6.5-8.5, Turbidity below 5 NTU is desirable, TDS (Solids)
desirable below 500 mg/L and maximum 1000 mg/L. Other parameters
are described by comparison with the training medians, which are
computed from the training set passed in.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# Human names, units and safe-range wording for the report/app.
HUMAN = {
    "ph": ("pH", "", "safe range 6.5 to 8.5"),
    "Hardness": ("Hardness", " mg/L", "typical training median"),
    "Solids": ("Total dissolved solids", " ppm", "desirable below 500, maximum 1000"),
    "Chloramines": ("Chloramines", " ppm", "typical training median"),
    "Sulfate": ("Sulfate", " mg/L", "typical training median"),
    "Conductivity": ("Conductivity", " uS/cm", "typical training median"),
    "Organic_carbon": ("Organic carbon", " ppm", "typical training median"),
    "Trihalomethanes": ("Trihalomethanes", " ug/L", "typical training median"),
    "Turbidity": ("Turbidity", " NTU", "desirable below 5"),
}

ACTIONS = {
    "unsafe": ("Do NOT drink this water. Boil or filter it if you must use "
               "it, and send a sample to the laboratory for a full test."),
    "safe": ("On these readings the water looks safe, but keep testing "
             "regularly. If taste, colour or smell changes, test again."),
}


def _fmt(col, value):
    name, unit, _ = HUMAN.get(col, (col, "", ""))
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{name} was not measured"
    return f"{name} is {value:.2f}{unit}"


def _action_for(verdict):
    return ACTIONS["unsafe"] if verdict == "unsafe" else ACTIONS["safe"]


def _tree_path_reasons(pipeline, case_df):
    """Translate a Decision Tree decision path into plain rules."""
    model = pipeline.named_steps["model"]
    # Transform the case through every step except the model itself.
    X = case_df
    for name, step in pipeline.steps[:-1]:
        X = step.transform(X)
    X = np.asarray(X)
    node = model.tree_
    feature_names = list(case_df.columns)
    # If the imputer added missing indicators, the tree sees extra
    # columns after the originals; ignore those in wording.
    path = model.decision_path(X).indices
    reasons = []
    for node_id in path:
        if node.feature[node_id] < 0:
            continue  # leaf
        feat_idx = int(node.feature[node_id])
        if feat_idx >= len(feature_names):
            continue  # missing-indicator feature, not a water parameter
        col = feature_names[feat_idx]
        thr = float(node.threshold[node_id])
        value = float(X[0, feat_idx])
        direction = "at or below" if value <= thr else "above"
        name, unit, _ = HUMAN.get(col, (col, "", ""))
        reasons.append(
            f"{name} is {value:.2f}{unit}, which is {direction} "
            f"{thr:.2f}{unit} on this branch of the decision tree.")
    return reasons


def _knn_reasons(pipeline, case_df, train_X, train_y):
    """Nearest training cases, in plain words."""
    # Transform case and training data through imputer+scaler steps.
    def transform(df):
        X = df
        for name, step in pipeline.steps[:-1]:
            X = step.transform(X)
        return np.asarray(X, dtype=float)

    x = transform(case_df)[0]
    Xt = transform(train_X)
    dist = np.sqrt(((Xt - x) ** 2).sum(axis=1))
    idx = np.argsort(dist)[:3]
    reasons = []
    for i in idx:
        label = "safe" if int(train_y.iloc[i]) == 1 else "unsafe"
        reasons.append(
            f"A very similar past sample (distance {dist[i]:.2f}) was {label}.")
    n_unsafe = int((train_y.iloc[idx] == 0).sum())
    reasons.append(
        f"Of the 3 most similar past samples, {n_unsafe} were unsafe "
        f"and {3 - n_unsafe} were safe.")
    return reasons


def _nb_reasons(pipeline, case_df):
    """Features that moved the Naive Bayes posterior most."""
    model = pipeline.named_steps["model"]
    X = case_df
    for name, step in pipeline.steps[:-1]:
        X = step.transform(X)
    X = np.asarray(X, dtype=float)[0]
    # GaussianNB: theta_ = class means, var_ = class variances,
    # classes are [0 (unsafe), 1 (safe)].
    logp_safe = norm.logpdf(X, loc=model.theta_[1], scale=np.sqrt(model.var_[1]))
    logp_unsafe = norm.logpdf(X, loc=model.theta_[0], scale=np.sqrt(model.var_[0]))
    # Positive difference = evidence towards safe.
    diff = logp_safe - logp_unsafe
    cols = list(case_df.columns)
    order = np.argsort(-np.abs(diff))[:3]
    reasons = []
    for j in order:
        col = cols[j] if j < len(cols) else f"feature {j}"
        direction = "towards safe" if diff[j] > 0 else "towards unsafe"
        reasons.append(f"{_fmt(col, float(X[j]))} pushes the result {direction}.")
    return reasons


def explain(pipeline, model_name, case_dict, train_X=None, train_y=None):
    """Return a dict the app can show directly.

    case_dict: {feature: value or None} for the 9 full features (or
    the 4 kit features for the kit model, whose model_name is
    'Field Kit'). Returns verdict, confidence (P(safe)), reasons, action.
    """
    cols = list(case_dict.keys())
    case_df = pd.DataFrame([{c: (np.nan if v is None else v) for c, v in case_dict.items()}])
    proba = pipeline.predict_proba(case_df)[0]
    # predict_proba columns follow pipeline.classes_ = [0, 1]
    p_safe = float(proba[list(pipeline.classes_).index(1)])
    verdict = "safe" if p_safe >= 0.5 else "unsafe"
    n_missing = int(case_df.isna().sum(axis=1).iloc[0])

    if "Decision Tree" in model_name or "Field Kit" in model_name:
        reasons = _tree_path_reasons(pipeline, case_df)
    elif "Neighbour" in model_name and train_X is not None:
        reasons = _knn_reasons(pipeline, case_df, train_X, train_y)
    elif "Naive Bayes" in model_name:
        reasons = _nb_reasons(pipeline, case_df)
    else:
        reasons = []

    # Add plain-language flags for the two parameters with WHO ranges.
    flags = []
    if case_dict.get("ph") is not None:
        if case_dict["ph"] < 6.5 or case_dict["ph"] > 8.5:
            flags.append("pH is outside the safe range 6.5 to 8.5.")
    if case_dict.get("Turbidity") is not None and case_dict["Turbidity"] >= 5:
        flags.append("Turbidity is 5 NTU or higher, which is cloudy water.")
    reasons = flags + reasons

    if n_missing > 0:
        reasons.append(
            f"Warning: {n_missing} reading(s) were missing, so this "
            f"prediction is less certain than a complete test. "
            f"Treat it as a warning, not a final lab result.")
    if n_missing >= 5:
        reasons.append(
            "Too many readings are missing for this result to be trusted. "
            "Please measure more parameters or send a sample to the lab.")

    return {
        "verdict": verdict,
        "confidence_safe": p_safe,
        "confidence": p_safe if verdict == "safe" else 1 - p_safe,
        "reasons": reasons,
        "action": _action_for(verdict),
        "n_missing": n_missing,
    }
