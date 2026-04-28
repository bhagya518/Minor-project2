"""
=============================================================
  ML Ensemble Pipeline
  Decentralized Website Monitoring — Malicious Node Detection
=============================================================
  Implements the 3-model ensemble + meta-learner from the
  proposed system architecture:

    Model 1 : Random Forest       (supervised)
    Model 2 : Isolation Forest    (unsupervised anomaly)
    Model 3 : Graph Anomaly       (peer deviation analysis)
    Meta    : Gradient Boosting   (stacks all 3 predictions)

  Usage:
    pip install pandas numpy scikit-learn joblib matplotlib
    python ml_pipeline.py

  Expects: train.csv and test.csv from split_dataset.py
  Outputs:
    models/rf_model.pkl
    models/iso_model.pkl
    models/gb_meta_model.pkl
    models/scaler.pkl
    models/feature_cols.json
    ml_report.txt
    confusion_matrix.png
    feature_importance.png
=============================================================
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    GradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
TRAIN_FILE   = "train.csv"
TEST_FILE    = "test.csv"
MODELS_DIR   = "models"
SEED         = 42

FEATURE_COLS = [
    "accuracy",
    "false_positive_rate",
    "false_negative_rate",
    "avg_rt_error",
    "max_rt_error",
    "peer_agreement_rate",
    "historical_accuracy",
    "accuracy_std_dev",
    "report_consistency",
    "sudden_change_score",
    "ssl_accuracy",
    "uptime_deviation",
    "rt_consistency",
]

os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_step(msg):
    print(f"\n  > {msg}")


# ─────────────────────────────────────────────
#  STEP 1 — LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    print_section("STEP 1 — Loading Data")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["is_malicious"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["is_malicious"].values

    print_step(f"Training samples : {len(X_train)}")
    print_step(f"Test samples     : {len(X_test)}")
    print_step(f"Features         : {len(FEATURE_COLS)}")
    print_step(f"Class balance (train) — Honest: {sum(y_train==0)}, Malicious: {sum(y_train==1)}")

    return X_train, y_train, X_test, y_test, train_df, test_df


# ─────────────────────────────────────────────
#  STEP 2 — SCALE FEATURES
# ─────────────────────────────────────────────
def scale_features(X_train, X_test):
    print_section("STEP 2 — Scaling Features")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    print_step("StandardScaler fitted and saved -> models/scaler.pkl")

    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
#  MODEL 1 — RANDOM FOREST (Supervised)
# ─────────────────────────────────────────────
def train_random_forest(X_train, y_train, X_test, y_test):
    print_section("MODEL 1 — Random Forest (Supervised)")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",   # handles class imbalance (80/20 split)
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1")
    print_step(f"5-Fold CV F1 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Test performance
    y_pred     = rf.predict(X_test)
    y_prob     = rf.predict_proba(X_test)[:, 1]
    f1         = f1_score(y_test, y_pred)
    precision  = precision_score(y_test, y_pred)
    recall     = recall_score(y_test, y_pred)
    roc_auc    = roc_auc_score(y_test, y_prob)

    print_step(f"Test F1        : {f1:.4f}")
    print_step(f"Test Precision : {precision:.4f}")
    print_step(f"Test Recall    : {recall:.4f}")
    print_step(f"Test ROC-AUC   : {roc_auc:.4f}")

    joblib.dump(rf, f"{MODELS_DIR}/rf_model.pkl")
    print_step("Model saved -> models/rf_model.pkl")

    return rf, y_pred, y_prob


# ─────────────────────────────────────────────
#  MODEL 2 — ISOLATION FOREST (Unsupervised)
# ─────────────────────────────────────────────
def train_isolation_forest(X_train, y_train, X_test, y_test):
    print_section("MODEL 2 — Isolation Forest (Unsupervised Anomaly Detection)")

    # Train ONLY on honest nodes — learns what "normal" looks like
    honest_mask = y_train == 0
    X_honest    = X_train[honest_mask]

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.20,    # expected malicious fraction
        max_samples="auto",
        random_state=SEED,
    )
    iso.fit(X_honest)
    print_step(f"Trained on {sum(honest_mask)} honest-only samples (unsupervised)")

    # IsolationForest returns -1 (anomaly) or +1 (normal)
    # Convert to 0 = honest, 1 = malicious
    iso_raw_test  = iso.predict(X_test)
    y_pred_iso    = (iso_raw_test == -1).astype(int)

    # Anomaly scores (lower = more anomalous) -> invert for probability
    iso_scores    = iso.score_samples(X_test)             # negative -> low = anomaly
    y_prob_iso    = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)

    f1       = f1_score(y_test, y_pred_iso)
    roc_auc  = roc_auc_score(y_test, y_prob_iso)

    print_step(f"Test F1      : {f1:.4f}")
    print_step(f"Test ROC-AUC : {roc_auc:.4f}")
    print_step("Note: Unsupervised — no labels used during training")

    joblib.dump(iso, f"{MODELS_DIR}/iso_model.pkl")
    print_step("Model saved -> models/iso_model.pkl")

    return iso, y_pred_iso, y_prob_iso


# ─────────────────────────────────────────────
#  MODEL 3 — GRAPH ANOMALY (Peer Deviation)
# ─────────────────────────────────────────────
def compute_graph_anomaly_scores(test_df, X_test_scaled, y_test):
    """
    Graph-based anomaly: detects nodes that consistently deviate
    from their peer group's behaviour within the same epoch.

    For each node at each epoch:
      - Compute mean peer values for each feature
      - Measure this node's z-score vs peer distribution
      - Aggregate across features -> graph_anomaly_score
    """
    print_section("MODEL 3 — Graph Anomaly (Peer Deviation Analysis)")

    # Reset index so positional indexing works correctly
    test_df = test_df.reset_index(drop=True)
    scores = []
    for idx, row in test_df.iterrows():
        epoch = row["epoch"]
        node  = row["node_id"]

        # Peers = all other nodes at the same epoch
        peers = test_df[(test_df["epoch"] == epoch) & (test_df["node_id"] != node)]

        if len(peers) == 0:
            scores.append(0.0)
            continue

        peer_means = peers[FEATURE_COLS].mean()
        peer_stds  = peers[FEATURE_COLS].std().fillna(0).replace(0, 1e-9)

        z_scores    = abs((row[FEATURE_COLS] - peer_means) / peer_stds)
        graph_score = float(np.nanmean(z_scores.values))
        if np.isnan(graph_score):
            graph_score = 0.0
        scores.append(graph_score)

    scores = np.array(scores)

    # Normalise to 0-1 probability
    y_prob_graph = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    # Threshold at 0.5 for binary prediction
    y_pred_graph = (y_prob_graph >= 0.5).astype(int)

    f1       = f1_score(y_test, y_pred_graph)
    roc_auc  = roc_auc_score(y_test, y_prob_graph)

    print_step(f"Test F1      : {f1:.4f}")
    print_step(f"Test ROC-AUC : {roc_auc:.4f}")
    print_step("Graph anomaly scores computed from peer z-score deviation")

    return y_pred_graph, y_prob_graph


# ─────────────────────────────────────────────
#  META-LEARNER — Gradient Boosting
# ─────────────────────────────────────────────
def train_meta_learner(
    X_train, y_train, X_test, y_test,
    rf_model, iso_model, test_df,
    X_train_scaled, X_test_scaled,
    train_df,
):
    print_section("META-LEARNER — Gradient Boosting (Stacking Ensemble)")

    # ── Build meta-features for TRAINING set ──
    print_step("Building meta-features for training set...")

    # RF probabilities on train
    rf_train_prob  = rf_model.predict_proba(X_train_scaled)[:, 1]
    rf_train_pred  = rf_model.predict(X_train_scaled)

    # ISO scores on train
    iso_raw_train  = iso_model.predict(X_train_scaled)
    iso_train_pred = (iso_raw_train == -1).astype(int)
    iso_train_sc   = iso_model.score_samples(X_train_scaled)
    iso_train_prob = 1 - (iso_train_sc - iso_train_sc.min()) / (iso_train_sc.max() - iso_train_sc.min() + 1e-9)

    # Graph anomaly on train
    train_graph_scores = []
    for idx, row in train_df.iterrows():
        epoch = row["epoch"]
        node  = row["node_id"]
        peers = train_df[(train_df["epoch"] == epoch) & (train_df["node_id"] != node)]
        if len(peers) == 0:
            train_graph_scores.append(0.0)
            continue
        peer_means = peers[FEATURE_COLS].mean()
        peer_stds  = peers[FEATURE_COLS].std().replace(0, 1e-9)
        z_scores   = abs((row[FEATURE_COLS] - peer_means) / peer_stds)
        train_graph_scores.append(float(z_scores.mean()))

    train_graph_arr  = np.array(train_graph_scores)
    graph_train_prob = (train_graph_arr - train_graph_arr.min()) / (train_graph_arr.max() - train_graph_arr.min() + 1e-9)
    graph_train_pred = (graph_train_prob >= 0.5).astype(int)

    # Stack into meta-feature matrix
    # Shape: (n_samples, 6) — 3 predictions + 3 probabilities
    X_meta_train = np.column_stack([
        rf_train_pred,   rf_train_prob,
        iso_train_pred,  iso_train_prob,
        graph_train_pred, graph_train_prob,
    ])
    X_meta_train = np.nan_to_num(X_meta_train, nan=0.0)

    # ── Build meta-features for TEST set ──
    rf_test_prob  = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_test_pred  = rf_model.predict(X_test_scaled)

    iso_raw_test  = iso_model.predict(X_test_scaled)
    iso_test_pred = (iso_raw_test == -1).astype(int)
    iso_test_sc   = iso_model.score_samples(X_test_scaled)
    iso_test_prob = 1 - (iso_test_sc - iso_test_sc.min()) / (iso_test_sc.max() - iso_test_sc.min() + 1e-9)

    # Graph test predictions (already computed)
    graph_test_scores = []
    for idx, row in test_df.iterrows():
        epoch = row["epoch"]
        node  = row["node_id"]
        peers = test_df[(test_df["epoch"] == epoch) & (test_df["node_id"] != node)]
        if len(peers) == 0:
            graph_test_scores.append(0.0)
            continue
        peer_means = peers[FEATURE_COLS].mean()
        peer_stds  = peers[FEATURE_COLS].std().replace(0, 1e-9)
        z_scores   = abs((row[FEATURE_COLS] - peer_means) / peer_stds)
        graph_test_scores.append(float(z_scores.mean()))

    graph_test_arr  = np.array(graph_test_scores)
    graph_test_prob = (graph_test_arr - graph_test_arr.min()) / (graph_test_arr.max() - graph_test_arr.min() + 1e-9)
    graph_test_pred = (graph_test_prob >= 0.5).astype(int)

    X_meta_test = np.column_stack([
        rf_test_pred,   rf_test_prob,
        iso_test_pred,  iso_test_prob,
        graph_test_pred, graph_test_prob,
    ])
    X_meta_test = np.nan_to_num(X_meta_test, nan=0.0)

    # ── Train Gradient Boosting meta-learner ──
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=SEED,
    )
    gb.fit(X_meta_train, y_train)

    y_pred_meta = gb.predict(X_meta_test)
    y_prob_meta = gb.predict_proba(X_meta_test)[:, 1]

    f1        = f1_score(y_test, y_pred_meta)
    precision = precision_score(y_test, y_pred_meta)
    recall    = recall_score(y_test, y_pred_meta)
    roc_auc   = roc_auc_score(y_test, y_prob_meta)

    print_step(f"Meta F1        : {f1:.4f}")
    print_step(f"Meta Precision : {precision:.4f}")
    print_step(f"Meta Recall    : {recall:.4f}")
    print_step(f"Meta ROC-AUC   : {roc_auc:.4f}")

    joblib.dump(gb, f"{MODELS_DIR}/gb_meta_model.pkl")
    print_step("Meta-learner saved -> models/gb_meta_model.pkl")

    return gb, y_pred_meta, y_prob_meta, X_meta_test


# ─────────────────────────────────────────────
#  REPUTATION SCORES from malicious_probability
# ─────────────────────────────────────────────
def compute_reputation_and_action(malicious_probability):
    """
    Converts a malicious probability (0.0–1.0) into:
      - reputation score (0.0–1.0)
      - action: ALLOW / WARN / QUARANTINE / SLASH
    Matches the proposed architecture thresholds exactly.
    """
    rep = 1.0 - malicious_probability

    if rep >= 0.75:
        action = "ALLOW"
        vote_weight = 1.0
    elif rep >= 0.50:
        action = "WARN"
        vote_weight = 0.5
    elif rep >= 0.25:
        action = "QUARANTINE"
        vote_weight = 0.0
    else:
        action = "SLASH"
        vote_weight = 0.0

    return round(rep, 4), action, vote_weight


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred_meta):
    cm = confusion_matrix(y_test, y_pred_meta)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Honest", "Malicious"])
    ax.set_yticklabels(["Honest", "Malicious"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Ensemble Meta-Learner")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.close()
    print_step("Confusion matrix saved -> confusion_matrix.png")


def plot_feature_importance(rf_model):
    importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2d7ef7" if v > importances.median() else "#94a3b8" for v in importances]
    bars = ax.barh(importances.index, importances.values, color=colors, edgecolor="none")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest — Feature Importance")
    ax.axvline(importances.median(), color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()
    print_step("Feature importance saved -> feature_importance.png")


# ─────────────────────────────────────────────
#  FINAL REPORT
# ─────────────────────────────────────────────
def save_report(y_test, y_pred_rf, y_pred_iso, y_pred_graph, y_pred_meta,
                y_prob_meta, test_df):
    print_section("FINAL REPORT")

    lines = []
    lines.append("=" * 60)
    lines.append("  ML Ensemble Report — Malicious Node Detection")
    lines.append("=" * 60)

    lines.append("\nMODEL COMPARISON")
    lines.append("-" * 40)
    models_results = {
        "Random Forest (supervised)": y_pred_rf,
        "Isolation Forest (unsupervised)": y_pred_iso,
        "Graph Anomaly (peer deviation)": y_pred_graph,
        "Gradient Boosting (meta-learner)": y_pred_meta,
    }
    for name, preds in models_results.items():
        f1 = f1_score(y_test, preds)
        p  = precision_score(y_test, preds)
        r  = recall_score(y_test, preds)
        lines.append(f"  {name:<40}  F1={f1:.3f}  P={p:.3f}  R={r:.3f}")

    lines.append("\nDETAILED CLASSIFICATION REPORT — Ensemble")
    lines.append("-" * 40)
    lines.append(classification_report(y_test, y_pred_meta,
                                        target_names=["Honest", "Malicious"]))

    lines.append("\nPER-NODE REPUTATION SUMMARY (test set, last epoch per node)")
    lines.append("-" * 40)
    lines.append(f"  {'Node':<10} {'Malicious Prob':>16} {'Reputation':>12} {'Action':>12} {'Vote Weight':>12} {'True Label':>12}")
    lines.append("  " + "-" * 76)

    last_epochs = test_df.groupby("node_id").tail(1).reset_index(drop=True)
    last_indices = last_epochs.index.tolist()
    for i, (_, row) in enumerate(last_epochs.iterrows()):
        if i >= len(y_prob_meta):
            break
        prob = y_prob_meta[i]
        rep, action, weight = compute_reputation_and_action(prob)
        true_label = "MALICIOUS" if row["is_malicious"] == 1 else "honest"
        lines.append(
            f"  node_{int(row['node_id']):02d}     {prob:>16.4f} {rep:>12.4f} {action:>12} {weight:>12.1f} {true_label:>12}"
        )

    lines.append("\nOUTPUT FILES")
    lines.append("-" * 40)
    lines.append("  models/rf_model.pkl         — Random Forest")
    lines.append("  models/iso_model.pkl        — Isolation Forest")
    lines.append("  models/gb_meta_model.pkl    — Gradient Boosting meta-learner")
    lines.append("  models/scaler.pkl           — StandardScaler")
    lines.append("  models/feature_cols.json    — Feature column names")
    lines.append("  confusion_matrix.png        — Confusion matrix plot")
    lines.append("  feature_importance.png      — RF feature importance plot")
    lines.append("=" * 60)

    report = "\n".join(lines)
    with open("ml_report.txt", "w") as f:
        f.write(report)

    print(report)
    print_step("Full report saved -> ml_report.txt")


# ─────────────────────────────────────────────
#  INFERENCE FUNCTION (use this in your nodes)
# ─────────────────────────────────────────────
def predict_node(feature_dict):
    """
    Call this from your FastAPI node to get a malicious probability
    and reputation action for a single node at runtime.

    Args:
        feature_dict: dict with keys matching FEATURE_COLS

    Returns:
        {
          "malicious_probability": 0.87,
          "reputation_score": 0.13,
          "action": "SLASH",
          "vote_weight": 0.0
        }

    Example:
        result = predict_node({
            "accuracy": 0.33,
            "false_positive_rate": 0.67,
            ...
        })
    """
    scaler    = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    rf_model  = joblib.load(f"{MODELS_DIR}/rf_model.pkl")
    iso_model = joblib.load(f"{MODELS_DIR}/iso_model.pkl")
    gb_model  = joblib.load(f"{MODELS_DIR}/gb_meta_model.pkl")

    with open(f"{MODELS_DIR}/feature_cols.json") as f:
        feature_cols = json.load(f)

    x = np.array([[feature_dict[col] for col in feature_cols]])
    x_scaled = scaler.transform(x)

    # Model 1 — RF
    rf_prob = rf_model.predict_proba(x_scaled)[0, 1]
    rf_pred = int(rf_model.predict(x_scaled)[0])

    # Model 2 — ISO
    iso_sc   = iso_model.score_samples(x_scaled)[0]
    iso_pred = int(iso_model.predict(x_scaled)[0] == -1)
    # Normalise (using a fixed reference range from training)
    iso_prob = float(np.clip(1 - (iso_sc + 0.5), 0, 1))

    # Model 3 — Graph (peer deviation; requires peers — use 0.5 as fallback at runtime)
    graph_pred = 0
    graph_prob = 0.5

    # Meta-learner
    meta_x     = np.array([[rf_pred, rf_prob, iso_pred, iso_prob, graph_pred, graph_prob]])
    final_prob = float(gb_model.predict_proba(meta_x)[0, 1])

    rep, action, weight = compute_reputation_and_action(final_prob)

    return {
        "malicious_probability": round(final_prob, 4),
        "reputation_score":      rep,
        "action":                action,
        "vote_weight":           weight,
        "model_breakdown": {
            "random_forest_prob":   round(rf_prob, 4),
            "isolation_forest_prob": round(iso_prob, 4),
            "graph_anomaly_prob":   round(graph_prob, 4),
        }
    }


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print_section("ML ENSEMBLE PIPELINE — Malicious Node Detection")

    # 1. Load
    X_train, y_train, X_test, y_test, train_df, test_df = load_data()

    # 2. Scale
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # 3. Model 1 — Random Forest
    rf_model, y_pred_rf, y_prob_rf = train_random_forest(
        X_train_sc, y_train, X_test_sc, y_test
    )

    # 4. Model 2 — Isolation Forest
    iso_model, y_pred_iso, y_prob_iso = train_isolation_forest(
        X_train_sc, y_train, X_test_sc, y_test
    )

    # 5. Model 3 — Graph Anomaly
    y_pred_graph, y_prob_graph = compute_graph_anomaly_scores(
        test_df, X_test_sc, y_test
    )

    # 6. Meta-learner
    gb_model, y_pred_meta, y_prob_meta, X_meta_test = train_meta_learner(
        X_train_sc, y_train, X_test_sc, y_test,
        rf_model, iso_model, test_df,
        X_train_sc, X_test_sc, train_df,
    )

    # 7. Save feature columns for inference
    with open(f"{MODELS_DIR}/feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    # 8. Plots
    plot_confusion_matrix(y_test, y_pred_meta)
    plot_feature_importance(rf_model)

    # 9. Report
    save_report(
        y_test, y_pred_rf, y_pred_iso, y_pred_graph, y_pred_meta,
        y_prob_meta, test_df
    )

    print_section("ALL DONE")
    print("  Models saved in ./models/")
    print("  Use predict_node() function to score nodes at runtime.")
    print()
    print("  Quick inference example:")
    print("    from ml_pipeline import predict_node")
    print("    result = predict_node({'accuracy': 0.33, 'false_positive_rate': 0.67, ...})")
    print("    print(result)  # -> {'action': 'SLASH', 'malicious_probability': 0.91, ...}")
    print()


if __name__ == "__main__":
    main()
