#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os


@dataclass(frozen=True)
class MitigationDecision:
    status: str
    action: str
    shard: str


def apply_mitigation_policy(reputation_score: float) -> MitigationDecision:
    if reputation_score > 0.8:
        return MitigationDecision(status="HEALTHY", action="ALLOW", shard="PRIMARY")
    if 0.5 < reputation_score <= 0.8:
        return MitigationDecision(status="SUSPICIOUS", action="WARN", shard="MONITORING")
    if 0.2 < reputation_score <= 0.5:
        return MitigationDecision(status="FAULTY", action="QUARANTINE", shard="QUARANTINE")
    return MitigationDecision(status="MALICIOUS", action="SLASH", shard="SLASHED")


def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _normalize_0_1(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    return (arr - mn) / (mx - mn + 1e-12)


def ewma_reputation_by_node(
    df_scores: pd.DataFrame,
    node_col: str,
    time_col: str,
    reputation_col: str,
    alpha: float,
) -> pd.Series:
    def _ewma_for_group(g: pd.DataFrame) -> pd.Series:
        g_sorted = g.sort_values(time_col)
        vals = g_sorted[reputation_col].astype(float).values
        out = np.empty_like(vals, dtype=float)
        if len(vals) == 0:
            return pd.Series([], index=g_sorted.index, dtype=float)
        out[0] = vals[0]
        for i in range(1, len(vals)):
            out[i] = alpha * out[i - 1] + (1.0 - alpha) * vals[i]
        return pd.Series(out, index=g_sorted.index)

    return df_scores.groupby(node_col, group_keys=False).apply(_ewma_for_group)


def load_artifacts(
    rf_path: str,
    iso_path: str,
) -> Tuple[Dict, Dict]:
    rf_art = joblib.load(rf_path)
    iso_art = joblib.load(iso_path)
    return rf_art, iso_art


def prepare_full_scaled_features(
    df: pd.DataFrame,
    feature_cols,
    scaler_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    used = [c for c in feature_cols if c in df.columns]
    if not used:
        raise ValueError("No requested features exist in dataset.")

    feat_df = _coerce_numeric(df[used].copy(), used)
    for c in used:
        if feat_df[c].isna().all():
            feat_df[c] = 0.0
        else:
            feat_df[c] = feat_df[c].fillna(feat_df[c].median())

    X = feat_df.values.astype(float)
    if scaler_type == "standard":
        X_scaled = StandardScaler().fit_transform(X)
    elif scaler_type == "minmax":
        X_scaled = MinMaxScaler().fit_transform(X)
    else:
        raise ValueError("Unknown scaler_type in RF artifact")

    return X_scaled, np.array(used, dtype=object)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mitigation + sharding engine based on reputation scores.")
    parser.add_argument("--data", default="backbone_dataset_10k_final.xlsx")
    parser.add_argument("--rf", default="rf_backbone.joblib")
    parser.add_argument("--iso", default="iso_backbone.joblib")
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--plot", default=os.path.join("results_output", "mitigation_summary.png"))
    args = parser.parse_args()

    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    rf_art, iso_art = load_artifacts(args.rf, args.iso)
    rf_model = rf_art["model"]
    feature_cols = rf_art["feature_cols"]
    scaler_type = rf_art.get("scaler_type", "standard")

    iso_model = iso_art["model"]
    behavioral_cols = iso_art["behavioral_cols"]
    beh_scaler = iso_art["scaler"]

    df = pd.read_excel(args.data, sheet_name=sheet)

    if "label" not in df.columns and "is_malicious" in df.columns:
        df = df.copy()
        df["label"] = pd.to_numeric(df["is_malicious"], errors="coerce").fillna(0).astype(int)

    if "label" not in df.columns:
        raise KeyError("Neither 'label' nor 'is_malicious' was found for ground truth.")

    y_all = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).values

    X_all_scaled, used_features = prepare_full_scaled_features(df, feature_cols, scaler_type=scaler_type)

    X_train, X_test, y_train, y_test, idx_train, idx_test = model_selection.train_test_split(
        X_all_scaled,
        y_all,
        df.index.values,
        test_size=0.2,
        random_state=42,
        stratify=y_all if len(np.unique(y_all)) > 1 else None,
    )

    rf_prob_test = rf_model.predict_proba(X_test)[:, 1]

    beh_df_test = df.loc[idx_test, behavioral_cols].copy()
    beh_df_test = _coerce_numeric(beh_df_test, behavioral_cols)
    for c in behavioral_cols:
        med = beh_df_test[c].median()
        if pd.isna(med):
            med = 0.0
        beh_df_test[c] = beh_df_test[c].fillna(med)

    X_beh_test = beh_scaler.transform(beh_df_test.values.astype(float))
    iso_score_test = -iso_model.decision_function(X_beh_test)
    iso_norm_test = _normalize_0_1(iso_score_test)

    w_rf, w_iso = 0.7, 0.3
    risk = (w_rf * rf_prob_test) + (w_iso * iso_norm_test)
    base_rep = 1.0 - np.clip(risk, 0.0, 1.0)

    test_df = df.loc[idx_test].copy()
    test_df["rf_prob_malicious"] = rf_prob_test
    test_df["iso_anomaly_norm"] = iso_norm_test
    test_df["base_reputation"] = base_rep

    if "node_id" in test_df.columns and "epoch" in test_df.columns:
        test_df["final_reputation"] = ewma_reputation_by_node(
            test_df,
            node_col="node_id",
            time_col="epoch",
            reputation_col="base_reputation",
            alpha=0.9,
        )
    else:
        test_df["final_reputation"] = test_df["base_reputation"]

    decisions = test_df["final_reputation"].apply(apply_mitigation_policy)
    test_df["mitigation_status"] = decisions.apply(lambda d: d.status)
    test_df["mitigation_action"] = decisions.apply(lambda d: d.action)
    test_df["assigned_shard"] = decisions.apply(lambda d: d.shard)

    primary = test_df[test_df["assigned_shard"] == "PRIMARY"]
    non_primary = test_df[test_df["assigned_shard"] != "PRIMARY"]

    def honest_rate(frame: pd.DataFrame) -> float:
        if len(frame) == 0:
            return float("nan")
        return float((frame["label"] == 0).mean() * 100.0)

    shard_integrity_primary = honest_rate(primary)
    shard_integrity_non_primary = honest_rate(non_primary)

    print("=== Shard Integrity ===")
    print(f"Primary Shard honest %: {shard_integrity_primary:.2f}")
    print(f"Other Shards honest %:  {shard_integrity_non_primary:.2f}")

    counts = test_df["mitigation_status"].value_counts().reindex(
        ["HEALTHY", "SUSPICIOUS", "FAULTY", "MALICIOUS"], fill_value=0
    )
    print("\n=== Mitigation Counts (test set) ===")
    for k, v in counts.items():
        print(f"{k}: {int(v)}")

    slashed_count = int(counts.get("MALICIOUS", 0))
    print(f"\nSlashed (MALICIOUS) count: {slashed_count}")

    plt.figure(figsize=(9, 5))
    plt.bar(["Healthy", "Warn", "Quarantine", "Slash"], [
        int(counts.get("HEALTHY", 0)),
        int(counts.get("SUSPICIOUS", 0)),
        int(counts.get("FAULTY", 0)),
        int(counts.get("MALICIOUS", 0)),
    ])
    plt.title("Mitigation Category Counts (Test Set)")
    plt.xlabel("Category")
    plt.ylabel("Node Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
    plt.savefig(args.plot, dpi=200)
    print(f"\nSaved plot: {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
