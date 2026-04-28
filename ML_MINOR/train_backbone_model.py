#!/usr/bin/env python3

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn import model_selection
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def _resolve_label_column(df: pd.DataFrame, requested: str = "label") -> str:
    if requested in df.columns:
        return requested

    lowered = {c.lower(): c for c in df.columns}
    if requested.lower() in lowered:
        return lowered[requested.lower()]

    raise KeyError(
        "Could not find target column 'label' (case-insensitive). "
        f"Available columns: {list(df.columns)}"
    )


def _ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
    if "label" in df.columns:
        return df

    if "is_malicious" in df.columns:
        out = df.copy()
        out["label"] = pd.to_numeric(out["is_malicious"], errors="coerce").fillna(0).astype(int)
        return out

    return df


def synthesize_epoch_history(
    df: pd.DataFrame,
    num_nodes: int = 200,
    epochs_per_node: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    n_required = int(num_nodes) * int(epochs_per_node)
    if n_required <= 0:
        return df

    base = df.copy()
    if len(base) < n_required:
        extra = base.sample(n=n_required - len(base), replace=True, random_state=seed)
        base = pd.concat([base, extra], ignore_index=True)
    else:
        base = base.sample(n=n_required, replace=False, random_state=seed).reset_index(drop=True)

    base = base.reset_index(drop=True)
    base["node_id"] = (np.arange(n_required) % num_nodes).astype(int)
    base["epoch"] = (np.arange(n_required) // num_nodes).astype(int)

    if "label" in base.columns:
        node_label = base.groupby("node_id")["label"].apply(lambda s: int(pd.to_numeric(s.iloc[0], errors="coerce") or 0)).astype(int)
        base["label"] = base["node_id"].map(node_label).astype(int)

    if "is_malicious" in base.columns and "label" in base.columns:
        base["is_malicious"] = base["label"].astype(int)

    # Add small noise to avoid exact duplicates across epochs
    for col in ["peer_agreement_rate", "ssl_accuracy", "avg_rt_error", "report_consistency", "sudden_change_score", "itt_jitter"]:
        if col in base.columns:
            vals = pd.to_numeric(base[col], errors="coerce")
            jitter = rng.uniform(-0.01, 0.01, size=len(base))
            base[col] = np.clip(vals.fillna(vals.median()) + jitter, 0.0, 1.0)

    return base


def apply_strategic_byzantine_noise(
    df: pd.DataFrame,
    seed: int = 42,
    mimic_fraction_of_malicious: float = 0.40,
    mimic_honest_low: float = 0.85,
    mimic_honest_high: float = 0.95,
    mimic_lie_every_n_epochs: int = 10,
) -> pd.DataFrame:
    if "label" not in df.columns:
        return df
    if "node_id" not in df.columns or "epoch" not in df.columns:
        return df

    out = df.copy()
    rng = np.random.RandomState(seed)

    malicious_nodes = out.loc[out["label"] == 1, "node_id"].dropna().unique()
    if len(malicious_nodes) == 0:
        return out

    mimic_count = int(np.floor(len(malicious_nodes) * mimic_fraction_of_malicious))
    mimic_nodes = set(rng.choice(malicious_nodes, size=max(0, mimic_count), replace=False).tolist())

    # Jitter camouflage: force malicious jitter distribution to overlap honest jitter
    if "itt_jitter" in out.columns:
        honest_jitter = pd.to_numeric(out.loc[out["label"] == 0, "itt_jitter"], errors="coerce").dropna()
        if len(honest_jitter) > 10:
            lo = float(honest_jitter.quantile(0.05))
            hi = float(honest_jitter.quantile(0.95))
            mal_idx = out.index[out["label"] == 1]
            out.loc[mal_idx, "itt_jitter"] = rng.uniform(lo, hi, size=len(mal_idx))

    # Mimic attack: subset of malicious nodes appear honest most epochs, lie periodically
    for node in mimic_nodes:
        node_mask = (out["node_id"] == node) & (out["label"] == 1)
        if not node_mask.any():
            continue

        # most epochs look honest
        if "peer_agreement_rate" in out.columns:
            honest_like_mask = node_mask & ((out["epoch"].astype(int) % mimic_lie_every_n_epochs) != 0)
            out.loc[honest_like_mask, "peer_agreement_rate"] = rng.uniform(
                mimic_honest_low, mimic_honest_high, size=int(honest_like_mask.sum())
            )
            lie_mask = node_mask & ((out["epoch"].astype(int) % mimic_lie_every_n_epochs) == 0)
            out.loc[lie_mask, "peer_agreement_rate"] = rng.uniform(0.30, 0.60, size=int(lie_mask.sum()))

        if "ssl_accuracy" in out.columns:
            honest_like_mask = node_mask & ((out["epoch"].astype(int) % mimic_lie_every_n_epochs) != 0)
            out.loc[honest_like_mask, "ssl_accuracy"] = rng.uniform(
                mimic_honest_low, mimic_honest_high, size=int(honest_like_mask.sum())
            )
            lie_mask = node_mask & ((out["epoch"].astype(int) % mimic_lie_every_n_epochs) == 0)
            out.loc[lie_mask, "ssl_accuracy"] = rng.uniform(0.30, 0.60, size=int(lie_mask.sum()))

    # Sleeper attack: malicious nodes act honest for first half of their epoch history
    if ("avg_rt_error" in out.columns) or ("fraud_attempts" in out.columns):
        per_node_max_epoch = out.groupby("node_id")["epoch"].max()
        for node in malicious_nodes:
            if node not in per_node_max_epoch.index:
                continue
            max_ep = float(per_node_max_epoch.loc[node])
            split_ep = max_ep / 2.0
            early_mask = (out["node_id"] == node) & (out["label"] == 1) & (out["epoch"] <= split_ep)
            late_mask = (out["node_id"] == node) & (out["label"] == 1) & (out["epoch"] > split_ep)

            if early_mask.any():
                # force honest-like low error/low fraud early
                if "avg_rt_error" in out.columns:
                    out.loc[early_mask, "avg_rt_error"] = rng.uniform(0.00, 0.15, size=int(early_mask.sum()))
                if "fraud_attempts" in out.columns:
                    out.loc[early_mask, "fraud_attempts"] = 0

            if late_mask.any():
                # later epochs show malicious signals
                if "avg_rt_error" in out.columns:
                    out.loc[late_mask, "avg_rt_error"] = rng.uniform(0.55, 1.00, size=int(late_mask.sum()))
                if "fraud_attempts" in out.columns:
                    out.loc[late_mask, "fraud_attempts"] = rng.poisson(lam=2.0, size=int(late_mask.sum()))

    return out


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_backbone_xlsx(path: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)

    if isinstance(df, dict):
        raise ValueError(
            "read_excel returned multiple sheets; specify a single sheet via --sheet. "
            f"Sheets: {list(df.keys())}"
        )

    return df


def verify_and_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("\nPreview (first 5 rows):")
    print(df.head(5))

    print("\nColumn names:")
    print(list(df.columns))

    novelty_cols = ["itt_jitter", "ewma_trust_score", "bayesian_confidence"]
    missing_novelty = [c for c in novelty_cols if c not in df.columns]
    if missing_novelty:
        print("\nMissing Research Novelty attributes:")
        for c in missing_novelty:
            print(f"- {c}")
    else:
        print("\nResearch Novelty attributes detected with dtypes:")
        for c in novelty_cols:
            print(f"- {c}: {df[c].dtype}")

    cleaned = df.copy()
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in cleaned.columns if c not in numeric_cols]

    for c in numeric_cols:
        if cleaned[c].isna().any():
            med = cleaned[c].median()
            if pd.isna(med):
                med = 0.0
            cleaned[c] = cleaned[c].fillna(med)

    for c in non_numeric_cols:
        if cleaned[c].isna().any():
            mode_series = cleaned[c].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else ""
            cleaned[c] = cleaned[c].fillna(fill_val)

    print("\nDtypes after cleaning:")
    print(cleaned.dtypes)

    return cleaned


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


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    scaler: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    missing_features = [c for c in feature_cols if c not in df.columns]
    used_features = [c for c in feature_cols if c in df.columns]

    if len(used_features) == 0:
        raise ValueError(
            "None of the requested feature columns exist in the dataset. "
            f"Requested: {feature_cols}"
        )

    feat_df = df[used_features].copy()
    feat_df = _coerce_numeric(feat_df, used_features)

    for c in used_features:
        if feat_df[c].isna().all():
            feat_df[c] = 0.0
        else:
            feat_df[c] = feat_df[c].fillna(feat_df[c].median())

    X = feat_df.values.astype(float)

    if scaler == "standard":
        X_scaled = StandardScaler().fit_transform(X)
    elif scaler == "minmax":
        X_scaled = MinMaxScaler().fit_transform(X)
    else:
        raise ValueError("--scaler must be one of: standard, minmax")

    y_series = df[label_col]
    y = pd.to_numeric(y_series, errors="coerce").fillna(0).astype(int).values

    return X_scaled, y, used_features, missing_features


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare backbone dataset features/labels for training.")
    parser.add_argument(
        "--data",
        default="backbone_dataset_10k_final.xlsx",
        help="Path to backbone_dataset_10k_final.xlsx",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Excel sheet name or index (default: 0)",
    )
    parser.add_argument(
        "--scaler",
        default="standard",
        choices=["standard", "minmax"],
        help="Feature scaling method",
    )

    args = parser.parse_args()

    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    df_raw = load_backbone_xlsx(args.data, sheet_name=sheet)
    df_raw = _ensure_label_column(df_raw)
    df = verify_and_clean_dataset(df_raw)
    df = synthesize_epoch_history(df, num_nodes=200, epochs_per_node=50, seed=42)
    df = apply_strategic_byzantine_noise(df, seed=42)

    feature_cols = [
        # Monitoring
        "peer_agreement_rate",
        "ssl_accuracy",
        "avg_rt_error",
        "report_consistency",
        "sudden_change_score",
        # Blockchain
        "blocks_mined",
        "orphan_blocks",
        "gas_price",
        "tx_submitted",
        # Research Novelty
        "itt_jitter",
        "ewma_trust_score",
        "bayesian_confidence",
    ]

    label_col = _resolve_label_column(df, requested="label")

    X_all, y_all, used_features, missing_features = prepare_xy(
        df,
        feature_cols=feature_cols,
        label_col=label_col,
        scaler=args.scaler,
    )

    print(f"\nLoaded dataset: {args.data}")
    print(f"DataFrame shape: {df.shape}")
    if missing_features:
        print("Missing requested feature columns:")
        for c in missing_features:
            print(f"- {c}")

    print(f"Used feature columns ({len(used_features)}):")
    for c in used_features:
        print(f"- {c}")

    print(f"X shape: {X_all.shape}")
    print(f"y shape: {y_all.shape}")
    print(f"Target column: {label_col}")

    X_train, X_test, y_train, y_test, idx_train, idx_test = model_selection.train_test_split(
        X_all,
        y_all,
        df.index.values,
        test_size=0.2,
        random_state=42,
        stratify=y_all if len(np.unique(y_all)) > 1 else None,
    )

    print(f"\nSplit: train={X_train.shape[0]} test={X_test.shape[0]}")

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    rf_proba_test = rf.predict_proba(X_test)[:, 1]
    y_pred_test = (rf_proba_test >= 0.5).astype(int)

    print("\n=== Random Forest Evaluation (test set) ===")
    print(classification_report(y_test, y_pred_test, digits=4))
    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score:  {f1:.4f}")

    behavioral_cols = ["itt_jitter", "avg_rt_error", "sudden_change_score"]
    behavioral_used = [c for c in behavioral_cols if c in df.columns]
    if len(behavioral_used) == 0:
        raise ValueError(
            "None of the requested behavioral features are present for IsolationForest. "
            f"Requested: {behavioral_cols}"
        )

    beh_train_df = df.loc[idx_train, behavioral_used].copy()
    beh_test_df = df.loc[idx_test, behavioral_used].copy()
    beh_train_df = _coerce_numeric(beh_train_df, behavioral_used)
    beh_test_df = _coerce_numeric(beh_test_df, behavioral_used)
    for c in behavioral_used:
        med = beh_train_df[c].median()
        if pd.isna(med):
            med = 0.0
        beh_train_df[c] = beh_train_df[c].fillna(med)
        beh_test_df[c] = beh_test_df[c].fillna(med)

    beh_scaler = StandardScaler()
    X_beh_train = beh_scaler.fit_transform(beh_train_df.values.astype(float))
    X_beh_test = beh_scaler.transform(beh_test_df.values.astype(float))

    iso = IsolationForest(
        n_estimators=300,
        contamination=0.15,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_beh_train)
    iso_score_test = -iso.decision_function(X_beh_test)
    iso_norm_test = _normalize_0_1(iso_score_test)

    w_rf = 0.7
    w_iso = 0.3
    risk_raw = (w_rf * rf_proba_test) + (w_iso * iso_norm_test)
    base_reputation = 1.0 - np.clip(risk_raw, 0.0, 1.0)

    test_df = df.loc[idx_test].copy()
    test_df["rf_prob_malicious"] = rf_proba_test
    test_df["iso_anomaly_norm"] = iso_norm_test
    test_df["base_reputation"] = base_reputation

    node_col = "node_id" if "node_id" in test_df.columns else None
    time_col = "epoch" if "epoch" in test_df.columns else None
    ewma_alpha = 0.9
    if node_col is not None and time_col is not None:
        test_df["final_reputation"] = ewma_reputation_by_node(
            test_df,
            node_col=node_col,
            time_col=time_col,
            reputation_col="base_reputation",
            alpha=ewma_alpha,
        )
    else:
        test_df["final_reputation"] = test_df["base_reputation"]

    print("\n=== Final Reputation Score Distribution (test set) ===")
    desc = test_df["final_reputation"].describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
    print(desc)
    hist, bin_edges = np.histogram(test_df["final_reputation"].values.astype(float), bins=10, range=(0.0, 1.0))
    print("\nHistogram (10 bins from 0 to 1):")
    for i in range(len(hist)):
        print(f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}): {hist[i]}")

    joblib.dump(
        {
            "model": rf,
            "feature_cols": used_features,
            "scaler_type": args.scaler,
        },
        "rf_backbone.joblib",
    )
    joblib.dump(
        {
            "model": iso,
            "behavioral_cols": behavioral_used,
            "scaler": beh_scaler,
        },
        "iso_backbone.joblib",
    )
    print("\nSaved models: rf_backbone.joblib, iso_backbone.joblib")

    print("Data successfully loaded and verified for 10,000 tuples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
