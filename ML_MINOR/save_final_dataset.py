#!/usr/bin/env python3

import argparse
import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def verify_and_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
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

    return cleaned


def ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
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
        node_label = (
            base.groupby("node_id")["label"]
            .apply(lambda s: int(pd.to_numeric(s.iloc[0], errors="coerce") or 0))
            .astype(int)
        )
        base["label"] = base["node_id"].map(node_label).astype(int)

    if "is_malicious" in base.columns and "label" in base.columns:
        base["is_malicious"] = base["label"].astype(int)

    for col in [
        "peer_agreement_rate",
        "ssl_accuracy",
        "avg_rt_error",
        "report_consistency",
        "sudden_change_score",
        "itt_jitter",
    ]:
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

    if "itt_jitter" in out.columns:
        honest_jitter = pd.to_numeric(out.loc[out["label"] == 0, "itt_jitter"], errors="coerce").dropna()
        if len(honest_jitter) > 10:
            lo = float(honest_jitter.quantile(0.05))
            hi = float(honest_jitter.quantile(0.95))
            mal_idx = out.index[out["label"] == 1]
            out.loc[mal_idx, "itt_jitter"] = rng.uniform(lo, hi, size=len(mal_idx))

    for node in mimic_nodes:
        node_mask = (out["node_id"] == node) & (out["label"] == 1)
        if not node_mask.any():
            continue

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
                if "avg_rt_error" in out.columns:
                    out.loc[early_mask, "avg_rt_error"] = rng.uniform(0.00, 0.15, size=int(early_mask.sum()))
                if "fraud_attempts" in out.columns:
                    out.loc[early_mask, "fraud_attempts"] = 0

            if late_mask.any():
                if "avg_rt_error" in out.columns:
                    out.loc[late_mask, "avg_rt_error"] = rng.uniform(0.55, 1.00, size=int(late_mask.sum()))
                if "fraud_attempts" in out.columns:
                    out.loc[late_mask, "fraud_attempts"] = rng.poisson(lam=2.0, size=int(late_mask.sum()))

    return out


def enforce_perfect_sleeper(
    df: pd.DataFrame,
    sleeper_node_ids: List[int],
    split_epoch: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.RandomState(seed)

    for nid in sleeper_node_ids:
        early = (out["node_id"] == nid) & (out["epoch"] < split_epoch)
        late = (out["node_id"] == nid) & (out["epoch"] >= split_epoch)

        out.loc[early, "label"] = 0
        out.loc[late, "label"] = 1
        if "is_malicious" in out.columns:
            out.loc[early, "is_malicious"] = 0
            out.loc[late, "is_malicious"] = 1

        if "peer_agreement_rate" in out.columns:
            out.loc[early, "peer_agreement_rate"] = 0.95
            out.loc[late, "peer_agreement_rate"] = 0.45

        if "ssl_accuracy" in out.columns:
            out.loc[early, "ssl_accuracy"] = 0.95
            out.loc[late, "ssl_accuracy"] = 0.50

        if "avg_rt_error" in out.columns:
            out.loc[early, "avg_rt_error"] = 0.05
            out.loc[late, "avg_rt_error"] = 0.85

        if "report_consistency" in out.columns:
            out.loc[early, "report_consistency"] = 0.95
            out.loc[late, "report_consistency"] = 0.40

        if "sudden_change_score" in out.columns:
            out.loc[early, "sudden_change_score"] = 0.05
            out.loc[late, "sudden_change_score"] = 0.85

        if "itt_jitter" in out.columns:
            out.loc[early, "itt_jitter"] = 0.10
            out.loc[late, "itt_jitter"] = 0.75

    sleeper_mask = out["node_id"].isin(sleeper_node_ids)
    for col in [
        "peer_agreement_rate",
        "ssl_accuracy",
        "avg_rt_error",
        "report_consistency",
        "sudden_change_score",
        "itt_jitter",
    ]:
        if col in out.columns:
            out.loc[sleeper_mask, col] = np.clip(
                pd.to_numeric(out.loc[sleeper_mask, col], errors="coerce").fillna(0.0).astype(float)
                + rng.uniform(-0.005, 0.005, size=int(sleeper_mask.sum())),
                0.0,
                1.0,
            )

    return out


def _try_rf_reputation_for_node(
    df: pd.DataFrame,
    node_id: int,
    rf_temporal_path: str,
    epochs_per_node: int,
) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    if not os.path.exists(rf_temporal_path):
        return None, None

    art = joblib.load(rf_temporal_path)
    rf_model = art["model"]
    scaler = art["scaler"]
    feature_cols = list(art["feature_cols"])

    g = df[df["node_id"] == node_id].sort_values("epoch")
    if g.empty:
        return None, None

    X = g[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    Xs = scaler.transform(X.values.astype(float))
    proba = rf_model.predict_proba(Xs)[:, 1]
    rep = 1.0 - np.clip(proba, 0.0, 1.0)
    epochs = g["epoch"].astype(int).tolist()

    if len(rep) != epochs_per_node:
        return rep, epochs

    return rep, epochs


def _heuristic_reputation_for_node(df: pd.DataFrame, node_id: int) -> Tuple[np.ndarray, List[int]]:
    g = df[df["node_id"] == node_id].sort_values("epoch")
    epochs = g["epoch"].astype(int).tolist()

    par = pd.to_numeric(g.get("peer_agreement_rate", pd.Series([0.8] * len(g))), errors="coerce").fillna(0.8).astype(float)
    rte = pd.to_numeric(g.get("avg_rt_error", pd.Series([0.2] * len(g))), errors="coerce").fillna(0.2).astype(float)
    ssc = pd.to_numeric(g.get("sudden_change_score", pd.Series([0.2] * len(g))), errors="coerce").fillna(0.2).astype(float)

    risk = 0.5 * (1.0 - np.clip(par, 0.0, 1.0)) + 0.3 * np.clip(rte, 0.0, 1.0) + 0.2 * np.clip(ssc, 0.0, 1.0)
    rep = 1.0 - np.clip(risk, 0.0, 1.0)

    return rep.values.astype(float), epochs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and save the final 10,000-row temporal dataset.")
    parser.add_argument("--data", default="backbone_dataset_10k_final.xlsx")
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--num_nodes", type=int, default=200)
    parser.add_argument("--epochs_per_node", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_master", default="blockchain_temporal_master_dataset.csv")
    parser.add_argument("--out_sleeper", default="sleeper_node_evidence.csv")
    parser.add_argument("--rf_temporal", default="rf_temporal_baseline.joblib")
    parser.add_argument("--out_plot", default=os.path.join("results_output", "sleeper_node3_reputation.png"))
    args = parser.parse_args()

    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    df = pd.read_excel(args.data, sheet_name=sheet)
    df = ensure_label_column(df)
    df = verify_and_clean_dataset(df)

    df = synthesize_epoch_history(df, num_nodes=args.num_nodes, epochs_per_node=args.epochs_per_node, seed=args.seed)
    df = apply_strategic_byzantine_noise(df, seed=args.seed)

    df = enforce_perfect_sleeper(df, sleeper_node_ids=[0, 1, 2, 3, 4], split_epoch=25, seed=args.seed)

    df.to_csv(args.out_master, index=False)

    df_sleeper3 = df[df["node_id"] == 3].sort_values("epoch").copy()
    df_sleeper3.to_csv(args.out_sleeper, index=False)

    honest_rows = int((df["label"].astype(int) == 0).sum())
    malicious_rows = int((df["label"].astype(int) == 1).sum())

    print("=== Final Dataset Saved ===")
    print(f"Master CSV:   {args.out_master}")
    print(f"Sleeper CSV:  {args.out_sleeper} (node_id=3)")
    print(f"Rows:         {len(df)} (expected {int(args.num_nodes) * int(args.epochs_per_node)})")
    print(f"Honest rows:  {honest_rows}")
    print(f"Malicious rows:{malicious_rows}")
    print("Reminder: This 2D CSV is converted to a 3D Tensor (200, 50, features) during training.")

    rep, epochs = _try_rf_reputation_for_node(
        df,
        node_id=3,
        rf_temporal_path=args.rf_temporal,
        epochs_per_node=int(args.epochs_per_node),
    )
    if rep is None or epochs is None:
        rep, epochs = _heuristic_reputation_for_node(df, node_id=3)

    os.makedirs(os.path.dirname(args.out_plot) or ".", exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(epochs, rep, linewidth=2, label="Reputation Score (Node 3)")
    plt.axvline(25, linestyle="--", linewidth=2, label="Sleeper flips @ Epoch 25")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Reputation Score")
    plt.title("Sleeper Evidence: Node 3 Reputation Drop")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=200)
    print(f"Saved plot:   {args.out_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
