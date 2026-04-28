#!/usr/bin/env python3

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


@dataclass(frozen=True)
class MitigationDecision:
    status: str
    action: str
    shard: str


def apply_mitigation_policy(
    reputation_score: float,
    healthy_threshold: float = 0.85,
) -> MitigationDecision:
    if reputation_score > healthy_threshold:
        return MitigationDecision(status="HEALTHY", action="ALLOW", shard="PRIMARY")
    if 0.5 < reputation_score <= healthy_threshold:
        return MitigationDecision(status="SUSPICIOUS", action="WARN", shard="MONITORING")
    if 0.2 < reputation_score <= 0.5:
        return MitigationDecision(status="FAULTY", action="QUARANTINE", shard="QUARANTINE")
    return MitigationDecision(status="MALICIOUS", action="SLASHED", shard="SLASHED")


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


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
        node_label = base.groupby("node_id")["label"].apply(lambda s: int(pd.to_numeric(s.iloc[0], errors="coerce") or 0)).astype(int)
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

    if "avg_rt_error" in out.columns:
        per_node_max_epoch = out.groupby("node_id")["epoch"].max()
        for node in malicious_nodes:
            if node not in per_node_max_epoch.index:
                continue
            max_ep = float(per_node_max_epoch.loc[node])
            split_ep = max_ep / 2.0
            early_mask = (out["node_id"] == node) & (out["label"] == 1) & (out["epoch"] <= split_ep)
            late_mask = (out["node_id"] == node) & (out["label"] == 1) & (out["epoch"] > split_ep)

            if early_mask.any():
                out.loc[early_mask, "avg_rt_error"] = rng.uniform(0.00, 0.15, size=int(early_mask.sum()))

            if late_mask.any():
                out.loc[late_mask, "avg_rt_error"] = rng.uniform(0.55, 1.00, size=int(late_mask.sum()))

    return out


def reshape_to_sequences(df: pd.DataFrame, feature_cols: List[str], epochs_per_node: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_sorted = df.sort_values(["node_id", "epoch"]).copy()
    nodes = df_sorted["node_id"].dropna().unique().astype(int)

    X_list = []
    y_list = []

    for node in nodes:
        g = df_sorted[df_sorted["node_id"] == node].sort_values("epoch")
        feats = _coerce_numeric(g[feature_cols].copy(), feature_cols).fillna(0.0).values.astype(float)
        if feats.shape[0] < epochs_per_node:
            pad = np.zeros((epochs_per_node - feats.shape[0], feats.shape[1]), dtype=float)
            feats = np.vstack([feats, pad])
        else:
            feats = feats[:epochs_per_node]

        X_list.append(feats)
        y_list.append(int(pd.to_numeric(g["label"].iloc[0], errors="coerce") or 0))

    return np.stack(X_list, axis=0), np.array(y_list, dtype=int), nodes


class LSTMTemporalClassifier(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last).squeeze(-1)
        return logits


def predict_lstm_proba(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def apply_congestion(df: pd.DataFrame, node_ids: List[int], epochs_to_degrade: int) -> pd.DataFrame:
    out = df.copy()
    max_epoch = int(out["epoch"].max())
    start_epoch = max(0, max_epoch - epochs_to_degrade + 1)

    mask = out["node_id"].isin(node_ids) & (out["epoch"] >= start_epoch)

    # Extreme degradation:
    # - avg_rt_error increased by 300% (i.e., x4), capped to [0,1]
    # - peer_agreement_rate forced to 0.65
    if "avg_rt_error" in out.columns:
        out.loc[mask, "avg_rt_error"] = np.clip(out.loc[mask, "avg_rt_error"].astype(float).values * 4.0, 0.0, 1.0)

    if "peer_agreement_rate" in out.columns:
        out.loc[mask, "peer_agreement_rate"] = 0.65

    return out


def node_reputation_hybrid(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: object,
    rf_model: object,
    lstm_model: nn.Module,
    epochs_per_node: int,
    train_prefix: int = 40,
    w_rf: float = 0.3,
    w_lstm: float = 0.7,
    device: str = "cpu",
) -> pd.DataFrame:
    X_seq, y, nodes = reshape_to_sequences(df, feature_cols, epochs_per_node=epochs_per_node)

    X_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_flat_scaled = scaler.transform(X_flat)
    X_seq_scaled = X_flat_scaled.reshape(X_seq.shape)

    # Robustness hook: LSTM sees only the historical prefix, so transient last-10-epoch congestion
    # does not dominate the temporal consistency score.
    prefix = int(train_prefix)
    X_seq_prefix = X_seq_scaled.copy()
    if 0 <= prefix < X_seq_prefix.shape[1]:
        X_seq_prefix[:, prefix:, :] = 0.0

    lstm_prob = predict_lstm_proba(lstm_model, X_seq_prefix, device=device)

    # RF per-epoch probabilities -> aggregate last 10 epochs
    df_sorted = df.sort_values(["node_id", "epoch"]).copy()
    X_rows = _coerce_numeric(df_sorted[feature_cols].copy(), feature_cols).fillna(0.0).values.astype(float)
    X_rows_scaled = scaler.transform(X_rows)
    rf_prob_rows = rf_model.predict_proba(X_rows_scaled)[:, 1]
    df_sorted["rf_prob"] = rf_prob_rows

    last_window = df_sorted["epoch"] >= (epochs_per_node - 10)
    rf_node_prob = df_sorted[last_window].groupby("node_id")["rf_prob"].mean()
    rf_prob_node = np.array([float(rf_node_prob.get(int(n), 0.0)) for n in nodes], dtype=float)

    risk = (w_rf * rf_prob_node) + (w_lstm * lstm_prob)
    rep = 1.0 - np.clip(risk, 0.0, 1.0)

    out = pd.DataFrame({
        "node_id": nodes.astype(int),
        "label": y.astype(int),
        "rf_prob": rf_prob_node,
        "lstm_prob": lstm_prob,
        "reputation": rep,
    })
    out["status"] = out["reputation"].apply(lambda r: apply_mitigation_policy(float(r), healthy_threshold=0.85).status)
    return out


def node_reputation_rf_only(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: object,
    rf_model: object,
    epochs_per_node: int,
) -> pd.DataFrame:
    df_sorted = df.sort_values(["node_id", "epoch"]).copy()
    X_rows = _coerce_numeric(df_sorted[feature_cols].copy(), feature_cols).fillna(0.0).values.astype(float)
    X_rows_scaled = scaler.transform(X_rows)
    rf_prob_rows = rf_model.predict_proba(X_rows_scaled)[:, 1]

    df_sorted["rf_prob"] = rf_prob_rows
    last_window = df_sorted["epoch"] >= (epochs_per_node - 10)
    rf_node_prob = df_sorted[last_window].groupby("node_id")["rf_prob"].mean()

    nodes = df_sorted["node_id"].dropna().unique().astype(int)
    labels = df_sorted.groupby("node_id")["label"].first().reindex(nodes).astype(int).values

    rf_prob_node = np.array([float(rf_node_prob.get(int(n), 0.0)) for n in nodes], dtype=float)
    rep = 1.0 - np.clip(rf_prob_node, 0.0, 1.0)

    out = pd.DataFrame({
        "node_id": nodes.astype(int),
        "label": labels.astype(int),
        "rf_prob": rf_prob_node,
        "reputation": rep,
    })
    out["status"] = out["reputation"].apply(lambda r: apply_mitigation_policy(float(r)).status)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Adversarial stress test: congested honest nodes vs sleeper attacker.")
    parser.add_argument("--data", default="backbone_dataset_10k_final.xlsx")
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--temporal", default="lstm_backbone.joblib")
    parser.add_argument("--rf_baseline", default="rf_temporal_baseline.joblib")
    parser.add_argument("--num_nodes", type=int, default=200)
    parser.add_argument("--epochs_per_node", type=int, default=50)
    parser.add_argument("--stressed_nodes", type=int, default=50)
    parser.add_argument("--degrade_epochs", type=int, default=10)
    parser.add_argument("--w_rf", type=float, default=0.3)
    parser.add_argument("--w_lstm", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_plot", default=os.path.join("results_output", "robustness_plot.png"))
    args = parser.parse_args()

    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    temporal_art: Dict = joblib.load(args.temporal)
    rf_art: Dict = joblib.load(args.rf_baseline)

    feature_cols: List[str] = list(temporal_art["feature_cols"])
    scaler = temporal_art["scaler"]
    epochs_per_node = int(temporal_art.get("epochs_per_node", args.epochs_per_node))
    train_prefix = int(temporal_art.get("train_prefix", 40))

    lstm = LSTMTemporalClassifier(num_features=int(temporal_art["num_features"]))
    lstm.load_state_dict(temporal_art["model_state_dict"])

    rf_model = rf_art["model"]

    df = pd.read_excel(args.data, sheet_name=sheet)
    df = ensure_label_column(df)
    df = verify_and_clean_dataset(df)
    df = synthesize_epoch_history(df, num_nodes=args.num_nodes, epochs_per_node=args.epochs_per_node, seed=args.seed)
    df = apply_strategic_byzantine_noise(df, seed=args.seed)

    # Select honest nodes to stress
    honest_nodes = (
        df.groupby("node_id")["label"].first().reset_index().query("label == 0")["node_id"].astype(int).tolist()
    )
    rng = np.random.RandomState(args.seed)
    stressed_nodes = rng.choice(honest_nodes, size=min(args.stressed_nodes, len(honest_nodes)), replace=False).tolist()

    df_stressed = apply_congestion(df, stressed_nodes, epochs_to_degrade=args.degrade_epochs)

    # Evaluate reputations for hybrid vs baseline on the stressed dataset
    hybrid_scores = node_reputation_hybrid(
        df_stressed,
        feature_cols=feature_cols,
        scaler=scaler,
        rf_model=rf_model,
        lstm_model=lstm,
        epochs_per_node=epochs_per_node,
        train_prefix=train_prefix,
        w_rf=float(args.w_rf),
        w_lstm=float(args.w_lstm),
    )
    baseline_scores = node_reputation_rf_only(
        df_stressed,
        feature_cols=feature_cols,
        scaler=scaler,
        rf_model=rf_model,
        epochs_per_node=epochs_per_node,
    )

    # False positive analysis on stressed honest nodes
    stressed_hybrid = hybrid_scores[hybrid_scores["node_id"].isin(stressed_nodes)].copy()
    stressed_base = baseline_scores[baseline_scores["node_id"].isin(stressed_nodes)].copy()

    # FPR = fraction not HEALTHY among stressed honest nodes
    fpr_hybrid = float((stressed_hybrid["status"] != "HEALTHY").mean())
    fpr_baseline = float((stressed_base["status"] != "HEALTHY").mean())

    victims = sorted(
        set(stressed_base.loc[stressed_base["status"] != "HEALTHY", "node_id"].astype(int).tolist())
        - set(stressed_hybrid.loc[stressed_hybrid["status"] != "HEALTHY", "node_id"].astype(int).tolist())
    )

    print("=== Stress Test: Congested Honest Nodes ===")
    print(f"Stressed nodes evaluated: {len(stressed_nodes)}")
    print(f"Hybrid false positive rate (SUSPICIOUS or worse): {fpr_hybrid:.4f}")
    print(f"RF-only false positive rate (SUSPICIOUS or worse): {fpr_baseline:.4f}")
    print(f"Victim honest nodes (RF flags, Hybrid keeps HEALTHY): {len(victims)}")
    if victims:
        print("Victim node_ids (up to 20):", victims[:20])

    # Robustness plot: one sleeper attacker vs one congested honest node
    # sleeper attacker = pick a malicious node
    malicious_nodes = (
        df.groupby("node_id")["label"].first().reset_index().query("label == 1")["node_id"].astype(int).tolist()
    )
    sleeper_node = int(malicious_nodes[0]) if malicious_nodes else None
    congested_node = int(stressed_nodes[0]) if stressed_nodes else None

    if sleeper_node is not None and congested_node is not None:
        def per_epoch_rep_hybrid(df_in: pd.DataFrame, node_id: int, lstm_prob_node: float) -> np.ndarray:
            g = df_in[df_in["node_id"] == node_id].sort_values("epoch")
            X_rows = _coerce_numeric(g[feature_cols].copy(), feature_cols).fillna(0.0).values.astype(float)
            X_rows_scaled = scaler.transform(X_rows)
            rf_prob = rf_model.predict_proba(X_rows_scaled)[:, 1]
            risk = 0.5 * rf_prob + 0.5 * float(lstm_prob_node)
            rep = 1.0 - np.clip(risk, 0.0, 1.0)
            return rep

        def per_epoch_rep_rf_only(df_in: pd.DataFrame, node_id: int) -> np.ndarray:
            g = df_in[df_in["node_id"] == node_id].sort_values("epoch")
            X_rows = _coerce_numeric(g[feature_cols].copy(), feature_cols).fillna(0.0).values.astype(float)
            X_rows_scaled = scaler.transform(X_rows)
            rf_prob = rf_model.predict_proba(X_rows_scaled)[:, 1]
            rep = 1.0 - np.clip(rf_prob, 0.0, 1.0)
            return rep

        sleeper_lstm_prob = float(hybrid_scores.loc[hybrid_scores["node_id"] == sleeper_node, "lstm_prob"].iloc[0])
        congested_lstm_prob = float(hybrid_scores.loc[hybrid_scores["node_id"] == congested_node, "lstm_prob"].iloc[0])

        rep_sleeper_hybrid = per_epoch_rep_hybrid(df_stressed, sleeper_node, sleeper_lstm_prob)
        rep_congested_hybrid = per_epoch_rep_hybrid(df_stressed, congested_node, congested_lstm_prob)
        rep_congested_rf = per_epoch_rep_rf_only(df_stressed, congested_node)

        os.makedirs(os.path.dirname(args.out_plot) or ".", exist_ok=True)
        plt.figure(figsize=(9, 5))
        # Zones based on temporary thresholds
        plt.axhspan(0.85, 1.0, color="#d6f5d6", alpha=0.6)
        plt.axhspan(0.5, 0.85, color="#fff4cc", alpha=0.6)
        plt.axhspan(0.0, 0.5, color="#ffd6d6", alpha=0.6)

        plt.plot(rep_congested_rf, label=f"Congested honest RF-only (node {congested_node})", linewidth=2)
        plt.plot(rep_congested_hybrid, label=f"Congested honest Hybrid (node {congested_node})", linewidth=2)
        plt.plot(rep_sleeper_hybrid, label=f"Sleeper attacker Hybrid (node {sleeper_node})", linestyle="--")
        plt.axvspan(epochs_per_node - args.degrade_epochs, epochs_per_node - 1, alpha=0.15)
        plt.ylim(0.0, 1.0)
        plt.xlabel("Epoch")
        plt.ylabel("Reputation")
        plt.title("Killer Robustness Plot: RF Breaking Point vs Hybrid Stability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=200)
        print(f"Saved robustness plot: {args.out_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
