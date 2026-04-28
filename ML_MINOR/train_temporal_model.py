#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def reshape_to_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    epochs: int = 50,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    if "node_id" not in df.columns or "epoch" not in df.columns:
        raise ValueError("Dataset must contain node_id and epoch for temporal reshaping")

    df_sorted = df.sort_values(["node_id", "epoch"]).copy()

    nodes = df_sorted["node_id"].dropna().unique().tolist()
    X_list = []
    y_list = []

    for node in nodes:
        g = df_sorted[df_sorted["node_id"] == node]
        g = g.sort_values("epoch")

        feats = _coerce_numeric(g[feature_cols].copy(), feature_cols).fillna(0.0).values.astype(float)
        if feats.shape[0] < epochs:
            pad = np.zeros((epochs - feats.shape[0], feats.shape[1]), dtype=float)
            feats = np.vstack([feats, pad])
        else:
            feats = feats[:epochs]

        X_list.append(feats)
        y_list.append(int(pd.to_numeric(g["label"].iloc[0], errors="coerce") or 0))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    return X, y, nodes


class LSTMTemporalClassifier(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last).squeeze(-1)
        return logits


def train_lstm(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    model.to(device)
    model.train()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    ds = torch.utils.data.TensorDataset(X_t, y_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()


def predict_proba(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def main() -> int:
    parser = argparse.ArgumentParser(description="Train temporal LSTM model on epoch sequences.")
    parser.add_argument("--data", default="backbone_dataset_10k_final.xlsx")
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--num_nodes", type=int, default=200)
    parser.add_argument("--epochs_per_node", type=int, default=50)
    parser.add_argument("--train_prefix", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lstm_epochs", type=int, default=25)
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

    feature_cols = [
        "peer_agreement_rate",
        "ssl_accuracy",
        "avg_rt_error",
        "report_consistency",
        "sudden_change_score",
        "blocks_mined",
        "orphan_blocks",
        "tx_submitted",
        "itt_jitter",
        "ewma_trust_score",
        "bayesian_confidence",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X_seq, y_node, nodes = reshape_to_sequences(df, feature_cols, epochs=args.epochs_per_node)

    # Scale features globally (fit on all timesteps of all nodes) to stabilize LSTM training
    scaler = StandardScaler()
    X_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_seq_scaled = X_flat_scaled.reshape(X_seq.shape)

    node_ids = np.array(nodes)
    train_nodes, test_nodes = train_test_split(
        node_ids,
        test_size=0.2,
        random_state=args.seed,
        stratify=y_node if len(np.unique(y_node)) > 1 else None,
    )

    train_mask = np.isin(node_ids, train_nodes)
    test_mask = np.isin(node_ids, test_nodes)

    X_train_full = X_seq_scaled[train_mask]
    y_train = y_node[train_mask]
    X_test_full = X_seq_scaled[test_mask]
    y_test = y_node[test_mask]

    # LSTM training uses only first 40 epochs (prefix); test uses full 50
    prefix = int(args.train_prefix)
    X_train_prefix = X_train_full.copy()
    X_train_prefix[:, prefix:, :] = 0.0

    model = LSTMTemporalClassifier(num_features=X_seq.shape[-1])
    train_lstm(model, X_train_prefix, y_train, epochs=args.lstm_epochs, device="cpu")

    lstm_prob = predict_proba(model, X_test_full, device="cpu")
    y_pred_lstm = (lstm_prob >= 0.5).astype(int)

    print("\n=== LSTM Evaluation (node-level, test set) ===")
    print(classification_report(y_test, y_pred_lstm, digits=4))
    lstm_acc = accuracy_score(y_test, y_pred_lstm)
    lstm_f1 = f1_score(y_test, y_pred_lstm, zero_division=0)
    print(f"LSTM Accuracy: {lstm_acc:.4f}")
    print(f"LSTM F1-score:  {lstm_f1:.4f}")

    # Baseline RF on per-epoch samples, aggregated to node-level
    X_rows = df[feature_cols].copy()
    X_rows = _coerce_numeric(X_rows, feature_cols).fillna(0.0).values.astype(float)
    X_rows_scaled = scaler.transform(X_rows)

    y_rows = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).values

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_rows_scaled, y_rows)

    rf_prob_rows = rf.predict_proba(X_rows_scaled)[:, 1]
    df_rf = df[["node_id", "epoch", "label"]].copy()
    df_rf["rf_prob"] = rf_prob_rows

    # Aggregate RF prob at node level (mean over last 10 epochs)
    df_rf["is_last_window"] = df_rf["epoch"] >= (args.epochs_per_node - 10)
    rf_node_prob = df_rf[df_rf["is_last_window"]].groupby("node_id")["rf_prob"].mean()

    rf_prob_node_test = np.array([rf_node_prob.get(n, 0.0) for n in test_nodes], dtype=float)
    y_pred_rf_node = (rf_prob_node_test >= 0.5).astype(int)
    rf_f1_node = f1_score(y_test, y_pred_rf_node, zero_division=0)

    print("\n=== Random Forest Baseline (node-level, test set) ===")
    print(f"RF node-level F1-score: {rf_f1_node:.4f}")

    # Hybrid scoring: combine immediate RF (late-epoch mean) with LSTM temporal consistency
    w_rf = 0.5
    w_lstm = 0.5
    risk = (w_rf * rf_prob_node_test) + (w_lstm * lstm_prob)
    reputation = 1.0 - np.clip(risk, 0.0, 1.0)

    print("\n=== Hybrid Reputation (test set) ===")
    print(pd.Series(reputation).describe())

    joblib.dump(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "scaler": scaler,
            "num_features": int(X_seq.shape[-1]),
            "epochs_per_node": int(args.epochs_per_node),
            "train_prefix": int(args.train_prefix),
        },
        "lstm_backbone.joblib",
    )
    joblib.dump(
        {
            "model": rf,
            "feature_cols": feature_cols,
            "scaler": scaler,
        },
        "rf_temporal_baseline.joblib",
    )

    print("\nSaved temporal artifacts: lstm_backbone.joblib, rf_temporal_baseline.joblib")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
