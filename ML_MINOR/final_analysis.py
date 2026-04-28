#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import model_selection
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    shard_integrity_primary_honest_pct: float
    shard_integrity_other_honest_pct: float
    tps: float
    latency_s: float


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
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


def apply_mitigation_policy(reputation_score: float) -> str:
    if reputation_score > 0.8:
        return "HEALTHY"
    if 0.5 < reputation_score <= 0.8:
        return "SUSPICIOUS"
    if 0.2 < reputation_score <= 0.5:
        return "FAULTY"
    return "MALICIOUS"


def honest_rate(frame: pd.DataFrame, label_col: str) -> float:
    if len(frame) == 0:
        return float("nan")
    return float((frame[label_col] == 0).mean() * 100.0)


def simulate_throughput_latency(
    validators_per_shard: int,
    parallel_shards: int,
    base_compute_capacity: float = 500000.0,
    base_latency_s: float = 2.0,
) -> Tuple[float, float]:
    validators_per_shard = max(1, int(validators_per_shard))
    parallel_shards = max(1, int(parallel_shards))

    tps = base_compute_capacity * float(parallel_shards) / float(validators_per_shard)
    latency = base_latency_s * (float(validators_per_shard) / 1000.0)
    return float(tps), float(latency)


def compute_reputation_for_test_split(
    df: pd.DataFrame,
    rf_art: Dict,
    iso_art: Dict,
    seed: int = 42,
) -> pd.DataFrame:
    if "label" not in df.columns and "is_malicious" in df.columns:
        df = df.copy()
        df["label"] = pd.to_numeric(df["is_malicious"], errors="coerce").fillna(0).astype(int)

    if "label" not in df.columns:
        raise KeyError("Neither 'label' nor 'is_malicious' was found.")

    rf_model = rf_art["model"]
    feature_cols = rf_art["feature_cols"]
    scaler_type = rf_art.get("scaler_type", "standard")

    iso_model = iso_art["model"]
    behavioral_cols = iso_art["behavioral_cols"]
    beh_scaler = iso_art["scaler"]

    used = [c for c in feature_cols if c in df.columns]
    feat_df = _coerce_numeric(df[used].copy(), used)
    for c in used:
        if feat_df[c].isna().all():
            feat_df[c] = 0.0
        else:
            feat_df[c] = feat_df[c].fillna(feat_df[c].median())

    X = feat_df.values.astype(float)
    if scaler_type == "minmax":
        mn = np.min(X, axis=0)
        mx = np.max(X, axis=0)
        X_scaled = (X - mn) / (mx - mn + 1e-12)
    else:
        X_scaled = StandardScaler().fit_transform(X)

    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).values

    X_train, X_test, y_train, y_test, idx_train, idx_test = model_selection.train_test_split(
        X_scaled,
        y,
        df.index.values,
        test_size=0.2,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    rf_prob_test = rf_model.predict_proba(X_test)[:, 1]

    beh_test_df = df.loc[idx_test, behavioral_cols].copy()
    beh_test_df = _coerce_numeric(beh_test_df, behavioral_cols)
    for c in behavioral_cols:
        med = beh_test_df[c].median()
        if pd.isna(med):
            med = 0.0
        beh_test_df[c] = beh_test_df[c].fillna(med)

    X_beh_test = beh_scaler.transform(beh_test_df.values.astype(float))
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

    test_df["mitigation_status"] = test_df["final_reputation"].apply(apply_mitigation_policy)

    return test_df


def make_correlation_heatmap(df: pd.DataFrame, out_path: str) -> None:
    cols = []
    if "itt_jitter" in df.columns:
        cols.append("itt_jitter")
    if "ewma_trust_score" in df.columns:
        cols.append("ewma_trust_score")
    if "label" in df.columns:
        cols.append("label")

    if len(cols) < 2:
        return

    corr = df[cols].corr(numeric_only=True)

    plt.figure(figsize=(6, 4))
    plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(cols)), cols, rotation=30, ha="right")
    plt.yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.title("Correlation Heatmap (Research Features vs Label)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def main() -> int:
    parser = argparse.ArgumentParser(description="Final benchmarking + analysis script.")
    parser.add_argument("--data", default="backbone_dataset_10k_final.xlsx")
    parser.add_argument("--rf", default="rf_backbone.joblib")
    parser.add_argument("--iso", default="iso_backbone.joblib")
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--out_table", default=os.path.join("results_output", "benchmark_comparison.csv"))
    parser.add_argument("--out_heatmap", default=os.path.join("results_output", "research_feature_heatmap.png"))
    parser.add_argument("--out_stats", default=os.path.join("results_output", "project_final_stats.txt"))
    args = parser.parse_args()

    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    rf_art = joblib.load(args.rf)
    iso_art = joblib.load(args.iso)

    df = pd.read_excel(args.data, sheet_name=sheet)
    if "label" not in df.columns and "is_malicious" in df.columns:
        df = df.copy()
        df["label"] = pd.to_numeric(df["is_malicious"], errors="coerce").fillna(0).astype(int)

    test_df = compute_reputation_for_test_split(df, rf_art=rf_art, iso_art=iso_art, seed=42)

    slashed_count = int((test_df["mitigation_status"] == "MALICIOUS").sum())

    y_true = pd.to_numeric(test_df["label"], errors="coerce").fillna(0).astype(int).values
    y_pred = (test_df["rf_prob_malicious"].values >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    n_total = int(df.shape[0])
    n_test = int(test_df.shape[0])

    # Scenario A: Standard BFT (all nodes validate)
    # Validators per report shard = all nodes, no parallel shards
    tps_a, lat_a = simulate_throughput_latency(validators_per_shard=n_total, parallel_shards=1)
    shard_a_primary_honest = float((df["label"] == 0).mean() * 100.0)
    shard_a_other_honest = shard_a_primary_honest

    # Scenario B: Basic sharding (random shard assignment)
    shards_b = 10
    df_b = df.copy()
    rng = np.random.RandomState(42)
    df_b["shard_id"] = rng.randint(0, shards_b, size=len(df_b))
    primary_b = df_b[df_b["shard_id"] == 0]
    other_b = df_b[df_b["shard_id"] != 0]
    shard_b_primary_honest = honest_rate(primary_b, "label")
    shard_b_other_honest = honest_rate(other_b, "label")
    validators_per_shard_b = int(np.ceil(len(df_b) / shards_b))
    tps_b, lat_b = simulate_throughput_latency(validators_per_shard=validators_per_shard_b, parallel_shards=shards_b)

    # Scenario C: PoR sharding (reputation-based + mitigation)
    # Primary shard = HEALTHY nodes from the test-set mitigation; others are excluded/restricted.
    primary_c = test_df[test_df["mitigation_status"] == "HEALTHY"]
    other_c = test_df[test_df["mitigation_status"] != "HEALTHY"]
    shard_c_primary_honest = honest_rate(primary_c, "label")
    shard_c_other_honest = honest_rate(other_c, "label")

    shards_c = 10
    validators_per_shard_c = int(np.ceil(max(1, len(primary_c)) / shards_c))
    tps_c, lat_c = simulate_throughput_latency(validators_per_shard=validators_per_shard_c, parallel_shards=shards_c)

    results = [
        ScenarioResult(
            scenario="A_Standard_BFT",
            shard_integrity_primary_honest_pct=shard_a_primary_honest,
            shard_integrity_other_honest_pct=shard_a_other_honest,
            tps=tps_a,
            latency_s=lat_a,
        ),
        ScenarioResult(
            scenario="B_Basic_Sharding",
            shard_integrity_primary_honest_pct=shard_b_primary_honest,
            shard_integrity_other_honest_pct=shard_b_other_honest,
            tps=tps_b,
            latency_s=lat_b,
        ),
        ScenarioResult(
            scenario="C_PoR_Sharding",
            shard_integrity_primary_honest_pct=shard_c_primary_honest,
            shard_integrity_other_honest_pct=shard_c_other_honest,
            tps=tps_c,
            latency_s=lat_c,
        ),
    ]

    table = pd.DataFrame(
        [
            {
                "Scenario": r.scenario,
                "PrimaryShardHonestPct": round(r.shard_integrity_primary_honest_pct, 2),
                "OtherShardsHonestPct": round(r.shard_integrity_other_honest_pct, 2),
                "TPS": round(r.tps, 2),
                "LatencySeconds": round(r.latency_s, 4),
            }
            for r in results
        ]
    )

    table.to_csv(args.out_table, index=False)

    make_correlation_heatmap(df, args.out_heatmap)

    tps_improvement = ((tps_c - tps_a) / (tps_a + 1e-12)) * 100.0
    os.makedirs(os.path.dirname(args.out_table) or ".", exist_ok=True)
    with open(args.out_stats, "w", encoding="utf-8") as f:
        f.write("Project Final Statistics\n")
        f.write("========================\n\n")
        f.write(f"Test set size: {n_test}\n")
        f.write(f"Accuracy (RF on test): {acc:.4f}\n")
        f.write(f"F1-score (RF on test): {f1:.4f}\n")
        f.write(f"Slashed count (test mitigation): {slashed_count}\n\n")
        f.write("Benchmark TPS\n")
        f.write(f"Scenario A TPS: {tps_a:.2f}\n")
        f.write(f"Scenario B TPS: {tps_b:.2f}\n")
        f.write(f"Scenario C TPS: {tps_c:.2f}\n")
        f.write(f"TPS improvement (C vs A): {tps_improvement:.2f}%\n")

    print("=== Benchmark Comparison Table ===")
    print(table.to_string(index=False))
    print("\nScenario C TPS:", round(tps_c, 2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
