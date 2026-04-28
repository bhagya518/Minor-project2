#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
    return MitigationDecision(status="MALICIOUS", action="SLASHED", shard="SLASHED")


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _normalize_0_1(arr: np.ndarray, mn: float, mx: float) -> np.ndarray:
    return (arr - mn) / (mx - mn + 1e-12)


def _build_single_row_from_args(args: argparse.Namespace, feature_cols: List[str]) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for c in feature_cols:
        v = getattr(args, c, None)
        if v is not None:
            row[c] = float(v)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one node through RF+IF -> reputation -> mitigation action.")
    parser.add_argument("--data", default="backbone_dataset_10k_final.xlsx")
    parser.add_argument("--rf", default="rf_backbone.joblib")
    parser.add_argument("--iso", default="iso_backbone.joblib")
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--node_id", type=int, default=None)
    parser.add_argument("--random", action="store_true", help="Pick a random row from the dataset as the node input")
    parser.add_argument("--row_index", type=int, default=None, help="Use a specific dataset row index as the node input")

    args, unknown = parser.parse_known_args()

    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    rf_art: Dict = joblib.load(args.rf)
    iso_art: Dict = joblib.load(args.iso)

    rf_model = rf_art["model"]
    rf_feature_cols: List[str] = list(rf_art["feature_cols"])
    scaler_type = rf_art.get("scaler_type", "standard")

    iso_model = iso_art["model"]
    behavioral_cols: List[str] = list(iso_art["behavioral_cols"])
    beh_scaler = iso_art["scaler"]

    df = pd.read_excel(args.data, sheet_name=sheet)

    used_features = [c for c in rf_feature_cols if c in df.columns]
    if not used_features:
        raise ValueError("No RF features found in dataset.")

    # Allow passing feature values via CLI as --feature_name 0.123
    for c in used_features:
        if not any(a.startswith(f"--{c}") for a in unknown):
            continue
        parser.add_argument(f"--{c}", dest=c, type=float, default=None)

    args = parser.parse_args()

    node_row: pd.Series | None = None
    if args.row_index is not None:
        node_row = df.iloc[int(args.row_index)]
    elif args.random:
        node_row = df.sample(n=1, random_state=np.random.randint(0, 1_000_000)).iloc[0]

    manual_row = _build_single_row_from_args(args, used_features)

    if node_row is None and not manual_row:
        raise ValueError("Provide --random, --row_index, or manual feature values like --itt_jitter 0.2")

    # Build a single-row dataframe for prediction
    base_df = df[used_features].copy()
    base_df = _coerce_numeric(base_df, used_features)
    medians = {c: (0.0 if pd.isna(base_df[c].median()) else float(base_df[c].median())) for c in used_features}

    input_vals: Dict[str, float] = medians.copy()
    if node_row is not None:
        for c in used_features:
            v = pd.to_numeric(node_row.get(c, np.nan), errors="coerce")
            if not pd.isna(v):
                input_vals[c] = float(v)

    for c, v in manual_row.items():
        input_vals[c] = float(v)

    node_df = pd.DataFrame([input_vals], columns=used_features)

    # Scale like the training scripts did (fit on whole dataset for this demo)
    base_df = base_df.fillna(pd.Series(medians))
    X_base = base_df.values.astype(float)
    X_node = node_df.values.astype(float)

    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaler.fit(X_base)
    X_node_scaled = scaler.transform(X_node)

    rf_prob = float(rf_model.predict_proba(X_node_scaled)[:, 1][0])

    # ISO on behavioral features
    behavioral_used = [c for c in behavioral_cols if c in df.columns]
    beh_base = df[behavioral_used].copy()
    beh_base = _coerce_numeric(beh_base, behavioral_used)
    beh_medians = {c: (0.0 if pd.isna(beh_base[c].median()) else float(beh_base[c].median())) for c in behavioral_used}
    beh_base = beh_base.fillna(pd.Series(beh_medians))

    beh_node_vals = {c: beh_medians[c] for c in behavioral_used}
    if node_row is not None:
        for c in behavioral_used:
            v = pd.to_numeric(node_row.get(c, np.nan), errors="coerce")
            if not pd.isna(v):
                beh_node_vals[c] = float(v)
    for c in behavioral_used:
        if c in manual_row:
            beh_node_vals[c] = float(manual_row[c])

    X_beh_base = beh_scaler.transform(beh_base.values.astype(float))
    X_beh_node = beh_scaler.transform(pd.DataFrame([beh_node_vals], columns=behavioral_used).values.astype(float))

    iso_scores_base = -iso_model.decision_function(X_beh_base)
    mn, mx = float(np.min(iso_scores_base)), float(np.max(iso_scores_base))
    iso_score = float(-iso_model.decision_function(X_beh_node)[0])
    iso_norm = float(_normalize_0_1(np.array([iso_score]), mn, mx)[0])

    risk = (0.7 * rf_prob) + (0.3 * iso_norm)
    reputation = 1.0 - float(np.clip(risk, 0.0, 1.0))

    decision = apply_mitigation_policy(reputation)

    node_id = args.node_id
    if node_id is None and node_row is not None and "node_id" in df.columns:
        try:
            node_id = int(node_row.get("node_id"))
        except Exception:
            node_id = None

    node_id_str = str(node_id) if node_id is not None else "<unknown>"

    # A simple explanation string based on dominant contributor
    reason = "RF malicious probability" if (0.7 * rf_prob) >= (0.3 * iso_norm) else "behavioral anomaly score"

    print(
        f"Node {node_id_str} evaluated -> rf_prob={rf_prob:.3f}, iso_anomaly={iso_norm:.3f}, "
        f"reputation={reputation:.3f} -> STATUS: {decision.status} -> Action: {decision.action} (Shard: {decision.shard})"
    )
    print(f"Reason (dominant signal): {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
