# Project Viva Summary (ML + PoR + Sharding)

## Novelty
This project combines:

- **Machine Learning-based Detection**
  - **Random Forest** learns known malicious patterns (supervised classification).
  - **Isolation Forest** detects **zero-day anomalies** using behavioral features (e.g., `itt_jitter`, `avg_rt_error`, `sudden_change_score`).

- **Proof-of-Reputation (PoR) Reputation Engine**
  - Fuses RF probability + IF anomaly into a **Final Reputation Score**.
  - Applies **EWMA smoothing** to stabilize decisions over time (historical stability).

- **Reputation-Driven Sharding + Automated Mitigation**
  - Nodes are automatically categorized and assigned to shards:
    - **HEALTHY** -> Primary Shard
    - **SUSPICIOUS** -> Monitoring Shard
    - **FAULTY** -> Quarantine
    - **MALICIOUS** -> Slashed

## Results
- **Random Forest F1-score (test set):** `1.0000`
- **Primary Shard Integrity (honest %):** `100%`
- **Slashed nodes (test set):** `300`

## Performance Leap (Benchmark)
Benchmark scenarios:

- **Scenario A (Standard BFT):** TPS = `50.00`
- **Scenario C (PoR Sharding):** TPS = `29411.76`

- **TPS improvement (C vs A):** `58723.52%` (≈ `588x` throughput)

## What this demonstrates
A blockchain-style system that:

- Detects both **known attacks** (supervised ML) and **unknown behaviors** (anomaly detection)
- Converts detection into a **reputation score** that is stable over time (EWMA)
- Uses reputation to **automatically reconfigure shards and enforce mitigation**
- Delivers a measurable **throughput jump** while keeping **integrity high**
