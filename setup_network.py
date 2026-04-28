"""
setup_network.py
Run this ONCE after starting all 4 nodes.
1. Fetches each node's real public key from /health
2. Registers every node with every other node (full mesh)
3. Injects repeated malicious reports from node_d
4. Prints live consensus results every 30 seconds
"""

import requests
import time
import json
from itertools import permutations

NODES = {
    "node_a":    "http://localhost:8005",
    "node_b":    "http://localhost:8006",
    "node_c":    "http://localhost:8007",
    "node_d": "http://localhost:8008",
}

MONITORED_URL = "https://httpbin.org/get"

# ── Step 1: fetch real public keys ────────────────────────────────────────────
print("\n=== STEP 1: Fetching public keys ===")
pubkeys = {}
for node_id, base_url in NODES.items():
    try:
        h = requests.get(f"{base_url}/health", timeout=5).json()
        pubkeys[node_id] = h["public_key"]
        print(f"  {node_id}: {pubkeys[node_id][:24]}...")
    except Exception as e:
        print(f"  ERROR fetching {node_id}: {e}")

missing = [n for n in NODES if n not in pubkeys]
if missing:
    print(f"\nERROR: Could not reach nodes: {missing}")
    print("Make sure all 4 nodes are running before running this script.")
    exit(1)

# ── Step 2: full mesh registration ───────────────────────────────────────────
print("\n=== STEP 2: Full mesh peer registration ===")
for (src_id, src_url), (dst_id, dst_url) in permutations(NODES.items(), 2):
    payload = {
        "node_id":        src_id,
        "url":            src_url,
        "public_key_hex": pubkeys[src_id],
    }
    try:
        r = requests.post(f"{dst_url}/peers/register", json=payload, timeout=5)
        result = r.json()
        status = result.get("status", "?")
        print(f"  {src_id} -> {dst_id}: {status}")
    except Exception as e:
        print(f"  {src_id} -> {dst_id}: ERROR {e}")

print("\n[OK] Full mesh registered. Waiting 5s for nodes to sync...")
time.sleep(5)

# ── Step 3: verify mesh ───────────────────────────────────────────────────────
print("\n=== STEP 3: Verifying peer registration ===")
for node_id, base_url in NODES.items():
    try:
        r = requests.get(f"{base_url}/peers/registered", timeout=5).json()
        peers = list(r.get("peers", {}).keys())
        peer_count = len(peers)
        expected = [n for n in NODES if n != node_id]
        missing_peers = [p for p in expected if p not in peers]
        if missing_peers:
            print(f"  {node_id}: WARNING missing {missing_peers}")
        else:
            print(f"  {node_id}: {peer_count} peers {'OK' if peer_count >= 3 else 'FAIL'}")
    except Exception as e:
        print(f"  {node_id}: ERROR {e}")

# ── Step 4: inject malicious reports from node_d ──────────────────────────
print("\n=== STEP 4: Injecting malicious reports from node_d ===")
print("node_d will report the site as DOWN (lie) to all honest nodes.")
print("Honest nodes (a, b, c) report UP -> majority=UP -> node_d gets SLASHED")
print()

honest_nodes = ["node_a", "node_b", "node_c"]
honest_ports = [8005, 8006, 8007]

for round_num in range(1, 4):
    epoch = int(time.time() // 60)
    print(f"--- Round {round_num} (epoch {epoch}) ---")

    for port in honest_ports:
        payload = {
            "node_address": "node_d",
            "url":          MONITORED_URL,
            "is_reachable": False,        # LIE: says site is DOWN
            "status_code":  0,
            "response_ms":  9999.0,
            "ssl_valid":    False,
            "timestamp":    time.time(),
            "epoch_id":     epoch,
            "signature":    pubkeys["node_d"],   # real pubkey as sig token
            "report_hash":  f"malicious_{round_num}_{port}",
            "content_hash": "000000000000",
        }
        try:
            r = requests.post(f"http://localhost:{port}/report",
                              json=payload, timeout=5)
            print(f"  -> port {port}: {r.json().get('status','?')}")
        except Exception as e:
            print(f"  -> port {port}: ERROR {e}")

    if round_num < 3:
        print(f"  Waiting 5s before next round...")
        time.sleep(5)

print()
print("OK Malicious reports injected. Waiting 15s for consensus to process...")
time.sleep(15)

# ── Step 5: print live results ────────────────────────────────────────────────
print("\n=== STEP 5: Live consensus results ===")

for node_id, base_url in NODES.items():
    print(f"\n--- {node_id} ({base_url}) ---")

    try:
        verdict = requests.get(f"{base_url}/verdict", timeout=5).json()
        verdicts = verdict.get("verdicts", {})
        reps     = verdict.get("node_reputations", {})

        if verdicts:
            latest_epoch = max(verdicts.keys())
            v = verdicts[latest_epoch]
            print(f"  Latest epoch:    {latest_epoch}")
            print(f"  Majority verdict:{v.get('majority_verdict', '—')}")
            print(f"  Honest nodes:    {v.get('honest', [])}")
            print(f"  Slashed nodes:   {v.get('slashed', [])}")
        else:
            print("  No verdicts yet")

        if reps:
            print("  Reputations:")
            for nid, score in reps.items():
                action = "✅ ALLOW" if score >= 0.8 else "⚠️  WARN" if score >= 0.6 else "🚫 SLASH"
                print(f"    {nid}: {score:.4f}  {action}")
    except Exception as e:
        print(f"  ERROR: {e}")

    try:
        reports = requests.get(f"{base_url}/reports/latest?limit=5", timeout=5).json()
        count = reports.get("total_available", 0)
        print(f"  Total reports stored: {count}")
    except Exception:
        pass

print("\n=== DONE ===")
print("Dashboard -> Multi-Node tab should now show:")
print("  - node_d with low reputation and SLASH action")
print("  - Honest nodes with reputation ~0.97-0.99")
print("  - Majority verdict: 'up'")
print("  - Slashed: ['node_d']")
print()
print("Blockchain tx hashes will appear in the Hardhat terminal.")
print("Run this script again any time to re-inject malicious reports.")
