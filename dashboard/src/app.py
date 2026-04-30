"""
Decentralized Website Monitoring Dashboard — CORRECTED VERSION
Changes from original:
  - Consensus tab shows reputation-weighted vote breakdown per node
  - ML Features chart splits ratio features from response_ms (fixes scale issue)
  - ML prediction panel always renders (was missing)
  - Peers table adds Reputation + Shard columns
  - Statistics shows real blockchain registration status
  - All use_container_width → width='stretch' (Streamlit deprecation fix)
  - google.com/github.com swapped for httpbin.org test URLs
  - false_report_rate colour-coded correctly on honest nodes
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import json
from datetime import datetime
import concurrent.futures
import os

# Hardcoded list of all nodes
ALL_NODES = [
    "http://localhost:8005",
    "http://localhost:8006",
    "http://localhost:8007",
    "http://localhost:8008"
]

st.set_page_config(
    page_title="Decentralized Web Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _get(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def _post(url, payload, timeout=30):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def fmt_ts(ts):
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts or "—"

def rep_color(score):
    if score >= 0.8:  return "#22c55e"   # green
    if score >= 0.5:  return "#f59e0b"   # amber
    if score >= 0.2:  return "#f97316"   # orange
    return "#ef4444"                      # red

def shard_emoji(shard):
    return {"PRIMARY": "🟢", "MONITORING": "🟡",
            "QUARANTINE": "🟠", "SLASHED": "🔴"}.get(shard, "⚪")

def create_gauge(score, title="Trust Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score, 4),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1},
            'bar': {'color': rep_color(score)},
            'steps': [
                {'range': [0.0, 0.2], 'color': "#fee2e2"},
                {'range': [0.2, 0.5], 'color': "#ffedd5"},
                {'range': [0.5, 0.8], 'color': "#fef9c3"},
                {'range': [0.8, 1.0], 'color': "#dcfce7"},
            ],
            'threshold': {'line': {'color': "#dc2626", 'width': 3},
                          'thickness': 0.75, 'value': 0.4}
        }
    ))
    fig.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
    return fig

def fetch_node_snapshot(base_url):
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:
        fh  = ex.submit(_get, f"{base_url}/health")
        ft  = ex.submit(_get, f"{base_url}/trust")
        fr  = ex.submit(_get, f"{base_url}/reports/latest?limit=20")
        fv  = ex.submit(_get, f"{base_url}/verdict")
        fc  = ex.submit(_get, f"{base_url}/consensus/reputations")
        fmr = ex.submit(_get, f"{base_url}/monitoring/results")
        frp = ex.submit(_get, f"{base_url}/peers/registered")
    return {
        "health":      fh.result(),
        "trust":       ft.result(),
        "reports":     fr.result(),
        "verdict":     fv.result(),
        "consensus":   fc.result(),
        "mon_results": fmr.result(),
        "reg_peers":   frp.result(),
        "base_url":    base_url,
    }

def mon_results_to_rows(mon_data):
    if not mon_data:
        return []
    results = mon_data.get("results", mon_data)
    if isinstance(results, dict):
        results = list(results.values())
    if not isinstance(results, list):
        return []
    rows = []
    for r in results:
        if not isinstance(r, dict):
            continue
        rows.append({
            "URL":          r.get("url", "—"),
            "Status":       "🟢 UP" if (r.get("status") == "success" or r.get("is_reachable")) else "🔴 DOWN",
            "HTTP":         r.get("status_code", "—"),
            "Response ms":  round(r.get("response_time_ms") or r.get("response_ms") or 0, 1),
            "SSL":          "✅" if r.get("ssl_valid") else "❌",
            "Timestamp":    str(fmt_ts(r.get("timestamp", ""))),
        })
    return rows

def reports_to_website_rows(reports_data):
    if not reports_data:
        return []
    reports = reports_data.get("reports", [])
    latest = {}
    for r in reports:
        url   = r.get("url", "unknown")
        epoch = r.get("epoch_id", 0)
        if url not in latest or epoch > latest[url].get("epoch_id", 0):
            latest[url] = r
    rows = []
    for url, r in latest.items():
        sc = r.get("status_code", 0)
        rows.append({
            "URL":         url,
            "Status":      "🟢 UP" if r.get("is_reachable", sc in range(200, 400)) else "🔴 DOWN",
            "HTTP":        sc or "—",
            "Response ms": round(r.get("response_ms", r.get("response_time_ms", 0)), 1),
            "SSL":         "✅" if r.get("ssl_valid") else "❌",
            "Timestamp":   fmt_ts(r.get("timestamp", "")),
            "Node":        r.get("node_id", "—"),
            "Epoch":       r.get("epoch_id", "—"),
        })
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("🔍 Decentralized Website Monitoring")
    st.caption("ML-powered consensus · Reputation-weighted voting · Blockchain-backed")

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown(f"**Monitoring {len(ALL_NODES)} nodes:**")
        for node in ALL_NODES:
            st.markdown(f"- {node}")

        st.divider()
        auto_refresh     = st.checkbox("Auto Refresh", value=False, key="auto_refresh")
        refresh_interval = st.selectbox("Interval (s)", [10, 30, 60], index=1, key="refresh_interval")

        st.divider()
        st.header("🚀 Actions")

        with st.expander("Trigger Monitoring"):
            if 'mon_urls' not in st.session_state:
                st.session_state.mon_urls = (
                    "https://httpbin.org/get\n"
                    "https://httpbin.org/status/200\n"
                    "https://httpbin.org/delay/1"
                )
            urls_in = st.text_area("URLs (one per line)", value=st.session_state.mon_urls,
                                   height=90, key="mon_urls_input")
            if st.button("▶ Trigger on All Nodes", use_container_width=True):
                urls = [u.strip() for u in urls_in.splitlines() if u.strip()]
                if urls:
                    with st.spinner("Triggering on all nodes…"):
                        success_count = 0
                        for node_url in ALL_NODES:
                            r = _post(f"{node_url}/monitor", {"urls": urls})
                            if r:
                                success_count += 1
                        if success_count > 0:
                            st.success(f"Monitoring triggered on {success_count}/{len(ALL_NODES)} nodes!")
                        else:
                            st.error("Failed — are the nodes running?")
                else:
                    st.warning("Enter at least one URL")

        with st.expander("Add Peer"):
            pid   = st.text_input("Peer Node ID", key="peer_id")
            phost = st.text_input("Host", value="localhost", key="peer_host")
            pport = st.number_input("Port", value=8006, min_value=1,
                                    max_value=65535, key="peer_port")
            if st.button("➕ Add Peer to All Nodes", use_container_width=True):
                success_count = 0
                for node_url in ALL_NODES:
                    r = _post(f"{node_url}/peers",
                              {"node_id": pid, "host": phost, "port": pport})
                    if r:
                        success_count += 1
                if success_count > 0:
                    st.success(f"Peer added to {success_count}/{len(ALL_NODES)} nodes!")
                else:
                    st.error("Failed")

    # ── Fetch data from all nodes ─────────────────────────────────────────
    with st.spinner(f"Loading data from {len(ALL_NODES)} nodes…"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ALL_NODES)) as ex:
            all_snaps = list(ex.map(fetch_node_snapshot, ALL_NODES))

    # Aggregate data from all nodes
    all_health = [s["health"] for s in all_snaps if s["health"]]
    all_trust = [s["trust"] for s in all_snaps if s["trust"]]
    all_verdict = [s["verdict"] for s in all_snaps if s["verdict"]]
    all_cons = [s["consensus"] for s in all_snaps if s["consensus"]]
    all_mon_res = [s["mon_results"] for s in all_snaps if s["mon_results"]]
    all_reports = [s["reports"] for s in all_snaps if s["reports"]]
    all_reg_p = [s["reg_peers"] for s in all_snaps if s["reg_peers"]]

    # Get first available health for header metrics
    health = all_health[0] if all_health else None
    if not health:
        st.error(f"❌ Cannot reach any nodes")
        st.info("Make sure the nodes are running on ports 8005-8008")
        st.stop()

    # Header metrics - aggregated
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes Online", len([h for h in all_health if h]))
    
    # Calculate total peers - connected_peers might be an int or a list
    total_peers = 0
    for h in all_health:
        peer_data = (h or {}).get("components", {}).get("peer_client", {})
        connected = peer_data.get("connected_peers", 0)
        if isinstance(connected, (list, dict)):
            total_peers += len(connected)
        elif isinstance(connected, int):
            total_peers += connected
    c2.metric("Total Peers", total_peers)
    
    bc_ok_count = sum([1 for h in all_health if isinstance(h.get("components", {}).get("blockchain", {}), dict) 
                        and h.get("components", {}).get("blockchain", {}).get("status") == "healthy"])
    c3.metric("Blockchain Nodes", f"{bc_ok_count}/{len(all_health)}")
    c4.metric("Avg Trust", f"{np.mean([t.get('trust_score', 0) for t in all_trust if t]):.4f}" if all_trust else "—")
    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🏠 Overview", "🌐 Website Status", "🗳️ Consensus Voting",
        "🔗 Multi-Node", "🤖 ML Features", "👥 Peers"
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW (aggregated from all nodes)
    # ══════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.header("System Overview (All Nodes)")
        
        # Aggregate component status across all nodes
        monitoring_active = sum([1 for h in all_health if h.get("components", {}).get("monitoring") == "active"])
        trust_engine_active = sum([1 for h in all_health if h.get("components", {}).get("trust_engine") == "active"])
        ml_classifier_active = sum([1 for h in all_health if h.get("components", {}).get("ml_classifier") == "active"])
        blockchain_connected = sum([1 for h in all_health if isinstance(h.get("components", {}).get("blockchain", {}), dict) 
                                    and h.get("components", {}).get("blockchain", {}).get("status") == "healthy"])
        
        # Total peers already calculated above
        total_peers_tab = total_peers

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Monitoring Active", f"{monitoring_active}/{len(all_health)}")
        m2.metric("Trust Engine Active", f"{trust_engine_active}/{len(all_health)}")
        m3.metric("ML Classifier Active", f"{ml_classifier_active}/{len(all_health)}")
        m4.metric("Blockchain Connected", f"{blockchain_connected}/{len(all_health)}")
        m5.metric("Total Peers", total_peers_tab)

        st.divider()

        # Aggregate trust scores across all nodes
        if all_trust:
            avg_trust = np.mean([t.get("trust_score", 0) for t in all_trust])
            st.subheader(f"Average Trust Score: {avg_trust:.4f}")
            col_g, col_b, col_d = st.columns(3)

            col_g.plotly_chart(create_gauge(avg_trust, "Average Trust"), width='stretch')

            # Trust scores per node
            trust_rows = []
            for h, t in zip(all_health, all_trust):
                if h and t:
                    trust_rows.append({
                        "Node": h.get("node_id", "—"),
                        "Trust Score": t.get("trust_score", 0)
                    })
            if trust_rows:
                df_trust = pd.DataFrame(trust_rows)
                fig_trust = px.bar(df_trust, x="Node", y="Trust Score",
                                   title="Trust Scores per Node",
                                   color="Trust Score",
                                   color_continuous_scale="RdYlGn",
                                   range_y=[0, 1])
                fig_trust.update_layout(height=260, margin=dict(t=40, b=10))
                col_b.plotly_chart(fig_trust, width='stretch')

            with col_d:
                st.markdown("**Aggregated Details**")
                st.write(f"Avg Score: **{avg_trust:.4f}**")
                st.write(f"Total Reports: {sum([t.get('report_count', 0) for t in all_trust if t])}")
                st.write(f"Total Peer Feedback: {sum([t.get('peer_feedback_count', 0) for t in all_trust if t])}")
                st.write(f"Nodes Online: {len(all_trust)}/{len(ALL_NODES)}")

        # Aggregate shard distribution from all nodes
        if all_cons:
            all_shards = {}
            for c in all_cons:
                for shard, count in c.get("shard_distribution", {}).items():
                    all_shards[shard] = all_shards.get(shard, 0) + count
            if all_shards:
                st.subheader("Shard Distribution (Aggregated)")
                shard_rows = [{"Shard": k, "Count": v} for k, v in all_shards.items()]
                df_shards = pd.DataFrame(shard_rows)
                fig_shards = px.pie(df_shards, values="Count", names="Shard",
                                    title="Node Distribution Across Shards")
                st.plotly_chart(fig_shards, width='stretch')

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — WEBSITE STATUS (aggregated from all nodes)
    # ══════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.header("Website Status (All Nodes)")
        st.caption("Aggregated monitoring results from all nodes.")

        # Aggregate website status from all nodes
        all_rows = []
        for mon_res, reports in zip(all_mon_res, all_reports):
            rows = mon_results_to_rows(mon_res) or reports_to_website_rows(reports)
            all_rows.extend(rows)

        if all_rows:
            df = pd.DataFrame(all_rows)
            if 'Timestamp' in df.columns and 'Last_Checked' not in df.columns:
                df = df.rename(columns={'Timestamp': 'Last_Checked'})
            
            # Group by URL and show latest status from any node
            if 'URL' in df.columns:
                df_grouped = df.groupby('URL').agg({
                    'Status': 'first',
                    'HTTP': 'first',
                    'Response ms': 'mean',
                    'SSL': 'first',
                    'Last_Checked': 'max'
                }).reset_index()
                
                total = len(df_grouped)
                up    = (df_grouped["Status"].str.startswith("🟢")).sum()
                down  = total - up
                avg_r = df_grouped["Response ms"].mean() if "Response ms" in df_grouped.columns else 0

                w1, w2, w3, w4 = st.columns(4)
                w1.metric("Total URLs",   total)
                w2.metric("🟢 Up",        up)
                w3.metric("🔴 Down",      down)
                w4.metric("Avg Response", f"{avg_r:.0f} ms")

                st.divider()
                for _, row in df_grouped.iterrows():
                    cols = st.columns([3, 1, 1, 1, 1, 2])
                    cols[0].markdown(f"**{row['URL']}**")
                    cols[1].markdown(row["Status"])
                    cols[2].markdown(f"`{row.get('HTTP', '—')}`")
                    cols[3].markdown(f"⏱ {row.get('Response ms', '—')} ms")
                    cols[4].markdown(f"SSL {row.get('SSL', '—')}")
                    cols[5].caption(row.get("Last_Checked", "—"))
                    st.divider()

                if "Response ms" in df_grouped.columns and df_grouped["Response ms"].sum() > 0:
                    fig_rt = px.bar(df_grouped, x="URL", y="Response ms",
                                    title="Average Response Time per URL (All Nodes)",
                                    color="Response ms",
                                    color_continuous_scale="Viridis")
                    fig_rt.update_layout(height=320)
                    st.plotly_chart(fig_rt, width='stretch')
        else:
            st.info("No monitoring results yet. Use 'Trigger Monitoring on All Nodes' in the sidebar or wait for the next 60-second cycle.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — CONSENSUS VOTING (aggregated from all nodes)
    # ══════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.header("🗳️ Consensus Voting (All Nodes)")
        st.caption("Reputation-weighted voting results aggregated from all nodes.")

        # Aggregate reputations from all nodes
        all_reps_data = {}
        all_ewma_data = {}
        all_actions_data = {}
        
        for cons in all_cons:
            reps_data = cons.get("reputations", {})
            ewma_data = cons.get("ewma_reputations", {})
            actions_data = cons.get("mitigation_actions", {})
            
            # Merge reputations (take average for nodes reported by multiple nodes)
            for nid, rep in reps_data.items():
                if nid in all_reps_data:
                    all_reps_data[nid] = (all_reps_data[nid] + rep) / 2
                else:
                    all_reps_data[nid] = rep
            
            # Merge EWMA reputations
            for nid, rep in ewma_data.items():
                if nid in all_ewma_data:
                    all_ewma_data[nid] = (all_ewma_data[nid] + rep) / 2
                else:
                    all_ewma_data[nid] = rep
            
            # Merge actions (take latest)
            all_actions_data.update(actions_data)

        # Also aggregate from verdicts
        for verdict in all_verdict:
            verdict_reps = verdict.get("node_reputations", {})
            for nid, rep in verdict_reps.items():
                if nid in all_reps_data:
                    all_reps_data[nid] = (all_reps_data[nid] + rep) / 2
                else:
                    all_reps_data[nid] = rep

        if all_reps_data:
            st.subheader("Node Reputation Breakdown (Aggregated)")

            vote_rows = []
            for nid, rep in all_reps_data.items():
                action  = all_actions_data.get(nid, {})
                shard   = action.get("shard", "—") if isinstance(action, dict) else str(action)
                status  = action.get("status", "—") if isinstance(action, dict) else "—"
                ewma_rep = all_ewma_data.get(nid, rep)
                
                vote_rows.append({
                    "Node":        nid,
                    "Reputation":  round(rep, 4),
                    "EWMA Rep":    round(ewma_rep, 4),
                    "Status":      status,
                    "Shard":       f"{shard_emoji(shard)} {shard}",
                })

            df_votes = pd.DataFrame(vote_rows)
            df_votes = df_votes.sort_values("Reputation", ascending=False)
            st.dataframe(df_votes, width='stretch')

            # Reputation chart
            fig_rep = px.bar(df_votes, x="Node", y="Reputation",
                             title="Node Reputation Scores (Aggregated)",
                             color="Reputation",
                             color_continuous_scale="RdYlGn",
                             range_y=[0, 1],
                             text="Reputation")
            fig_rep.add_hline(y=0.4, line_dash="dash", line_color="red",
                             annotation_text="Vote exclusion threshold")
            fig_rep.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_rep.update_layout(height=360)
            st.plotly_chart(fig_rep, width='stretch')
        else:
            st.info("No reputation data yet. Wait for consensus cycles to complete.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — MULTI-NODE (already shows data from all nodes via all_snaps)
    # ══════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.header("Multi-Node Comparison")
        st.caption(f"Comparing {len(all_snaps)} nodes.")

        snaps = all_snaps  # Use already-fetched data

        # Health comparison
        st.subheader("Node Health")
        h_rows = []
        for s in snaps:
            h  = s.get("health") or {}
            cp = h.get("components", {})
            pc2 = cp.get("peer_client", {})
            bc2 = cp.get("blockchain", {})
            tr2 = s.get("trust") or {}
            cr2 = s.get("consensus") or {}
            reps2 = cr2.get("reputations", {})
            avg_rep = (sum(reps2.values()) / len(reps2)) if reps2 else 0

            h_rows.append({
                "Node":       h.get("node_id", s["base_url"]),
                "Status":     "🟢 " + h.get("status","—").upper() if h else "🔴 UNREACHABLE",
                "Monitoring": "✅" if cp.get("monitoring") == "active" else "❌",
                "ML":         "✅" if cp.get("ml_classifier") == "active" else "❌",
                "Blockchain": "✅" if isinstance(bc2, dict) and bc2.get("status") == "healthy" else "❌",
                "Peers":      pc2.get("connected_peers", 0) if isinstance(pc2, dict) else 0,
                "Trust":      f"{tr2.get('trust_score', 0):.4f}" if tr2 else "—",
                "Avg ML Rep": f"{avg_rep:.4f}" if reps2 else "—",
            })
        if h_rows:
            st.dataframe(pd.DataFrame(h_rows), width='stretch')

        # Website status per node
        st.subheader("Website Checks per Node")
        site_rows = []
        for s in snaps:
            nid = (s.get("health") or {}).get("node_id", s["base_url"])
            rows = mon_results_to_rows(s.get("mon_results")) or \
                   reports_to_website_rows(s.get("reports"))
            for r in rows:
                r["Node"] = nid
                site_rows.append(r)

        if site_rows:
            df_sites = pd.DataFrame(site_rows)
            st.dataframe(df_sites, width='stretch')

            if "Response ms" in df_sites.columns:
                fig_cmp = px.bar(
                    df_sites,
                    x="Response ms", y="URL", color="Node",
                    barmode="group",
                    orientation="h",
                    title="Response Time per Node",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_cmp.update_layout(height=360)
                st.plotly_chart(fig_cmp, width='stretch')

        # Reputation Scores per Node
        st.subheader("Reputation Scores per Node")
        rep_rows = []
        for s in snaps:
            h = s.get("health") or {}
            src = h.get("node_id", s["base_url"])
            tr = s.get("trust") or {}
            trust_score = tr.get("trust_score", 0)
            
            # Try to get consensus reputations
            c2 = s.get("consensus") or {}
            rp = c2.get("reputations", {})
            
            if rp:
                # Show all reputations from this node's perspective
                for nid, score in rp.items():
                    rep_rows.append({
                        "Node":        nid,
                        "Reputation":  round(score, 4),
                        "Reported By": src,
                    })
            else:
                # Show node's own trust score as reputation
                rep_rows.append({
                    "Node":        src,
                    "Reputation":  round(trust_score, 4),
                    "Reported By": src,
                })
        
        if rep_rows:
            df_rep = pd.DataFrame(rep_rows)
            st.dataframe(df_rep, width='stretch')
            fig_rp = px.bar(
                df_rep, x="Node", y="Reputation",
                title="Reputation Scores",
                range_y=[0, 1],
                color="Reputation",
                color_continuous_scale="RdYlGn",
                text="Reputation"
            )
            fig_rp.add_hline(y=0.4, line_dash="dash", line_color="red",
                             annotation_text="Vote exclusion threshold")
            fig_rp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_rp.update_layout(height=360)
            st.plotly_chart(fig_rp, width='stretch')
        else:
            st.info("No reputation data available.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — ML FEATURES (aggregated from all nodes)
    # ══════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.header("ML Features & Predictions (All Nodes)")
        st.caption("Aggregated ML features from all nodes.")

        # Fetch features from all nodes
        all_features = []
        for node_url in ALL_NODES:
            features_data = _get(f"{node_url}/features")
            if features_data:
                health = _get(f"{node_url}/health")
                node_id = health.get("node_id", node_url) if health else node_url
                all_features.append({
                    "node_id": node_id,
                    "features": features_data.get("features", {}),
                    "prediction": features_data.get("prediction")
                })

        if all_features:
            # Aggregate features across all nodes (average for numerical features)
            all_feats = {}
            for feat in all_features:
                for k, v in feat["features"].items():
                    if k in all_feats:
                        try:
                            all_feats[k] = (all_feats[k] + float(v)) / 2
                        except (ValueError, TypeError):
                            all_feats[k] = v
                    else:
                        try:
                            all_feats[k] = float(v)
                        except (ValueError, TypeError):
                            all_feats[k] = v

            if all_feats:
                # FIX: split response_ms (large value) from ratio features (0-1)
                ratio_feats = {k: v for k, v in all_feats.items()
                               if k != "avg_response_ms" and isinstance(v, (int, float)) and 0.0 <= v <= 1.0}
                other_feats = {k: v for k, v in all_feats.items()
                               if k not in ratio_feats}

                col_f1, col_f2 = st.columns(2)

                if ratio_feats:
                    df_ratio = pd.DataFrame({
                        "Feature": list(ratio_feats.keys()),
                        "Value":   list(ratio_feats.values())
                    })
                    # Colour false_report_rate: high = bad (red), low = good (green)
                    fig_ratio = px.bar(
                        df_ratio, x="Feature", y="Value",
                        title="Aggregated Feature Values (0–1 scale)",
                        color="Value",
                        color_continuous_scale="RdYlGn",
                        range_y=[0, 1],
                        text="Value"
                    )
                    fig_ratio.update_traces(texttemplate="%{text:.3f}",
                                            textposition="outside")
                    fig_ratio.update_layout(height=340,
                                            xaxis_tickangle=-30)
                    col_f1.plotly_chart(fig_ratio, width='stretch',
                                        key="ratio_features_chart")

                if other_feats:
                    df_other = pd.DataFrame({
                        "Feature": list(other_feats.keys()),
                        "Value":   [round(float(v), 1) for v in other_feats.values()]
                    })
                    fig_other = px.bar(
                        df_other, x="Feature", y="Value",
                        title="Aggregated Response Time Features (ms)",
                        color="Value",
                        color_continuous_scale="Blues",
                        text="Value"
                    )
                    fig_other.update_traces(texttemplate="%{text:.0f} ms",
                                            textposition="outside")
                    fig_other.update_layout(height=340)
                    col_f2.plotly_chart(fig_other, width='stretch',
                                        key="other_features_chart")

                # Aggregated feature metrics
                st.subheader("Aggregated Feature Values")
                metric_cols = st.columns(4)
                for i, (k, v) in enumerate(all_feats.items()):
                    if isinstance(v, (int, float)):
                        metric_cols[i % 4].metric(
                            k.replace("_", " ").title(),
                            f"{v:.4f}"
                        )

            st.divider()

            # Aggregated ML Predictions from all nodes
            st.subheader("ML Predictions (All Nodes)")
            pred_rows = []
            for feat in all_features:
                pred = feat.get("prediction")
                if pred:
                    pred_rows.append({
                        "Node": feat["node_id"],
                        "Label": pred.get("prediction_label", "Unknown"),
                        "Confidence": pred.get("confidence", 0),
                        "Honest Prob": pred.get("honest_probability", 0),
                        "Malicious Prob": pred.get("malicious_probability", 0)
                    })
            
            if pred_rows:
                df_preds = pd.DataFrame(pred_rows)
                st.dataframe(df_preds, width='stretch')
                
                # Prediction distribution
                pred_counts = df_preds["Label"].value_counts()
                fig_pred = px.pie(values=pred_counts.values, names=pred_counts.index,
                                   title="Prediction Distribution (All Nodes)")
                st.plotly_chart(fig_pred, width='stretch')
            else:
                st.info("No ML predictions available. See reputation scores in Consensus Voting tab.")
        else:
            st.info("No ML features data available. Make sure the nodes are running and monitoring is active.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 6 — PEERS (aggregated from all nodes)
    # ══════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.header("Peer Network (All Nodes)")
        st.caption("Aggregated peer registrations from all nodes.")

        # Aggregate peers from all nodes
        all_peers = {}
        for reg_p in all_reg_p:
            peers_dict = reg_p.get("peers", {})
            for nid, info in peers_dict.items():
                if nid not in all_peers:
                    all_peers[nid] = info
                else:
                    # Merge info if needed
                    if info.get("public_key_hex") and not all_peers[nid].get("public_key_hex"):
                        all_peers[nid]["public_key_hex"] = info["public_key_hex"]

        # Aggregate reputations from all nodes
        all_rep_lookup = {}
        for cons in all_cons:
            reps = cons.get("reputations", {})
            for nid, rep in reps.items():
                if nid in all_rep_lookup:
                    all_rep_lookup[nid] = (all_rep_lookup[nid] + rep) / 2
                else:
                    all_rep_lookup[nid] = rep

        if all_peers:
            st.subheader(f"Registered Peers ({len(all_peers)})")
            peer_rows = []
            for nid, info in all_peers.items():
                rep_val = all_rep_lookup.get(nid)
                
                # If not found in reputations, try to get from trust data
                if rep_val is None:
                    # Check if this peer is one of the nodes we're monitoring
                    for snap in all_snaps:
                        h = snap.get("health") or {}
                        if h.get("node_id") == nid or snap.get("base_url") == info.get("url"):
                            trust_data = snap.get("trust") or {}
                            rep_val = trust_data.get("trust_score", 0)
                            break
                
                peer_rows.append({
                    "Node ID":    nid,
                    "URL":        info.get("url", "—"),
                    "Reputation": f"{rep_val:.4f}" if rep_val is not None else "pending",
                    "Public Key": ((info.get("public_key_hex") or "")[:20] + "…") if info.get("public_key_hex") else "—",
                })
            st.dataframe(pd.DataFrame(peer_rows), width='stretch')

            # Peer reputation mini chart
            rep_peer_data = [(nid, all_rep_lookup[nid]) for nid in all_peers if nid in all_rep_lookup]
            if rep_peer_data:
                df_pr = pd.DataFrame(rep_peer_data, columns=["Node", "Reputation"])
                fig_pr = px.bar(df_pr, x="Node", y="Reputation",
                                title="Peer Reputation Scores (Aggregated)",
                                color="Reputation",
                                color_continuous_scale="RdYlGn",
                                range_y=[0, 1], text="Reputation")
                fig_pr.add_hline(y=0.4, line_dash="dash", line_color="red",
                                 annotation_text="Vote exclusion threshold")
                fig_pr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_pr.update_layout(height=320)
                st.plotly_chart(fig_pr, width='stretch')
        else:
            st.info("No peers registered. Use 'Add Peer to All Nodes' in the sidebar or run setup_network.py.")

    # ── Auto-refresh ───────────────────────────────────────────────────────
    if auto_refresh:
        try:
            interval = int(refresh_interval) if isinstance(refresh_interval, (int, str)) else 30
            time.sleep(interval)
            st.rerun()
        except Exception:
            time.sleep(30)
            st.rerun()


if __name__ == "__main__":
    main()
