"""
Streamlit Dashboard for Decentralized Website Monitoring System
Provides real-time visualization of node status, trust scores, and system metrics
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime, timedelta
import sys
import os
import concurrent.futures

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

st.set_page_config(
    page_title="Web Monitoring Dashboard",
    page_icon=":globe:",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8005")


# ── API helpers ───────────────────────────────────────────────────────────────────

def _get(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def _post(url, payload, timeout=10):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


class DashboardAPI:
    @staticmethod
    def get_health(base=None):
        return _get(f"{base or API_BASE_URL}/health")

    @staticmethod
    def get_trust_info(base=None):
        return _get(f"{base or API_BASE_URL}/trust")

    @staticmethod
    def get_features(base=None):
        return _get(f"{base or API_BASE_URL}/features")

    @staticmethod
    def get_peers(base=None):
        return _get(f"{base or API_BASE_URL}/peers")

    @staticmethod
    def get_statistics(base=None):
        return _get(f"{base or API_BASE_URL}/statistics")

    @staticmethod
    def get_reports_latest(base=None, limit=20):
        return _get(f"{base or API_BASE_URL}/reports/latest?limit={limit}")

    @staticmethod
    def get_monitoring_results(base=None):
        return _get(f"{base or API_BASE_URL}/monitoring/results")

    @staticmethod
    def get_registered_peers(base=None):
        return _get(f"{base or API_BASE_URL}/peers/registered")

    @staticmethod
    def get_verdict(base=None):
        return _get(f"{base or API_BASE_URL}/verdict")

    @staticmethod
    def get_consensus_reputations(base=None):
        return _get(f"{base or API_BASE_URL}/consensus/reputations")

    @staticmethod
    def trigger_monitoring(urls, base=None):
        return _post(f"{base or API_BASE_URL}/monitor", {"urls": urls}, timeout=30)

    @staticmethod
    def add_peer(node_id, host, port, base=None):
        return _post(f"{base or API_BASE_URL}/peers",
                     {"node_id": node_id, "host": host, "port": port})


# ── Utilities ─────────────────────────────────────────────────────────────────────

def fmt_ts(ts):
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts or "—"

def trust_color(score):
    if score >= 0.8:   return "green"
    if score >= 0.6:   return "lightgreen"
    if score >= 0.4:   return "orange"
    if score >= 0.2:   return "red"
    return "darkred"

def create_trust_gauge(score, title="Trust Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': trust_color(score)},
            'steps': [
                {'range': [0.0, 0.4], 'color': "#ffcccc"},
                {'range': [0.4, 0.7], 'color': "#fff3cc"},
                {'range': [0.7, 1.0], 'color': "#ccffcc"},
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.8}
        }
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    return fig


def fetch_node_snapshot(base_url):
    """Fetch all relevant data from one node in parallel requests."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        fh  = ex.submit(_get, f"{base_url}/health")
        ft  = ex.submit(_get, f"{base_url}/trust")
        fr  = ex.submit(_get, f"{base_url}/reports/latest?limit=20")
        fv  = ex.submit(_get, f"{base_url}/verdict")
        fc  = ex.submit(_get, f"{base_url}/consensus/reputations")
        fmr = ex.submit(_get, f"{base_url}/monitoring/results")
        frp = ex.submit(_get, f"{base_url}/peers/registered")
    return {
        "health":    fh.result(),
        "trust":     ft.result(),
        "reports":   fr.result(),
        "verdict":   fv.result(),
        "consensus": fc.result(),
        "mon_results": fmr.result(),
        "reg_peers": frp.result(),
        "base_url":  base_url,
    }


# ── Website Status helpers ────────────────────────────────────────────────────────

def reports_to_website_rows(reports_data):
    """
    Convert /reports/latest payload into per-URL summary rows.
    Returns a list of dicts with keys: url, status, status_code,
    response_ms, ssl_valid, last_seen, node_id, epoch_id.
    """
    if not reports_data:
        return []
    reports = reports_data.get("reports", [])
    if not reports:
        return []

    # Group by URL, keep the most recent entry per URL
    latest = {}
    for r in reports:
        url = r.get("url", "unknown")
        epoch = r.get("epoch_id", 0)
        if url not in latest or epoch > latest[url].get("epoch_id", 0):
            latest[url] = r

    rows = []
    for url, r in latest.items():
        status_code = r.get("status_code", 0)
        is_reachable = r.get("is_reachable", status_code in range(200, 400))
        rows.append({
            "URL": url,
            "Status": "🟢 UP" if is_reachable else "🔴 DOWN",
            "Status Code": status_code if status_code else "—",
            "Response (ms)": round(r.get("response_ms", r.get("response_time_ms", 0)), 1),
            "SSL Valid": "✅" if r.get("ssl_valid") else "❌",
            "Last_Checked": fmt_ts(r.get("timestamp", "")),
            "Reported By": r.get("node_id", r.get("node_address", "—")),
            "Epoch": r.get("epoch_id", "—"),
        })
    return rows


def mon_results_to_rows(mon_data):
    """
    Convert /monitoring/results payload (dict or list) into rows.
    """
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
            "URL": r.get("url", "—"),
            "Status": "🟢 UP" if r.get("status") == "success" or r.get("is_reachable") else "🔴 DOWN",
            "Status Code": r.get("status_code", "—"),
            "Response (ms)": round(r.get("response_time_ms") or r.get("response_ms") or 0, 1),
            "SSL Valid": "✅" if r.get("ssl_valid") else "❌",
            "Content Hash": (r.get("content_hash", "") or "")[:12] + "…" if r.get("content_hash") else "—",
            "Timestamp": fmt_ts(r.get("timestamp", "")),
        })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    st.title("Decentralized Website Monitoring Dashboard")
    st.markdown("---")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.header("Configuration")
    global API_BASE_URL
    API_BASE_URL = st.sidebar.text_input("API Base URL", value=API_BASE_URL)

    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
    refresh_interval = st.sidebar.selectbox("Refresh Interval", [10, 30, 60], index=2)

    # Extra node URLs for multi-node view
    st.sidebar.markdown("**Extra Node URLs** (one per line)")
    extra_nodes_raw = st.sidebar.text_area(
        "Additional nodes",
        value="http://localhost:8006\nhttp://localhost:8007\nhttp://localhost:8008",
        height=120,
        label_visibility="collapsed"
    )
    extra_node_urls = [u.strip() for u in extra_nodes_raw.splitlines() if u.strip()]

    st.sidebar.header("Configuration")
    
    # Initialize session state for URLs if not exists
    if 'monitored_urls' not in st.session_state:
        st.session_state.monitored_urls = ""
    
    # Default URLs for manual monitoring
    default_urls = st.sidebar.text_area(
        "Default Monitored URLs",
        value=st.session_state.monitored_urls,
        height=100,
        help="These URLs will be used as defaults when triggering manual monitoring",
        key="default_urls_input",
        on_change=lambda: st.session_state.update(monitored_urls=st.session_state.default_urls_input)
    )
    
    # Reset URLs button
    if st.sidebar.button("Clear URLs"):
        st.session_state.monitored_urls = ""
        st.rerun()
    
    st.sidebar.header("Actions")
    with st.sidebar.expander("Manual Monitoring"):
        # Use the current value from session state
        urls_input = st.text_area("URLs (one per line)", value=st.session_state.monitored_urls, height=80, key="manual_urls_input",
                               on_change=lambda: st.session_state.update(monitored_urls=st.session_state.manual_urls_input))
        if st.button("Trigger Monitoring"):
            urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
            if urls:
                with st.spinner("Triggering…"):
                    r = DashboardAPI.trigger_monitoring(urls)
                if r:
                    st.sidebar.success("Monitoring triggered!")
                else:
                    st.sidebar.error("Failed — is the node running?")
            else:
                st.sidebar.warning("Enter at least one URL")

    with st.sidebar.expander("Add Peer"):
        peer_node_id = st.text_input("Peer Node ID")
        peer_host    = st.text_input("Peer Host", value="localhost")
        peer_port    = st.number_input("Peer Port", value=8006, min_value=1, max_value=65535)
        if st.button("Add Peer"):
            if peer_node_id:
                r = DashboardAPI.add_peer(peer_node_id, peer_host, peer_port)
                if r:
                    st.sidebar.success("Peer added!")
                else:
                    st.sidebar.error("Failed to add peer")

    # ── Fetch primary node data ────────────────────────────────────────────────
    with st.spinner("Loading…"):
        snap = fetch_node_snapshot(API_BASE_URL)

    health_data  = snap["health"]
    trust_data   = snap["trust"]
    reports_data = snap["reports"]
    verdict_data = snap["verdict"]
    consensus    = snap["consensus"]
    mon_results  = snap["mon_results"]
    reg_peers    = snap["reg_peers"]

    if not health_data or health_data is None:
        st.error("Cannot reach node at " + API_BASE_URL)
        st.error("Please make sure the node is running and blockchain is available")
        st.info("Start blockchain: cd blockchain && npx hardhat node")
        st.info("Start node: cd node_service && python main.py --port 8005 --node-id node_a")
        st.stop()

    # Header row
    c1, c2, c3 = st.columns([1, 2, 1])
    c1.metric("Node ID", health_data.get("node_id", "—"))
    status = health_data.get("status", "unknown")
    c2.markdown(f"**Status:** :{'green' if status == 'healthy' else 'red'}[{status.upper()}]")
    c3.metric("Last Update", fmt_ts(health_data.get("timestamp", "")))
    st.markdown("---")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_overview, tab_websites, tab_multi, tab_trust, tab_ml, tab_peers, tab_stats = st.tabs([
        "Overview", "Website Status", "Multi-Node", "Trust Analysis", "ML Features", "Peers", "Statistics"
    ])

    # ── TAB 1 — Overview ───────────────────────────────────────────────────────
    with tab_overview:
        st.header("System Overview")
        comps = health_data.get("components", {})

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Monitoring",    "Active"      if comps.get("monitoring") == "active"   else "Inactive")
        c2.metric("Trust Engine",  "Active"      if comps.get("trust_engine") == "active" else "Inactive")
        pc = comps.get("peer_client", {})
        c3.metric("Connected Peers", pc.get("connected_peers", 0) if isinstance(pc, dict) else 0)
        c4.metric("ML Classifier", "Active"      if comps.get("ml_classifier") == "active" else "Inactive")
        bc = comps.get("blockchain", {})
        c5.metric("Blockchain", "Connected" if isinstance(bc, dict) and bc.get("status") == "healthy" else "Disconnected")

        if trust_data:
            st.subheader("Current Trust Score")
            c1, c2, c3 = st.columns(3)
            ts = trust_data.get("trust_score", 0)
            c1.plotly_chart(create_trust_gauge(ts, "Overall Trust"), width='stretch')

            comp_scores = trust_data.get("components", {})
            if comp_scores:
                df = pd.DataFrame({"Component": list(comp_scores.keys()), "Score": list(comp_scores.values())})
                fig = px.bar(df, x="Component", y="Score", title="Trust Components",
                             color="Score", color_continuous_scale="RdYlGn")
                fig.update_layout(yaxis=dict(range=[0, 1]))
                c2.plotly_chart(fig, width='stretch')

            with c3:
                st.subheader("Trust Details")
                st.write(f"**Trust Score:** {ts:.4f}")
                st.write(f"**Report Count:** {trust_data.get('report_count', 0)}")
                st.write(f"**Peer Feedback:** {trust_data.get('peer_feedback_count', 0)}")
                st.write(f"**Last Update:** {fmt_ts(trust_data.get('last_update', ''))}")

        # Verdict / consensus block
        if verdict_data:
            st.subheader("Latest Consensus Verdicts")
            verdicts = verdict_data.get("verdicts", {})
            reps     = verdict_data.get("node_reputations", {})
            if verdicts:
                rows = []
                for epoch_id, v in sorted(verdicts.items(), reverse=True)[:5]:
                    rows.append({
                        "Epoch": epoch_id,
                        "Status": v.get("status", "—"),
                        "Majority": v.get("majority_verdict", "—"),
                        "Honest": ", ".join(v.get("honest", [])) or "—",
                        "Slashed": ", ".join(v.get("slashed", [])) or "none",
                        "Timestamp": fmt_ts(v.get("timestamp", "")),
                    })
                st.dataframe(pd.DataFrame(rows), width='stretch')
            if reps:
                st.markdown("**Node Status & Reputations:**")
                
                # Enhanced ML engine - show 4-tier categories
                reputation_data = get_consensus_reputations()
                if reputation_data and reputation_data.get("engine_type") == "enhanced":
                    # Enhanced ML engine - show 4-tier categories
                    mitigation_actions = reputation_data.get("mitigation_actions", {})
                    shard_distribution = reputation_data.get("shard_distribution", {})
                    
                    # Display shard distribution
                    st.markdown("**Shard Distribution:**")
                    shard_cols = st.columns(len(shard_distribution))
                    for i, (shard, count) in enumerate(shard_distribution.items()):
                        color_map = {
                            "PRIMARY": "🟢",
                            "MONITORING": "🟡", 
                            "QUARANTINE": "🟠",
                            "SLASHED": "🔴"
                        }
                        shard_cols[i].metric(f"{color_map.get(shard, '')} {shard}", count)
                    
                    # Display individual node status
                    for nid, score in reps.items():
                        with st.expander(f"Node {nid} - Reputation: {score:.4f}"):
                            if nid in mitigation_actions:
                                action = mitigation_actions[nid]
                                st.write(f"**Status:** {action['status']}")
                                st.write(f"**Action:** {action['action']}")
                                st.write(f"**Shard:** {action['shard']}")
                                
                                # Status color coding
                                status_colors = {
                                    "HEALTHY": "🟢",
                                    "SUSPICIOUS": "🟡",
                                    "FAULTY": "🟠", 
                                    "MALICIOUS": "🔴"
                                }
                                st.markdown(f"{status_colors.get(action['status'], '')} **{action['status']}**")
                else:
                    # Original ML engine - simple reputation display
                    rep_cols = st.columns(len(reps))
                    for i, (nid, score) in enumerate(reps.items()):
                        rep_cols[i].metric(nid, f"{score:.4f}")

    # ── TAB 2 — Website Status ─────────────────────────────────────────────────
    with tab_websites:
        st.header("Website Status")
        st.caption("Live status of all monitored URLs reported by this node.")

        # Try /monitoring/results first (own node's raw checks), fall back to /reports/latest
        rows = mon_results_to_rows(mon_results)
        source = "monitoring/results"
        if not rows:
            rows = reports_to_website_rows(reports_data)
            source = "reports/latest"

        if rows:
            df = pd.DataFrame(rows)
            
            # Normalize timestamp column name (mon_results uses 'Timestamp', reports uses 'Last_Checked')
            if 'Timestamp' in df.columns and 'Last_Checked' not in df.columns:
                df = df.rename(columns={'Timestamp': 'Last_Checked'})

            # Deduplicate by URL, keeping latest entry for each URL
            df = df.sort_values('Last_Checked').drop_duplicates(subset=['URL'], keep='last')

            # Summary metrics
            total  = len(df)
            up     = (df["Status"].str.startswith("🟢")).sum()
            down   = total - up
            avg_rt = df["Response (ms)"].mean() if "Response (ms)" in df.columns else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total URLs",      total)
            c2.metric("🟢 Up",           up,   delta=None)
            c3.metric("🔴 Down",         down, delta=None)
            c4.metric("Avg Response",    f"{avg_rt:.0f} ms")

            st.markdown("---")

            # Per-URL cards
            for _, row in df.iterrows():
                with st.container():
                    cols = st.columns([3, 1, 1, 1, 1, 2])
                    cols[0].markdown(f"**{row['URL']}**")
                    cols[1].markdown(row["Status"])
                    cols[2].markdown(f"`{row.get('Status Code','—')}`")
                    cols[3].markdown(f"⏱ {row.get('Response (ms)', '—')} ms")
                    cols[4].markdown(f"SSL {row.get('SSL Valid','—')}")
                    cols[5].caption(row.get("Timestamp", row.get("Last_Checked", "—")))
                    st.divider()

            # Response time bar chart
            if "Response (ms)" in df.columns and df["Response (ms)"].sum() > 0:
                fig = px.bar(
                    df, x="URL", y="Response (ms)",
                    title="Response Times",
                    color="Response (ms)",
                    color_continuous_scale="RdYlGn_r",
                    text="Response (ms)"
                )
                fig.update_traces(texttemplate="%{text:.0f} ms", textposition="outside")
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, width='stretch', key="response_times_chart")

            st.caption(f"Source: `/{source}` — refreshed at {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.info("No monitoring results yet. Wait for the next 60-second cycle, or use Manual Monitoring in the sidebar.")

    # ── TAB 3 — Multi-Node ────────────────────────────────────────────────────
    with tab_multi:
        st.header("Multi-Node View")

        all_urls = [API_BASE_URL] + [u for u in extra_node_urls if u != API_BASE_URL]

        if len(all_urls) < 2:
            st.info("Add extra node URLs in the sidebar to see the multi-node comparison.")
        else:
            st.caption(f"Polling {len(all_urls)} nodes: {', '.join(all_urls)}")

        # Fetch all nodes in parallel
        with st.spinner(f"Polling {len(all_urls)} node(s)…"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_urls)) as ex:
                snaps = list(ex.map(fetch_node_snapshot, all_urls))

        # ── Node health summary table ──────────────────────────────────────────
        st.subheader("Node Health Comparison")
        health_rows = []
        for s in snaps:
            h = s.get("health") or {}
            comps = h.get("components", {})
            pc = comps.get("peer_client", {})
            bc = comps.get("blockchain", {})
            t  = s.get("trust") or {}
            health_rows.append({
                "Node ID":       h.get("node_id", s["base_url"]),
                "URL":           s["base_url"],
                "Status":        "🟢 " + h.get("status", "—").upper() if h else "🔴 UNREACHABLE",
                "Monitoring":    "✅" if comps.get("monitoring") == "active" else "❌",
                "ML Classifier": "✅" if comps.get("ml_classifier") == "active" else "❌",
                "Blockchain":    "✅" if isinstance(bc, dict) and bc.get("status") == "healthy" else "❌",
                "Peers":         pc.get("connected_peers", 0) if isinstance(pc, dict) else 0,
                "Trust Score":   f"{t.get('trust_score', 0):.4f}" if t else "—",
                "Last Update":   fmt_ts(h.get("timestamp", "")),
            })
        if health_rows:
            st.dataframe(pd.DataFrame(health_rows), width='stretch')

        # ── Per-node website status ────────────────────────────────────────────
        st.subheader("Website Status per Node")
        all_site_rows = []
        for s in snaps:
            h    = s.get("health") or {}
            nid  = h.get("node_id", s["base_url"])
            rows = mon_results_to_rows(s.get("mon_results"))
            if not rows:
                rows = reports_to_website_rows(s.get("reports"))
            for r in rows:
                r["Node"] = nid
                all_site_rows.append(r)

        if all_site_rows:
            df_sites = pd.DataFrame(all_site_rows)
            st.dataframe(df_sites, width='stretch')

            # Response time comparison across nodes
            if "Response (ms)" in df_sites.columns:
                url_col  = "URL"  if "URL"  in df_sites.columns else df_sites.columns[0]
                node_col = "Node" if "Node" in df_sites.columns else df_sites.columns[-1]
                fig = px.bar(
                    df_sites, x=url_col, y="Response (ms)", color=node_col,
                    barmode="group",
                    title="Response Time Comparison Across Nodes",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=380)
                st.plotly_chart(fig, width='stretch', key="website_status_chart")
        else:
            st.info("No website results available from any node yet.")

        # ── Reputation comparison ──────────────────────────────────────────────
        st.subheader("Consensus Reputations (all nodes)")
        rep_rows = []
        for s in snaps:
            c = s.get("consensus") or {}
            reps = c.get("reputations", {})
            acts = c.get("mitigation_actions", {})
            src_node = (s.get("health") or {}).get("node_id", s["base_url"])
            for nid, score in reps.items():
                rep_rows.append({
                    "Reported By": src_node,
                    "Node":        nid,
                    "Reputation":  round(score, 4),
                    "Action":      acts.get(nid, "—"),
                })
        if rep_rows:
            df_rep = pd.DataFrame(rep_rows)
            st.dataframe(df_rep, width='stretch')
            fig = px.bar(
                df_rep, x="Node", y="Reputation", color="Reported By",
                barmode="group", title="Reputation Scores by Node",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                range_y=[0, 1]
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No reputation data yet — consensus runs after the first epoch with 2+ peer reports.")

        # ── Shared reports across nodes ────────────────────────────────────────
        st.subheader("Recent Reports (across all nodes)")
        all_report_rows = []
        for s in snaps:
            r_data = s.get("reports") or {}
            nid    = (s.get("health") or {}).get("node_id", s["base_url"])
            for rep in r_data.get("reports", []):
                all_report_rows.append({
                    "Node":        nid,
                    "URL":         rep.get("url", "—"),
                    "Epoch":       rep.get("epoch_id", "—"),
                    "Response ms": round(rep.get("response_ms", rep.get("response_time_ms", 0)), 1),
                    "SSL":         "✅" if rep.get("ssl_valid") else "❌",
                    "Reachable":   "🟢" if rep.get("is_reachable") else "🔴",
                })
        if all_report_rows:
            df_reps = pd.DataFrame(all_report_rows).sort_values("Epoch", ascending=False)
            st.dataframe(df_reps.head(30), width='stretch')

    # ── TAB 4 — Trust Analysis ─────────────────────────────────────────────────
    with tab_trust:
        st.header("Trust Analysis")
        if trust_data:
            st.subheader("Trust Score Trend (simulated 24 h)")
            ts_val = trust_data.get("trust_score", 0.5)
            dates  = pd.date_range(end=datetime.now(), periods=24, freq='h')
            scores = [ts_val * (0.9 + 0.2 * (i % 3) / 3) for i in range(24)]
            fig = px.line(pd.DataFrame({"Timestamp": dates, "Trust Score": scores}),
                          x="Timestamp", y="Trust Score", markers=True,
                          title="Trust Score Over Time (24 h)")
            fig.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, width='stretch', key="reputation_chart")

            comps = trust_data.get("components", {})
            if comps:
                fig = px.bar(
                    pd.DataFrame({"Component": list(comps.keys()), "Score": list(comps.values())}),
                    x="Component", y="Score", title="Trust Components",
                    color="Score", color_continuous_scale="RdYlGn"
                )
                fig.update_layout(yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig, width='stretch', key="trust_components_chart")
        else:
            st.warning("No trust data available")

    # ── TAB 5 — ML Features ───────────────────────────────────────────────────
    with tab_ml:
        st.header("ML Features and Predictions")
        if snap.get("health"):
            features_data = DashboardAPI.get_features()
        else:
            features_data = None

        if features_data:
            feats  = features_data.get("features", {})
            pred   = features_data.get("prediction")
            c1, c2 = st.columns(2)
            with c1:
                df = pd.DataFrame({"Feature": list(feats.keys()), "Value": list(feats.values())})
                fig = px.bar(df, x="Feature", y="Value", title="ML Feature Values",
                             color="Value", color_continuous_scale="Viridis")
                st.plotly_chart(fig, width='stretch', key="ml_features_chart")
            with c2:
                for k, v in feats.items():
                    st.metric(k.replace("_", " ").title(), f"{v:.4f}")
            if pred:
                st.subheader("ML Prediction")
                c1, c2, c3 = st.columns(3)
                lbl  = pred.get("prediction_label", "Unknown")
                conf = pred.get("confidence", 0)
                c1.metric("Prediction",  lbl)
                c1.metric("Confidence",  f"{conf:.4f}")
                hp = pred.get("honest_probability", 0)
                mp = pred.get("malicious_probability", 0)
                c2.metric("Honest Prob",    f"{hp:.4f}")
                c2.metric("Malicious Prob", f"{mp:.4f}")
                c3.plotly_chart(create_trust_gauge(hp if lbl == "Honest" else 1 - mp, "ML Trust"),
                                width='stretch')
        else:
            st.warning("No ML features data available")

    # ── TAB 6 — Peers ─────────────────────────────────────────────────────────
    with tab_peers:
        st.header("Peer Network")

        # Registered peers (richer data)
        reg = reg_peers or {}
        peers_dict = reg.get("peers", {})

        if peers_dict:
            st.subheader(f"Registered Peers ({len(peers_dict)})")
            rows = []
            for nid, info in peers_dict.items():
                rows.append({
                    "Node ID":    nid,
                    "URL":        info.get("url", "—"),
                    "Public Key": (info.get("public_key_hex", "") or "")[:16] + "…",
                })
            st.dataframe(pd.DataFrame(rows), width='stretch')
        else:
            peers_data = DashboardAPI.get_peers()
            if peers_data:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Peers",   peers_data.get("total_peers", 0))
                c2.metric("Active Peers",  peers_data.get("active_peers", 0))
                c3.metric("Inactive",      peers_data.get("inactive_peers", 0))
                avg = peers_data.get("average_trust_score", 0)
                c4.metric("Avg Trust",     f"{avg:.4f}")
                peer_list = peers_data.get("peer_list", [])
                if peer_list:
                    st.dataframe(pd.DataFrame(peer_list), width='stretch')
            else:
                st.info("No peers connected yet")

    # ── TAB 7 — Statistics ────────────────────────────────────────────────────
    with tab_stats:
        st.header("System Statistics")
        stats_data = DashboardAPI.get_statistics()
        if stats_data:
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Node ID:** {stats_data.get('node_id', '—')}")
                st.write(f"**Timestamp:** {fmt_ts(stats_data.get('timestamp', ''))}")
                ts = stats_data.get("trust", {})
                if ts:
                    st.write("**Trust Statistics:**")
                    st.write(f"- Total Nodes: {ts.get('total_nodes', 0)}")
                    st.write(f"- Avg Trust: {ts.get('average_trust', 0):.4f}")
                    st.write(f"- Total Reports: {ts.get('total_reports', 0)}")
            with c2:
                ms = stats_data.get("monitoring", {})
                if ms:
                    st.write("**Monitoring:**")
                    st.write(f"- Websites: {ms.get('websites_count', 0)}")
                    st.write(f"- Latest results: {ms.get('latest_results_count', 0)}")
                    st.write(f"- Interval: {ms.get('monitoring_interval', 0)} s")
                bs = stats_data.get("blockchain", {})
                if bs:
                    st.write("**Blockchain:**")
                    st.write(f"- Registered: {bs.get('node_registered', False)}")
                    rep = bs.get("reputation")
                    if rep:
                        st.write(f"- Reputation: {rep.get('reputation', 0):.4f}")
                        st.write(f"- ML Score: {rep.get('ml_score', 0):.4f}")
        else:
            st.warning("No statistics data available")

    # ── Auto-refresh ───────────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
