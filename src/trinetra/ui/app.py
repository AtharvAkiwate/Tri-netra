"""TRI-NETRA Streamlit governance dashboard entry point."""

from __future__ import annotations

import streamlit as st
from trinetra.governance.audit import AuditStore, AuditStoreCorruptError
import os

from trinetra.pipeline.contracts import DataIntegrityRun
from trinetra.ui.components.cards import render_metric_cards
from trinetra.ui.components.findings import render_finding_browser
from trinetra.ui.components.navigation import render_navigation
from trinetra.ui.providers import DemoProvider, PipelineProvider

st.set_page_config(
    page_title="TRI-NETRA Governance",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp { background: #0b1118; color: #d9e3e7; }
      [data-testid="stSidebar"] { background: #101922; border-right: 1px solid #25333d; }
      [data-testid="stHeader"] { background: rgba(11,17,24,.94); }
      h1, h2, h3 { color: #e7f0f1 !important; letter-spacing: .025em; }
      .eyebrow { color: #68c3be; font-size: .72rem; letter-spacing: .18em; font-weight: 700; }
      .subtle { color: #8fa2ab; }
      .metric-card { background: #111c25; border: 1px solid #293943; border-radius: 9px;
        padding: 18px 18px 15px; min-height: 112px; }
      .metric-label { color: #91a6af; font-size: .68rem; letter-spacing: .12em; font-weight: 700; }
      .metric-value { color: #e4eff0; font-size: 1.9rem; line-height: 1.3; font-weight: 650; }
      .metric-hint { color: #728891; font-size: .76rem; }
      .severity { padding: 4px 9px; border-radius: 4px; font-size: .7rem; font-weight: 700;
        letter-spacing: .08em; background: #47371b; color: #e6bd68; }
      .severity-high, .severity-critical { background: #482326; color: #fa8883; }
      .severity-info, .severity-low { background: #15343a; color: #76d4cd; }
      .severity-medium, .severity-warning { background: #47371b; color: #e6bd68; }
      .layer-tag { color: #7dc7cf; font-size: .72rem; letter-spacing: .08em; }
      .status-chip { display: inline-block; border: 1px solid #31524b; background: #172b27;
        color: #9bc8ae; border-radius: 4px; padding: 4px 8px; margin: 3px; font-size: .72rem; }
      .demo-chip { border-color: #9d6b2c; color: #f1c36e; background: #332719; }
      div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #293943; }
      .stButton > button { border-color: #3b5059; }
      hr { border-color: #273740; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    st.sidebar.markdown('<div class="eyebrow">MISSION CONTROL / TRI-NETRA</div>', unsafe_allow_html=True)
    st.sidebar.title("GOVERNANCE")
    mode_label = st.sidebar.radio(
        "DATA SOURCE",
        ("DEMO / SAMPLE", "TRI-NETRA PIPELINE"),
        index=0,
        help="Demo data is synthetic. Pipeline mode uses a DataIntegrityRun supplied by the TRI-NETRA runner.",
    )
    if mode_label == "DEMO / SAMPLE":
        snapshot = DemoProvider().load()
    else:
        run = st.session_state.get("trinetra_data_integrity_run")
        if run is not None and not isinstance(run, DataIntegrityRun):
            st.sidebar.error("Pipeline session object must be a DataIntegrityRun.")
            run = None
        snapshot = PipelineProvider(run).load()

    page = render_navigation(st)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"RUN  ·  {snapshot.run_id}")
    st.sidebar.caption(f"DATASET  ·  {snapshot.dataset_name}")
    st.sidebar.markdown('<span class="status-chip">LOCAL UI FOUNDATION</span>', unsafe_allow_html=True)
    if snapshot.mode == "demo":
        st.sidebar.markdown(f'<span class="status-chip demo-chip">{snapshot.demo_mark}</span>', unsafe_allow_html=True)

    if page == "OVERVIEW":
        _render_overview(snapshot)
    elif page in {"FINDINGS", "DATA INTEGRITY"}:
        _render_integrity(snapshot, show_all=page == "DATA INTEGRITY")
    elif page == "AUDIT LOG":
        _render_audit_log(snapshot)
    else:
        _render_not_connected(page)


def _render_overview(snapshot) -> None:
    st.markdown('<div class="eyebrow">TRUSTWORTHY AI SECURITY & ASSURANCE PLATFORM</div>', unsafe_allow_html=True)
    st.title("TRI-NETRA")
    st.markdown("Evidence-first data integrity review across computer-vision assets.")
    if snapshot.mode == "demo":
        st.warning("DEMO / SAMPLE DATA — synthetic findings and imagery; not operational analysis.")
    elif snapshot.run_id == "pipeline-not-connected":
        st.info("TRI-NETRA PIPELINE mode is ready. No DataIntegrityRun has been supplied to this session.")

    render_metric_cards(snapshot)
    left, right = st.columns([1.1, .9], gap="large")
    with left:
        st.markdown("### RUN CONTEXT")
        st.write(f"**Run ID:** `{snapshot.run_id}`")
        st.write(f"**Dataset:** {snapshot.dataset_name}")
        st.write(f"**Assets:** {snapshot.total_assets}")
        st.write(f"**Mode:** {'DEMO / SAMPLE' if snapshot.mode == 'demo' else 'TRI-NETRA PIPELINE'}")
        feature_context = snapshot.metadata.get("feature_context")
        if feature_context:
            st.markdown("**Layer 1B feature context**")
            st.caption(
                f"{feature_context.get('model_id')} · {feature_context.get('embedding_dim')}D · "
                f"{feature_context.get('device')} · space `{feature_context.get('embedding_space_id')}`"
            )
    with right:
        st.markdown("### ACTIVE INTEGRITY MODULES")
        if snapshot.active_modules:
            for module in snapshot.active_modules:
                st.markdown(f'<span class="status-chip">{module}</span>', unsafe_allow_html=True)
        else:
            st.info("No pipeline modules have supplied results.")
        if snapshot.unavailable_modules:
            st.caption("No result supplied: " + " · ".join(snapshot.unavailable_modules))

    st.markdown("### RECENT FINDINGS")
    if not snapshot.findings:
        st.info("No findings available.")
    else:
        for finding in snapshot.findings[:4]:
            with st.container(border=True):
                c1, c2, c3 = st.columns([.12, .63, .25])
                c1.markdown(f"**L{finding.layer}**")
                c2.markdown(f"**{finding.title}**  \n{finding.reason}")
                c3.markdown(f"`{finding.severity.upper()}` · {finding.recommendation}")


def _render_integrity(snapshot, *, show_all: bool) -> None:
    st.markdown('<div class="eyebrow">DATA INTEGRITY / LAYERS 1A–1H</div>', unsafe_allow_html=True)
    st.title("Findings & Asset Review" if not show_all else "Data Integrity")
    if snapshot.mode == "demo":
        st.warning("DEMO / SAMPLE DATA — all displayed findings and the heatmap are synthetic.")
    if snapshot.mode == "pipeline" and not snapshot.assets and not snapshot.findings:
        st.info("No TRI-NETRA pipeline results are connected. Supply a DataIntegrityRun through the pipeline provider.")
        return
    if show_all:
        st.markdown("**Module coverage**")
        st.write("Active: " + (" · ".join(snapshot.active_modules) or "none"))
        if snapshot.unavailable_modules:
            st.write("No result supplied: " + " · ".join(snapshot.unavailable_modules))
        st.markdown("---")
    render_finding_browser(snapshot)


def _render_not_connected(page: str) -> None:
    st.markdown('<div class="eyebrow">GOVERNANCE PLATFORM</div>', unsafe_allow_html=True)
    st.title(page.title())
    st.info("MODULE NOT CONNECTED")
    st.caption("This module is reserved for future integration. No functionality is simulated here.")


def _render_audit_log(snapshot) -> None:
    st.markdown('<div class="eyebrow">LOCAL GOVERNANCE RECORD</div>', unsafe_allow_html=True)
    st.title("Audit Log")
    if snapshot.mode == "demo":
        st.warning("DEMO / SAMPLE MODE — any decisions shown are local demonstration records, not operational actions.")
    store = AuditStore(os.environ.get("TRINETRA_AUDIT_PATH", "outputs/audit/events.jsonl"))
    try:
        events = store.load_events()
        payload = store.export_json()
    except (AuditStoreCorruptError, OSError) as exc:
        st.error(f"Audit history is unavailable and was not changed: {exc}")
        return
    st.metric("EVENT COUNT", len(events))
    if events:
        rows = [{
            "Timestamp": event.timestamp,
            "Event": event.event_type,
            "Finding": event.finding_id,
            "Asset": str(event.affected_image_id) if event.affected_image_id is not None else "—",
            "Analyst decision": event.analyst_decision or "—",
            "Recommendation": event.source_recommendation or "—",
            "Detector": event.source_detector or "—",
            "Actor": event.actor,
            "Mode": event.mode.upper(),
        } for event in reversed(events)]
        st.markdown("### LATEST EVENTS")
        for row in rows[:50]:
            with st.container(border=True):
                st.markdown(f"**{row['Event'].replace('_', ' ').upper()}** · {row['Timestamp']}")
                st.write(f"Finding `{row['Finding']}` · Asset `{row['Asset']}`")
                st.write(
                    f"Analyst decision: **{row['Analyst decision']}** · "
                    f"Recommendation: {row['Recommendation']} · Detector: `{row['Detector']}`"
                )
                st.caption(f"Actor: {row['Actor']} · Mode: {row['Mode']}")
        st.caption(f"Showing latest {min(50, len(rows))} events; export contains all {len(events)} events.")
    else:
        st.info("No governance events have been recorded yet.")
    st.download_button("EXPORT AUDIT LOG JSON", data=payload, file_name="trinetra-audit-log.json",
                       mime="application/json", width="stretch")
    st.caption("Export is read-only and includes the complete persisted event history.")


if __name__ == "__main__":
    main()
