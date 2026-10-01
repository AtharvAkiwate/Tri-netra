"""Finding selection and detail panels."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4
import os

import streamlit as st

from trinetra.pipeline.contracts import DashboardAsset, DashboardFinding, DashboardSnapshot
from trinetra.governance.audit import AuditStore, AuditStoreCorruptError, decision_from_finding
from trinetra.ui.components.image_view import render_image_view


def finding_label(finding: DashboardFinding) -> str:
    return f"[{finding.layer}] {finding.title} · {finding.finding_id}"


def render_finding_browser(snapshot: DashboardSnapshot, *, detail: bool = True) -> Optional[DashboardFinding]:
    if not snapshot.findings:
        st.info("No findings are available for this run. Missing detectors are shown under module status.")
        return None
    selected_index = st.selectbox(
        "SELECT FINDING / ASSET",
        range(len(snapshot.findings)),
        format_func=lambda index: finding_label(snapshot.findings[index]),
        key=f"finding-selector-{snapshot.run_id}",
    )
    selected = snapshot.findings[selected_index]
    if detail:
        render_finding_detail(snapshot, selected)
    return selected


def render_finding_detail(snapshot: DashboardSnapshot, finding: DashboardFinding) -> None:
    left, right = st.columns([1.12, 0.88], gap="large")
    with left:
        asset_id = finding.affected_image_id
        asset = next((item for item in snapshot.assets if item.image_id == asset_id), None)
        if asset is not None:
            render_image_view(
                st,
                asset,
                image_bytes=snapshot.image_artifacts.get(asset.image_id),
                heatmap=snapshot.heatmap_artifacts.get(asset.image_id),
                pipeline_mode=snapshot.mode == "pipeline",
            )
        elif finding.image_path:
            path = Path(finding.image_path)
            path_asset = DashboardAsset(
                image_id=asset_id if asset_id is not None else finding.finding_id,
                file_name=path.name,
                image_path=str(path),
                width=0,
                height=0,
            )
            render_image_view(
                st,
                path_asset,
                heatmap=snapshot.heatmap_artifacts.get(asset_id) if asset_id is not None else None,
                pipeline_mode=snapshot.mode == "pipeline",
            )
        else:
            st.info("No image path is available for this finding; evidence is still shown.")
        if len(finding.affected_image_ids) > 1:
            st.caption("Related assets: " + ", ".join(map(str, finding.affected_image_ids)))
    with right:
        st.markdown(f"### {finding.title}")
        st.markdown(
            f'<span class="severity severity-{finding.severity.lower()}">{finding.severity.upper()}</span>'
            f' &nbsp; <span class="layer-tag">LAYER {finding.layer}</span>',
            unsafe_allow_html=True,
        )
        if finding.confidence is not None:
            st.metric("DEMO SCORE" if snapshot.mode == "demo" else "SCORE", f"{finding.confidence:.2f}")
        st.markdown("**Affected asset**")
        st.write(str(finding.affected_image_id) if finding.affected_image_id is not None else "Group-level finding")
        st.markdown("**Reason**")
        st.write(finding.reason)
        st.markdown("**DETECTOR RECOMMENDATION**")
        st.write(finding.recommendation)
        st.markdown("**Source detector**")
        st.code(finding.source_detector)
        st.markdown("**Evidence**")
        st.json(dict(finding.evidence), expanded=True)
        if finding.metadata:
            st.markdown("**Provenance / metadata**")
            st.json(dict(finding.metadata), expanded=False)
        st.caption(f"Original typed result retained as `{finding.original_result_reference}`")
    render_governance_actions(snapshot, finding)


def _audit_store() -> AuditStore:
    return AuditStore(os.environ.get("TRINETRA_AUDIT_PATH", "outputs/audit/events.jsonl"))


def render_governance_actions(snapshot: DashboardSnapshot, finding: DashboardFinding) -> None:
    st.markdown("#### ANALYST GOVERNANCE DECISION")
    if snapshot.mode == "demo":
        st.warning("DEMO / SAMPLE WORKFLOW — decisions are local demonstration records, not operational actions.")
    store = _audit_store()
    try:
        current = store.latest_decision(run_id=finding.run_id, finding_id=finding.finding_id)
    except (AuditStoreCorruptError, OSError) as exc:
        st.error(f"Local audit log cannot be read safely: {exc}")
        return
    if current is not None:
        st.success(f"ANALYST DECISION · {current.analyst_decision}")
        st.caption(f"Recorded by {current.actor} at {current.timestamp}. Rationale: {current.rationale}")
    else:
        st.info("ANALYST DECISION · Not yet recorded")
    confirmation = st.session_state.pop("trinetra_governance_confirmation", None)
    if confirmation:
        st.toast(confirmation, icon="✅")

    with st.form(key=f"governance-{snapshot.mode}-{finding.run_id}-{finding.finding_id}"):
        actor_label = "DEMO ACTOR (SAMPLE ONLY)" if snapshot.mode == "demo" else "ANALYST / ACTOR"
        actor = st.text_input(actor_label, value="demo-analyst" if snapshot.mode == "demo" else "")
        rationale = st.text_area("RATIONALE (REQUIRED)", max_chars=1000,
                                 placeholder="Briefly explain the basis for this decision.")
        confirmed = st.checkbox("I reviewed the evidence and intend to record this decision.")
        st.caption(f"Detector recommendation remains: {finding.recommendation}")
        cols = st.columns(3)
        accepted = cols[0].form_submit_button("ACCEPT", width="stretch")
        review = cols[1].form_submit_button("REVIEW", width="stretch")
        quarantine = cols[2].form_submit_button("QUARANTINE", width="stretch")
    chosen = "ACCEPT" if accepted else "REVIEW" if review else "QUARANTINE" if quarantine else None
    if chosen:
        if not actor.strip() or not rationale.strip() or not confirmed:
            st.error("Enter an actor and non-empty rationale, then confirm before recording the decision.")
            return
        try:
            decision = decision_from_finding(finding, decision=chosen, actor=actor,
                rationale=rationale, mode=snapshot.mode, decision_id=str(uuid4()))
            store.append_decision(decision)
        except (AuditStoreCorruptError, OSError, ValueError) as exc:
            st.error(f"Decision was not recorded: {exc}")
            return
        st.session_state["trinetra_governance_confirmation"] = f"{chosen} recorded for {finding.finding_id}."
        st.rerun()


__all__ = ["finding_label", "render_finding_browser", "render_finding_detail", "render_governance_actions"]
