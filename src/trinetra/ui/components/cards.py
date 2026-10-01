"""Compact status and metric cards for the dashboard."""

from __future__ import annotations

import streamlit as st

from trinetra.pipeline.contracts import DashboardSnapshot


def render_metric_cards(snapshot: DashboardSnapshot) -> None:
    review_count = sum(item.recommendation == "REVIEW" for item in snapshot.findings)
    quarantine_count = sum(item.recommendation == "QUARANTINE" for item in snapshot.findings)
    values = (
        ("FINDINGS", str(len(snapshot.findings)), "Observed candidates"),
        ("REVIEW REQUIRED", str(review_count), "Analyst follow-up"),
        ("QUARANTINE CANDIDATES", str(quarantine_count), "Recommendation only"),
        ("ASSETS ANALYZED", str(snapshot.total_assets), "Current run"),
    )
    columns = st.columns(4)
    for column, (label, value, hint) in zip(columns, values):
        column.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div><div class="metric-hint">{hint}</div></div>',
            unsafe_allow_html=True,
        )


__all__ = ["render_metric_cards"]
