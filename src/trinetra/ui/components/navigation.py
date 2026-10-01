"""Dashboard navigation options."""

from __future__ import annotations

NAV_ITEMS = (
    "OVERVIEW",
    "FINDINGS",
    "DATA INTEGRITY",
    "MODEL INTEGRITY",
    "INFERENCE",
    "AUDIT LOG",
)


def render_navigation(st, *, location: str = "sidebar") -> str:
    if location == "sidebar":
        return st.sidebar.radio("MISSION NAVIGATION", NAV_ITEMS, label_visibility="collapsed")
    return st.radio("MISSION NAVIGATION", NAV_ITEMS, horizontal=True, label_visibility="collapsed")


__all__ = ["NAV_ITEMS", "render_navigation"]
