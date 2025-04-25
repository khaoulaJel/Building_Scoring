import streamlit as st
import pandas as pd

def display_building_lookup(
    df: pd.DataFrame,
    info_fields: list[str]
) -> str:
    """
    1) Let the user enter a free-text query
    2) Search across all `info_fields` (case-insensitive substring)
    3) Show a selectbox of matching building_ids (labeled by adresse_ban or ID)
    4) Return the chosen building_id (or None)
    """
    query = st.text_input(
        "Search by address / street / postal code:",
        key="building_search"
    ).strip()
    if not query:
        return None

    mask = pd.Series(False, index=df.index)
    for field in info_fields:
        if field in df.columns:
            mask |= df[field].astype(str).str.contains(query, case=False, na=False)

    results = df[mask]
    if results.empty:
        st.write("No matches found.")
        return None

    def fmt_label(bid: str) -> str:
        row = results.loc[results["building_id"] == bid].iloc[0]
        addr = row.get("adresse_ban") or row.get("adresse_brut") or bid
        return f"{bid} — {addr}"

    selected = st.selectbox(
        "Select a building:",
        options=results["building_id"].tolist(),
        format_func=fmt_label,
        key="building_lookup_select"
    )
    return selected
import streamlit as st
import pandas as pd

import streamlit as st
import pandas as pd

def display_building_classifications(
    df: pd.DataFrame,
    building_id: str
) -> None:
    """
    Display true DPE & GES categories and computed classifications
    as styled cards for the given building_id.
    """
    if building_id is None:
        st.info("🏢 Select a building to view its performance categories.")
        return

    # Fetch the building row
    bd = df.loc[df["building_id"] == building_id].iloc[0]

    # Header
    st.markdown(f"## Building #{building_id}")

    # — True DPE & GES cards —
    true_items = [
        ("Energy Performance (DPE)", bd["true_energy_label"], "#3498db"),  # blue
        ("Emissions Performance (GES)", bd["true_ges_label"], "#8e44ad")  # purple
    ]
    cols = st.columns(len(true_items))
    for col_slot, (title, cat, bg) in zip(cols, true_items):
        col_slot.markdown(
            f"""
            <div style="
                background:{bg};
                border-radius:8px;
                padding:12px;
                text-align:center;
                color:white;
                box-shadow:0 2px 6px rgba(0,0,0,0.15);
            ">
                <h4 style="margin:0;font-family:Arial;">{title}</h4>
                <p style="font-size:24px;font-weight:bold;margin:8px 0 0 0;">
                    {cat}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Computed Classifications")

    # — Computed method cards —
    color_map = {
        "A": "#27ae60",  # green
        "B": "#2ecc71",
        "C": "#f1c40f",
        "D": "#e67e22",
        "E": "#e74c3c",
        "F": "#c0392b"
    }
    class_cols = [c for c in df.columns if c.startswith("class_")]
    methods = [
        (col.replace("class_", "").title(), bd[col])
        for col in class_cols
    ]
    cols = st.columns(len(methods))
    for col_slot, (method, cls) in zip(cols, methods):
        bg = color_map.get(cls, "#95a5a6")
        col_slot.markdown(
            f"""
            <div style="
                background:{bg};
                border-radius:8px;
                padding:12px;
                text-align:center;
                color:white;
                box-shadow:0 2px 6px rgba(0,0,0,0.15);
            ">
                <h4 style="margin:0;font-family:Arial;">{method}</h4>
                <p style="font-size:24px;font-weight:bold;margin:8px 0 0 0;">
                    {cls}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
