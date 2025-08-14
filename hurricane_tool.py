
import sys, subprocess
import os
def ensure_user_pkg(mod_name: str, spec: str):
    try:
        __import__(mod_name)
        return
    except ImportError:
        pass

    target = os.path.join(os.path.expanduser("~"), "app-packages")  # e.g. /home/adminuser/app-packages
    os.makedirs(target, exist_ok=True)

    # Install into the target dir (not the read-only venv)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--target", target, spec]
    )
    if target not in sys.path:
        sys.path.insert(0, target)
    __import__(mod_name)

ensure_user_pkg("plotly", "plotly==5.20.0")

import plotly.express as px


import sqlite3
from typing import Set, Optional, Dict
import streamlit as st, sys, subprocess

import pandas as pd
import numpy as np
import streamlit as st, sys, subprocess

import plotly.graph_objects as go 

# ── Streamlit page setup ───────────────────────────────────────────────
st.set_page_config(layout="wide")

# ── Data files (assumed in same folder as this script) ─────────────────
SQL_FILE = "storm_data2.sql"   # NEW: use the SQL dump instead of storm_data.db
PLAT_SQL = "platforms.sql"

# ── SQLite connection ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_conn(sql_file: str = SQL_FILE, file_mtime: Optional[float] = None) -> sqlite3.Connection:
    """
    Build an in-memory SQLite database from a SQL dump file and return the connection.
    The `file_mtime` parameter is used to invalidate Streamlit's cache when the SQL file changes.
    """
    if not os.path.exists(sql_file):
        raise FileNotFoundError(
            f"{sql_file!r} not found — create it with: sqlite3 storm_data.db \".dump\" > {sql_file}"
        )

    # If not provided, compute modification time to act as part of the cache key
    if file_mtime is None:
        file_mtime = os.path.getmtime(sql_file)

    # Create in-memory DB and load SQL
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    with open(sql_file, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    return conn


# ── Data loaders ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_summary() -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM hurricanes", get_conn(SQL_FILE, os.path.getmtime(SQL_FILE)))
    if "max_rain_in" in df.columns:
        df["rain_in"] = df["max_rain_in"]
        df["rain_mm"] = df["max_rain_in"] * 25.4
    else:
        df[["rain_in", "rain_mm"]] = pd.NA
    if {"hit_texas", "hit_louisiana"}.issubset(df.columns):
        df = df[(df.hit_texas == 1) | (df.hit_louisiana == 1)]
    return df


@st.cache_data(show_spinner=False)
def load_tracks(ids: Set[str]) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame(columns=[
            "storm_id", "date", "latitude", "longitude", "wind_kt", "pressure_hPa"
        ])

    placeholders = ",".join(["?"] * len(ids))
    sql = f"""
        SELECT
            storm_id,
            datetime   AS date,
            latitude,
            longitude,
            wind_knots AS wind_kt,
            pressure_hPa
        FROM tracks
        WHERE storm_id IN ({placeholders})
    """
    return pd.read_sql_query(
        sql,
        get_conn(SQL_FILE, os.path.getmtime(SQL_FILE)),
        params=list(ids),
        parse_dates=["date"],
    )


@st.cache_data(show_spinner=False)
def load_platforms() -> pd.DataFrame:
    # unchanged: we still use the standalone platforms.sql
    if not os.path.exists(PLAT_SQL):
        return pd.DataFrame(columns=["latitude", "longitude"])  # graceful fallback
    conn = sqlite3.connect(":memory:")
    with open(PLAT_SQL, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    return pd.read_sql_query("SELECT latitude, longitude FROM platforms", conn)


@st.cache_data(show_spinner=False)
def load_refineries() -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM refineries", get_conn(SQL_FILE, os.path.getmtime(SQL_FILE)))
    return df[
        df.refinery_latitude.between(18, 31)
        & df.refinery_longitude.between(-98, -80)
        & (df.country_name == "United States")
    ]


# ── Bootstrap all data ──────────────────────────────────────────────────
df_summary = load_summary()
df_tracks = load_tracks(set(df_summary.id))
df_platforms = load_platforms()
df_refineries = load_refineries()

# ── Rapid‐intensification flag ──────────────────────────────────────────
ri_flags: Dict[str, bool] = {}
for sid in df_summary.id:
    g = df_tracks[df_tracks.storm_id == sid].sort_values("date")
    ri_flags[sid] = any(
        (
            g[(g.date > r.date) & (g.date <= r.date + pd.Timedelta(hours=24))].wind_kt.max()
            - r.wind_kt
        )
        >= 30
        for _, r in g.iterrows()
    )
df_summary["rapid_intensification"] = df_summary.id.map(ri_flags)


# ── The Streamlit app ──────────────────────────────────────────────────
def main():
    st.title("Hurricane Similarity Dashboard")

    # — Sidebar —
    metrics = {
        "Rainfall (in)": "rain_in",
        "Rainfall (mm)": "rain_mm",
        "Max Wind (m/s)": "max_wind_ms",
        "Min Pressure": "min_mslp_hPa",
        "Surge (ft)": "Surge_FT",
        "Surge (m)": "Surge_M",
        "SST (°C)": "sst",
        "SST Anomaly": "sst_anomaly",
    }
    m_lbl = st.sidebar.selectbox("Metric", list(metrics))
    m_col = metrics[m_lbl]
    default = float(df_summary[m_col].median()) if df_summary[m_col].notna().any() else 0.0
    target = st.sidebar.number_input(f"Target {m_lbl}", value=default)

    phases = sorted(df_summary.enso.dropna().unique())
    sel_ph = st.sidebar.multiselect("ENSO phase", ["All"] + phases, ["All"])
    allowed_e = phases if "All" in sel_ph else sel_ph

    ri_opt = st.sidebar.radio("Rapid Intensification", ["All", "Yes", "No"], 0)
    allowed_r = [True, False] if ri_opt == "All" else [ri_opt == "Yes"]

    base = df_summary.query("enso in @allowed_e and rapid_intensification in @allowed_r").copy()
    base["diff"] = (base[m_col] - target).abs()
    top5 = base.nsmallest(5, "diff")

    fmt = lambda r: f"{r['name']} ({r['id']})" if pd.notna(r['name']) else r['id']
    custom = st.sidebar.multiselect("Storms (optional)", [fmt(r) for _, r in base.iterrows()])

    if custom:
        storm_ids = [c.split("(")[-1][:-1] for c in custom]
        display = custom
    else:
        storm_ids = top5.id.tolist()
        display = [fmt(r) for _, r in top5.iterrows()]

    choice = st.sidebar.radio("Highlight", ["None"] + display)
    highlight_id = None if choice == "None" else storm_ids[display.index(choice)]
    highlight_label = None if choice == "None" else choice
    highlight_name = highlight_label.split(" (")[0] if highlight_label else ""

    # assemble track df
    sel = (
        df_tracks[df_tracks.storm_id.isin(storm_ids)]
        .merge(df_summary, left_on="storm_id", right_on="id", how="left")
    )
    sel["storm_name"] = sel.name.fillna(sel.storm_id)

    col_map, col_info = st.columns([3, 1])
    with col_map:
        # create the map, color-coded by pressure
        fig = px.scatter_geo(
            sel,
            lat="latitude",
            lon="longitude",
            color="pressure_hPa",
            color_continuous_scale="RdYlBu",
            hover_name="storm_name",
            hover_data={"date": True, "wind_kt": ":.0f", "pressure_hPa": ":.0f"},
            projection="natural earth",
        )
        fig.update_traces(marker_size=12)
        # cap the colorbar at ~1000 hPa
        fig.update_coloraxes(cmax=995)

        # black lines for each track
        for sid in storm_ids:
            seq = sel[sel.storm_id == sid].sort_values("date")
            fig.add_trace(
                go.Scattergeo(
                    lat=seq.latitude,
                    lon=seq.longitude,
                    mode="lines",
                    line=dict(color="black", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # platforms, refineries, LPG terminals, highlight, etc.
        if not df_platforms.empty:
            fig.add_trace(
                go.Scattergeo(
                    lat=df_platforms.latitude,
                    lon=df_platforms.longitude,
                    mode="markers",
                    marker=dict(size=4, opacity=0.15, color="gray"),
                    name="Platforms",
                    hoverinfo="skip",
                )
            )

        if not df_refineries.empty:
            fig.add_trace(
                go.Scattergeo(
                    lat=df_refineries.refinery_latitude,
                    lon=df_refineries.refinery_longitude,
                    mode="markers",
                    marker=dict(size=6, symbol="triangle-up", opacity=0.8, color="darkred"),
                    name="Refineries",
                    hovertemplate="Owner: %{customdata[0]}<br>City: %{customdata[1]}<br>Unit: %{customdata[2]}",
                    customdata=df_refineries[["refinery_owner", "city_name", "ref_unit"]].values,
                )
            )

        key_locs = [
            ("Corpus Christi", 27.8006, -97.3964),
            ("Freeport", 28.9670, -95.3508),
            ("Sabine Pass", 29.7104, -93.8519),
            ("Cameron", 29.7443, -93.3070),
            ("Calcasieu Pass", 29.7920, -93.2826),
            ("Plaquemines", 29.4320, -89.5320),
            ("JAX-FL", 30.3322, -81.6557),
            ("Elba Island", 30.6911, -88.0241),
        ]
        fig.add_trace(
            go.Scattergeo(
                lat=[r[1] for r in key_locs],
                lon=[r[2] for r in key_locs],
                mode="markers",
                marker=dict(size=8, symbol="star", color="black"),
                name="LPG terminals",
                hovertext=[r[0] for r in key_locs],
                hoverinfo="text",
            )
        )

        if highlight_id:
            seq = sel[sel.storm_id == highlight_id].sort_values("date")
            name = seq.storm_name.iloc[0]
            fig.add_trace(
                go.Scattergeo(
                    lat=seq.latitude,
                    lon=seq.longitude,
                    mode="lines+markers",
                    line=dict(color="red", width=4),
                    marker=dict(size=14, color="red"),
                    name=name,
                    hoverinfo="text",
                    text=[
                        f"{name}<br>Date: {r.date}<br>Wind: {r.wind_kt} kt"
                        + (
                            f"<br>Pressure: {r.pressure_hPa} hPa"
                            if pd.notna(r.pressure_hPa)
                            else ""
                        )
                        for _, r in seq.iterrows()
                    ],
                )
            )

        fig.update_geos(
            lataxis_range=[15, 35],
            lonaxis_range=[-110, -75],
            showcountries=True,
            showcoastlines=True,
            landcolor="lightgray",
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title="hPa"),
            legend=dict(title="Layers", orientation="h", x=0.5, y=1.02, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        if highlight_id:
            info = df_summary.loc[df_summary.id == highlight_id].squeeze()
            st.markdown(f"### [{highlight_name}]({info.link})", unsafe_allow_html=True)
            st.write(f"**ENSO Phase:** {info.enso}")
            st.write(f"**Rapid Intensification:** {'Yes' if info.rapid_intensification else 'No'}")
            st.write(f"**Max Wind (m/s):** {info.max_wind_ms}")
            st.write(f"**Rainfall:** {info.rain_in} in / {info.rain_mm:.1f} mm")
            st.write(f"**Surge:** {info.Surge_FT} ft / {info.Surge_M} m")
            st.write(f"**Min Pressure:** {info.min_mslp_hPa} hPa")
            st.write(f"**SST:** {info.sst} °C")
            st.write(f"**SST Anomaly:** {info.sst_anomaly} °C")


if __name__ == "__main__":
    main()
