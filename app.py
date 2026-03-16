"""
NCR Ride Bookings — Multi-Target Prediction Dashboard
Streamlit Application

Author  : Mohamed Tamer
Models  : XGBoost Classifier (Cancellation Risk)
          XGBoost Regressor  (Driver Rating)
          XGBoost Regressor  (Customer Rating)
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import json
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------------
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ===========================================================================
# Application Configuration
# ===========================================================================

st.set_page_config(
    page_title="NCR Ride Bookings — Analytics & Prediction",
    page_icon="assets/icon.png" if os.path.exists("assets/icon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------
COLOR_BG        = "#1a1a2e"   # Deep navy — page background
COLOR_SURFACE   = "#16213e"   # Slightly lighter — cards / sidebar
COLOR_PRIMARY   = "#e94560"   # Crimson accent
COLOR_SECONDARY = "#0f3460"   # Dark blue
COLOR_PURPLE    = "#533483"
COLOR_TEAL      = "#2d6a4f"
COLOR_AMBER     = "#f4a261"
COLOR_SLATE     = "#264653"
COLOR_GOLD      = "#e9c46a"

CHART_PALETTE = [
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_PURPLE,
    COLOR_TEAL, COLOR_AMBER, COLOR_SLATE, COLOR_GOLD,
]

# ---------------------------------------------------------------------------
# Module-Level Constants  (feature schema must match the training notebook)
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES = [
    "Avg VTAT", "Avg CTAT", "Ride Distance", "Booking Value",
    "Hour", "DayOfWeek", "Month", "IsWeekend", "IsRushHour",
    "Pickup_Enc", "Drop_Enc",
]
CATEGORICAL_OHE = ["Vehicle Type"]
MODEL_FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_OHE

VEHICLE_TYPES = ["Auto", "Go Mini", "Go Sedan", "Bike", "Premier Sedan", "eBike", "Uber XL"]

# Complete list of locations present in the training data.
# Stored here so the app has no hard dependency on the CSV being loaded
# before the Predictor page is rendered.
LOCATIONS = sorted([
    "Palam Vihar", "Shastri Nagar", "Khandsa", "Central Secretariat",
    "Ghitorni Village", "AIIMS", "Vaishali", "Mayur Vihar",
    "Noida Sector 62", "Rohini", "Dwarka Sector 21", "Saket",
    "Lajpat Nagar", "Connaught Place", "Karol Bagh", "Rajouri Garden",
    "Pitampura", "Janakpuri", "Uttam Nagar", "Punjabi Bagh",
    "Gurgaon Sector 56", "Cyber Hub", "DLF Phase 1", "Noida Sector 18",
    "Greater Noida", "Faridabad", "Nehru Place", "Khan Market",
    "Hauz Khas", "Green Park", "South Extension", "Inderlok",
    "Malviya Nagar", "Jhilmil", "Adarsh Nagar", "Narsinghpur",
    "Noida Sector 44", "Gurgaon Sector 14", "Vasant Kunj", "Mehrauli",
    "Old Delhi", "Chandni Chowk", "Laxmi Nagar", "Preet Vihar", "Shahdara",
])

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

MODELS_DIR   = "saved_models"
DATASET_PATH = "ncr_ride_bookings.csv"


# ===========================================================================
# Global Stylesheet
# ===========================================================================

st.markdown(
    f"""
    <style>
      /* ── Base layout ──────────────────────────────────────────────────── */
      .stApp                          {{ background-color: {COLOR_BG}; color: #f0f0f0; }}
      [data-testid="stSidebar"]       {{ background-color: {COLOR_SURFACE}; }}
      hr                              {{ border-color: rgba(255,255,255,0.08); }}

      /* ── KPI metric cards ─────────────────────────────────────────────── */
      [data-testid="metric-container"] {{
          background    : {COLOR_SURFACE};
          border        : 1px solid rgba(255,255,255,0.09);
          border-radius : 12px;
          padding       : 16px 20px;
      }}
      [data-testid="stMetricValue"] {{
          color     : {COLOR_PRIMARY};
          font-size : 1.9rem !important;
      }}
      [data-testid="stMetricLabel"] {{ color: #aaaaaa; }}

      /* ── Section heading strip ────────────────────────────────────────── */
      .section-heading {{
          background    : linear-gradient(90deg, {COLOR_SURFACE}, {COLOR_BG});
          border-left   : 4px solid {COLOR_PRIMARY};
          border-radius : 6px;
          padding       : 10px 18px;
          margin        : 24px 0 14px 0;
          font-size     : 1.05rem;
          font-weight   : 600;
          letter-spacing: 0.4px;
          color         : #e0e0e0;
      }}

      /* ── Prediction result cards ──────────────────────────────────────── */
      .result-card {{
          background    : {COLOR_SURFACE};
          border        : 1px solid rgba(255,255,255,0.08);
          border-radius : 14px;
          padding       : 24px;
          margin        : 10px 0;
          text-align    : center;
      }}
      .result-card .card-value {{
          font-size  : 2.4rem;
          font-weight: 700;
          margin     : 8px 0 4px 0;
      }}
      .result-card .card-label {{
          color    : #999;
          font-size: 0.85rem;
          margin   : 0;
      }}
      .result-card .card-sub {{
          color    : #ccc;
          font-size: 1rem;
          margin   : 4px 0 0 0;
      }}

      /* ── Risk alert banners ───────────────────────────────────────────── */
      .risk-low    {{
          border: 1px solid {COLOR_TEAL};
          background: #1a3a2a;
          border-radius: 10px;
          padding: 14px 18px;
      }}
      .risk-medium {{
          border: 1px solid {COLOR_AMBER};
          background: #3a2a10;
          border-radius: 10px;
          padding: 14px 18px;
      }}
      .risk-high   {{
          border: 1px solid {COLOR_PRIMARY};
          background: #3a1a1a;
          border-radius: 10px;
          padding: 14px 18px;
      }}

      /* ── Risk level badge ─────────────────────────────────────────────── */
      .badge {{
          display       : inline-block;
          padding       : 3px 10px;
          border-radius : 20px;
          font-size     : 0.78rem;
          font-weight   : 700;
          letter-spacing: 0.8px;
          text-transform: uppercase;
      }}
      .badge-low    {{ background: {COLOR_TEAL};    color: #fff; }}
      .badge-medium {{ background: {COLOR_AMBER};   color: #111; }}
      .badge-high   {{ background: {COLOR_PRIMARY}; color: #fff; }}

      /* ── Analyst insight box ──────────────────────────────────────────── */
      .insight-box {{
          background    : rgba(22, 33, 62, 0.6);
          border-left   : 3px solid {COLOR_AMBER};
          border-radius : 0 8px 8px 0;
          padding       : 10px 16px;
          margin        : -8px 0 12px 0;
          font-size     : 0.82rem;
          color         : #bbb;
          line-height   : 1.55;
      }}

      /* ── Form input colours ───────────────────────────────────────────── */
      label {{ color: #cccccc !important; }}
      .stSelectbox > div > div,
      .stNumberInput > div > div > input {{
          background-color: #0d1117 !important;
          color: white !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Helper Functions
# ===========================================================================

def apply_dark_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply a consistent dark theme to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e0e0e0", size=14)),
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_BG,
        font=dict(color="#cccccc", size=12),
        margin=dict(l=20, r=20, t=48, b=20),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cccccc")),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.07)",
        zerolinecolor="rgba(255,255,255,0.12)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.07)",
        zerolinecolor="rgba(255,255,255,0.12)",
    )
    return fig


def section_heading(text: str) -> None:
    """Render a styled section heading strip."""
    st.markdown(f"<div class='section-heading'>{text}</div>", unsafe_allow_html=True)


def insight_box(text: str) -> None:
    """Render a short analyst-style observation beneath a chart."""
    st.markdown(f"<div class='insight-box'>{text}</div>", unsafe_allow_html=True)


def build_gauge(value: float, max_value: float, label: str, color: str) -> go.Figure:
    """
    Build a Plotly gauge (Indicator) figure for a single numeric prediction.

    Parameters
    ----------
    value     : The predicted numeric value to display.
    max_value : The upper bound of the gauge scale.
    label     : Title text displayed above the gauge.
    color     : Hex color for the gauge fill bar.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge=dict(
            axis=dict(range=[0, max_value], tickcolor="white"),
            bar=dict(color=color),
            bgcolor=COLOR_BG,
            bordercolor="rgba(255,255,255,0.12)",
            steps=[
                dict(range=[0,              max_value * 0.33], color=COLOR_SURFACE),
                dict(range=[max_value*0.33, max_value * 0.66], color="#1e2a3a"),
                dict(range=[max_value*0.66, max_value       ], color="#0d1826"),
            ],
        ),
        number=dict(font=dict(color="white", size=30)),
        title=dict(text=label, font=dict(color="#aaaaaa", size=12)),
    ))
    fig.update_layout(
        paper_bgcolor=COLOR_SURFACE,
        height=220,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(color="white"),
    )
    return fig


def encode_location(label_encoder, location: str) -> int:
    """
    Safely transform a location string using a fitted LabelEncoder.
    Falls back to 0 if the location was not seen during training.
    """
    if location in label_encoder.classes_:
        return int(label_encoder.transform([location])[0])
    return 0


def star_string(rating: float, filled: str = "\u2605", empty: str = "\u2606") -> str:
    """
    Convert a numeric rating (1-5) into a Unicode star string.
    Example: 4.2 -> '★★★★☆'
    """
    filled_count = int(round(rating))
    return filled * filled_count + empty * (5 - filled_count)


# ===========================================================================
# Data & Model Loading  (cached to avoid redundant I/O on each interaction)
# ===========================================================================

@st.cache_data(show_spinner="Loading dataset, please wait...")
def load_data() -> pd.DataFrame:
    """
    Read the raw bookings CSV and parse temporal columns.
    Results are cached; the CSV is only read once per session.
    """
    df = pd.read_csv(DATASET_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")
    df["Hour"]      = df["Time"].dt.hour
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    return df


@st.cache_resource(show_spinner="Loading prediction models, please wait...")
def load_models() -> dict:
    """
    Load all serialised joblib artefacts from the saved_models directory.

    The function handles three failure modes without crashing the application:

    1. Missing file      -- recorded in models["_missing"]
    2. sklearn version clash (AttributeError or ModuleNotFoundError when
       unpickling a ColumnTransformer built with a different sklearn release)
                         -- recorded in models["_version_error"]
    3. Proactive version warning read from version_manifest.json before any
       load attempt      -- recorded in models["_version_mismatch_warning"]
    """
    import sklearn  # imported here to avoid polluting module namespace

    models: dict = {}

    # Step 1 — Check version_manifest.json written by the training notebook.
    # This lets us warn the user before a load failure rather than after.
    manifest_path = os.path.join(MODELS_DIR, "version_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as manifest_file:
            manifest = json.load(manifest_file)
        training_sklearn = manifest.get("sklearn", "unknown")
        runtime_sklearn  = sklearn.__version__
        if training_sklearn != runtime_sklearn:
            models["_version_mismatch_warning"] = (training_sklearn, runtime_sklearn)

    # Step 2 — Attempt to load each artefact independently so that a single
    # corrupted file does not prevent the others from being available.
    required_artefacts = {
        "clf":       "xgb_cancellation_model.pkl",
        "preproc":   "preprocessor.pkl",
        "dr":        "xgb_driver_rating_pipeline.pkl",
        "cr":        "xgb_customer_rating_pipeline.pkl",
        "le_pickup": "le_pickup.pkl",
        "le_drop":   "le_drop.pkl",
    }
    missing_files  = []
    version_errors = []

    for model_key, filename in required_artefacts.items():
        filepath = os.path.join(MODELS_DIR, filename)

        if not os.path.exists(filepath):
            missing_files.append(filename)
            continue

        try:
            models[model_key] = joblib.load(filepath)

        except AttributeError as exc:
            # Typical when ColumnTransformer internals change between sklearn
            # minor versions, e.g. "_RemainderColsList" in 1.4 vs 1.6.
            version_errors.append((filename, str(exc)))

        except ModuleNotFoundError as exc:
            # Occurs when an entire sklearn submodule is reorganised.
            version_errors.append((filename, str(exc)))

        except Exception as exc:
            # Catch-all for corrupted files, numpy dtype mismatches, etc.
            version_errors.append((filename, f"Unexpected error: {exc}"))

    if missing_files:
        models["_missing"] = missing_files
    if version_errors:
        models["_version_error"]    = version_errors
        models["_sklearn_version"]  = sklearn.__version__

    # If every artefact loaded cleanly and no manifest exists yet, write one
    # now so that future sessions can do a proactive version check before
    # attempting to load the files at all.
    no_errors   = not missing_files and not version_errors
    no_manifest = not os.path.exists(manifest_path)
    if no_errors and no_manifest:
        try:
            import platform
            import xgboost
            manifest_data = {
                "python":   platform.python_version(),
                "sklearn":  sklearn.__version__,
                "xgboost":  xgboost.__version__,
                "joblib":   joblib.__version__,
            }
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(manifest_path, "w") as mf:
                json.dump(manifest_data, mf, indent=2)
        except Exception:
            pass   # Manifest write failure is non-fatal

    return models


# ===========================================================================
# Sidebar Navigation
# ===========================================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 16px 0 20px 0;">
            <p style="font-size:1.8rem; margin:0; color:{COLOR_PRIMARY}; font-weight:700;">
                NCR Ride AI
            </p>
            <p style="color:#777; font-size:0.78rem; margin:4px 0 0 0; letter-spacing:0.5px;">
                ANALYTICS &amp; PREDICTION
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        ["Ride Analysis Dashboard", "Multi-Target Predictor"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        <div style="font-size:0.74rem; color:#666; line-height:1.8; padding: 4px 0;">
            <span style="color:#aaa; font-weight:600;">Prediction Models</span><br>
            XGBoost Cancellation Classifier<br>
            XGBoost Driver Rating Regressor<br>
            XGBoost Customer Rating Regressor<br>
            <br>
            <span style="color:#aaa; font-weight:600;">Dataset</span><br>
            150,000 ride bookings<br>
            Delhi-NCR Region, India
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1 — Ride Analysis Dashboard
# ===========================================================================

if page == "Ride Analysis Dashboard":

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLOR_SURFACE}, {COLOR_BG});
                    border-radius: 14px; padding: 28px 32px; margin-bottom: 24px;
                    border: 1px solid rgba(255,255,255,0.07);">
            <h1 style="margin:0; font-size:1.9rem; font-weight:700;">
                Ride Analysis Dashboard
            </h1>
            <p style="color:#999; margin: 8px 0 0 0; font-size:0.9rem;">
                Operational intelligence across 150,000 NCR bookings — demand patterns,
                service quality, and revenue distribution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load dataset with graceful error handling.
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(
            f"Dataset not found at `{DATASET_PATH}`. "
            "Ensure the CSV file is in the same directory as app.py."
        )
        st.stop()

    completed   = df[df["Booking Status"] == "Completed"]
    total_rides = len(df)
    cancel_mask = df["Booking Status"].str.startswith("Cancelled")

    # -------------------------------------------------------------------
    # KPI Row
    # -------------------------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Total Bookings",      f"{total_rides:,}")
    k2.metric(
        "Completed Rides",
        f"{len(completed):,}",
        delta=f"{len(completed) / total_rides * 100:.1f}% completion rate",
    )
    k3.metric(
        "Cancellation Rate",
        f"{cancel_mask.sum() / total_rides * 100:.1f}%",
    )
    k4.metric(
        "Avg Driver Rating",
        f"{df['Driver Ratings'].mean():.2f} / 5.00",
    )
    k5.metric(
        "Avg Customer Rating",
        f"{df['Customer Rating'].mean():.2f} / 5.00",
    )

    st.divider()

    # -------------------------------------------------------------------
    # Section 1 — Booking Status Breakdown
    # -------------------------------------------------------------------
    section_heading("Booking Status Breakdown")
    insight_box(
        "Completed rides account for roughly 62% of all bookings. "
        "Driver-initiated cancellations represent the largest single failure mode, "
        "suggesting that supply-side retention and dispatch optimisation are the "
        "highest-leverage levers for improving fulfilment rates."
    )

    col_pie, col_bar = st.columns(2)

    status_counts = (
        df["Booking Status"]
        .value_counts()
        .reset_index()
        .rename(columns={"Booking Status": "Status", "count": "Count"})
    )
    # Pandas 2.x uses 'count' as the column name after value_counts reset_index
    if "count" in status_counts.columns:
        status_counts = status_counts.rename(columns={"count": "Count"})
    if "Booking Status" in status_counts.columns and "Status" not in status_counts.columns:
        status_counts = status_counts.rename(columns={"Booking Status": "Status"})

    with col_pie:
        fig_donut = px.pie(
            status_counts,
            values="Count",
            names="Status",
            color_discrete_sequence=CHART_PALETTE,
            hole=0.45,
        )
        fig_donut.update_traces(
            textposition="outside",
            textinfo="percent+label",
            textfont=dict(color="white", size=11),
        )
        apply_dark_theme(fig_donut, "Proportion by Booking Outcome")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        fig_status_bar = go.Figure()
        fig_status_bar.add_trace(go.Bar(
            x=status_counts["Status"],
            y=status_counts["Count"],
            marker_color=CHART_PALETTE[:len(status_counts)],
            text=[f"{v:,}" for v in status_counts["Count"]],
            textposition="outside",
            textfont=dict(color="white"),
        ))
        apply_dark_theme(fig_status_bar, "Volume by Booking Outcome")
        st.plotly_chart(fig_status_bar, use_container_width=True)

    # -------------------------------------------------------------------
    # Section 2 — Rating Distributions
    # -------------------------------------------------------------------
    section_heading("Driver and Customer Rating Distributions")
    insight_box(
        "Both rating distributions are left-skewed and concentrated in the 4.0–5.0 band, "
        "which is typical of ride-hailing platforms where low-rated drivers are periodically "
        "removed. Customers rate slightly more generously on average than drivers, "
        "with a tighter spread suggesting more consistent satisfaction with completed trips."
    )

    col_hist, col_box = st.columns(2)

    with col_hist:
        fig_ratings_hist = go.Figure()
        fig_ratings_hist.add_trace(go.Histogram(
            x=df["Driver Ratings"].dropna(),
            nbinsx=30,
            name="Driver Rating",
            marker_color=COLOR_PRIMARY,
            opacity=0.8,
        ))
        fig_ratings_hist.add_trace(go.Histogram(
            x=df["Customer Rating"].dropna(),
            nbinsx=30,
            name="Customer Rating",
            marker_color=COLOR_SECONDARY,
            opacity=0.8,
        ))
        fig_ratings_hist.update_layout(barmode="overlay")
        apply_dark_theme(fig_ratings_hist, "Rating Frequency Distribution (Overlaid)")
        st.plotly_chart(fig_ratings_hist, use_container_width=True)

    with col_box:
        fig_box = go.Figure()
        for column, color, label in [
            ("Driver Ratings",  COLOR_PRIMARY,   "Driver"),
            ("Customer Rating", COLOR_SECONDARY, "Customer"),
        ]:
            fig_box.add_trace(go.Box(
                y=df[column].dropna(),
                name=label,
                marker_color=color,
                boxmean="sd",
                line=dict(color=color),
            ))
        apply_dark_theme(fig_box, "Rating Spread — Median, IQR, and SD")
        st.plotly_chart(fig_box, use_container_width=True)

    # -------------------------------------------------------------------
    # Section 3 — Revenue and Ratings by Vehicle Type
    # -------------------------------------------------------------------
    section_heading("Revenue and Service Quality by Vehicle Type")
    insight_box(
        "Premier Sedan and Uber XL generate the highest average fare per trip, "
        "reflecting their positioning in the premium segment. Auto and Bike categories "
        "drive the largest share of total volume. Notably, rating scores are fairly uniform "
        "across vehicle classes, indicating that service quality perceptions are not "
        "strongly correlated with the tier of vehicle booked."
    )

    vehicle_summary = (
        completed.groupby("Vehicle Type")
        .agg(
            avg_revenue   = ("Booking Value", "mean"),
            total_revenue = ("Booking Value", "sum"),
            avg_driver_rating   = ("Driver Ratings",  "mean"),
            avg_customer_rating = ("Customer Rating",  "mean"),
            ride_count    = ("Booking ID", "count"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    col_rev, col_rat = st.columns(2)

    with col_rev:
        fig_revenue = go.Figure()
        fig_revenue.add_trace(go.Bar(
            x=vehicle_summary["Vehicle Type"],
            y=vehicle_summary["total_revenue"],
            name="Total Revenue (INR)",
            marker_color=CHART_PALETTE[0],
        ))
        fig_revenue.add_trace(go.Scatter(
            x=vehicle_summary["Vehicle Type"],
            y=vehicle_summary["avg_revenue"],
            name="Avg Revenue per Trip (INR)",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color=COLOR_GOLD, width=2),
            marker=dict(size=8),
        ))
        fig_revenue.update_layout(
            yaxis2=dict(
                overlaying="y",
                side="right",
                title="Avg Revenue per Trip (INR)",
                title_font=dict(color=COLOR_GOLD),
                tickfont=dict(color=COLOR_GOLD),
            )
        )
        apply_dark_theme(fig_revenue, "Revenue by Vehicle Category")
        st.plotly_chart(fig_revenue, use_container_width=True)

    with col_rat:
        fig_ratings_vt = go.Figure()
        fig_ratings_vt.add_trace(go.Bar(
            x=vehicle_summary["Vehicle Type"],
            y=vehicle_summary["avg_driver_rating"],
            name="Avg Driver Rating",
            marker_color=COLOR_PRIMARY,
            text=vehicle_summary["avg_driver_rating"].round(2),
            textposition="outside",
            textfont=dict(color="white"),
        ))
        fig_ratings_vt.add_trace(go.Bar(
            x=vehicle_summary["Vehicle Type"],
            y=vehicle_summary["avg_customer_rating"],
            name="Avg Customer Rating",
            marker_color=COLOR_SECONDARY,
            text=vehicle_summary["avg_customer_rating"].round(2),
            textposition="outside",
            textfont=dict(color="white"),
        ))
        fig_ratings_vt.update_layout(barmode="group")
        apply_dark_theme(fig_ratings_vt, "Average Ratings by Vehicle Category")
        st.plotly_chart(fig_ratings_vt, use_container_width=True)

    # -------------------------------------------------------------------
    # Section 4 — Temporal Demand Patterns
    # -------------------------------------------------------------------
    section_heading("Demand Patterns — Hour of Day and Day of Week")
    insight_box(
        "Booking volumes peak sharply during the morning (8–10 AM) and evening (5–8 PM) "
        "commute windows, a pattern consistent with urban ride-hailing behaviour. "
        "Friday and Saturday record the highest weekly volumes, driven by both end-of-week "
        "commuting and leisure travel. Sunday shows a pronounced morning leisure peak "
        "with a softer evening return compared to weekdays."
    )

    col_hour, col_day = st.columns(2)

    with col_hour:
        hourly_demand = df.groupby("Hour").size().reset_index(name="Bookings")
        fig_hourly = px.area(
            hourly_demand,
            x="Hour",
            y="Bookings",
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        fig_hourly.update_traces(fill="tozeroy", line=dict(width=2))
        apply_dark_theme(fig_hourly, "Bookings by Hour of Day")
        st.plotly_chart(fig_hourly, use_container_width=True)

    with col_day:
        daily_demand = df.groupby("DayOfWeek").size().reset_index(name="Bookings")
        daily_demand["Day"] = daily_demand["DayOfWeek"].map(
            lambda idx: DAY_LABELS[idx]
        )
        fig_daily = px.bar(
            daily_demand,
            x="Day",
            y="Bookings",
            color="Bookings",
            color_continuous_scale="Reds",
        )
        apply_dark_theme(fig_daily, "Bookings by Day of Week")
        st.plotly_chart(fig_daily, use_container_width=True)

    # -------------------------------------------------------------------
    # Raw Data Explorer
    # -------------------------------------------------------------------
    with st.expander("Raw Data Explorer — First 500 Rows", expanded=False):
        st.caption(
            "Colour gradients on numeric columns reflect relative magnitude within "
            "the visible sample. Green indicates higher values; red indicates lower."
        )
        st.dataframe(
            df.head(500).style.background_gradient(
                subset=["Driver Ratings", "Customer Rating", "Ride Distance"],
                cmap="RdYlGn",
            ),
            use_container_width=True,
        )


# ===========================================================================
# PAGE 2 — Multi-Target Predictor
# ===========================================================================

else:

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLOR_SURFACE}, {COLOR_BG});
                    border-radius: 14px; padding: 28px 32px; margin-bottom: 24px;
                    border: 1px solid rgba(255,255,255,0.07);">
            <h1 style="margin:0; font-size:1.9rem; font-weight:700;">
                Multi-Target Predictor
            </h1>
            <p style="color:#999; margin: 8px 0 0 0; font-size:0.9rem;">
                Enter the trip profile below to receive simultaneous predictions for
                cancellation probability, expected driver rating, and anticipated
                customer satisfaction score.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models = load_models()

    # -------------------------------------------------------------------
    # Environment Diagnostics
    # -------------------------------------------------------------------
    if "_version_mismatch_warning" in models:
        training_ver, runtime_ver = models["_version_mismatch_warning"]
        st.warning(
            f"**scikit-learn version mismatch detected.** "
            f"The model artefacts were built with scikit-learn **{training_ver}** "
            f"but the current environment is running **{runtime_ver}**. "
            f"Model loading may fail. See the resolution steps below."
        )

    if "_missing" in models:
        st.error(
            "The following model files were not found in the `saved_models/` directory: "
            f"`{', '.join(models['_missing'])}`. "
            "Please run `Modeling_Notebook.ipynb` in full and then restart this application."
        )

    if "_version_error" in models:
        import re as _re

        runtime_ver = models.get("_sklearn_version", "unknown")

        # Priority 1: version_manifest.json (written by the notebook)
        # Priority 2: parse the sklearn version from the raw error message
        # Priority 3: generic fallback label
        if "_version_mismatch_warning" in models:
            training_ver = models["_version_mismatch_warning"][0]
        else:
            # The error message from joblib usually contains the full module
            # path which encodes the sklearn version in the site-packages path,
            # e.g. "...site-packages/sklearn/..." — not helpful.
            # But we can scan all error strings for a semver pattern like "1.6.1".
            all_errors = " ".join(msg for _, msg in models["_version_error"])
            semver_candidates = _re.findall(r"\b(\d+\.\d+\.\d+)\b", all_errors)
            # Exclude the runtime version itself; the remaining candidate is
            # likely the training version embedded in the pickle stream.
            other_versions = [v for v in semver_candidates if v != runtime_ver]
            if other_versions:
                training_ver = other_versions[0]
            else:
                training_ver = None   # genuinely unknown

        affected_files = [fname for fname, _ in models["_version_error"]]

        # Build the pip command strings conditionally so they are never
        # rendered with placeholder text if the version could not be determined.
        if training_ver:
            option_a = (
                f"**Option A — Downgrade to match the training environment (fastest)**\n\n"
                f"```bash\n"
                f"pip install scikit-learn=={training_ver}\n"
                f"streamlit run app.py\n"
                f"```\n\n"
            )
            trained_with_line = f"serialised with scikit-learn **{training_ver}**"
        else:
            option_a = (
                "**Option A — Downgrade to the training version**\n\n"
                "The exact training version could not be determined automatically.\n"
                "Check your notebook environment with:\n"
                "```bash\n"
                "python -c \"import sklearn; print(sklearn.__version__)\"\n"
                "```\n"
                "Then run: `pip install scikit-learn==<that version>`\n\n"
            )
            trained_with_line = "serialised with a **different scikit-learn version**"

        st.error(
            f"**Model loading failed — scikit-learn version incompatibility.**\n\n"
            f"The `.pkl` artefacts were {trained_with_line} "
            f"but the active environment has **{runtime_ver}** installed. "
            f"The internal structure of `ColumnTransformer` changed between these releases, "
            f"preventing `joblib` from deserialising the preprocessor.\n\n"
            f"---\n\n"
            f"{option_a}"
            f"**Option B — Re-export from the notebook using the current version (recommended)**\n\n"
            f"```bash\n"
            f"pip install scikit-learn=={runtime_ver}\n"
            f"# Open Modeling_Notebook.ipynb and run all cells top-to-bottom,\n"
            f"# then restart this app:\n"
            f"streamlit run app.py\n"
            f"```\n\n"
            f"Affected files: `{', '.join(affected_files)}`"
        )
        for filename, error_msg in models["_version_error"]:
            with st.expander(f"Error detail — {filename}", expanded=False):
                st.code(error_msg, language=None)

    # -------------------------------------------------------------------
    # Input Form
    # -------------------------------------------------------------------
    section_heading("Trip Details")

    left_col, right_col = st.columns(2)

    with left_col:
        vehicle_type  = st.selectbox("Vehicle Type",     VEHICLE_TYPES)
        pickup_loc    = st.selectbox("Pickup Location",  LOCATIONS, index=LOCATIONS.index("Central Secretariat") if "Central Secretariat" in LOCATIONS else 0)
        drop_loc      = st.selectbox("Drop Location",    LOCATIONS, index=LOCATIONS.index("Hauz Khas") if "Hauz Khas" in LOCATIONS else 1)
        ride_distance = st.slider("Ride Distance (km)", min_value=1.0, max_value=50.0, value=15.0, step=0.5)

    with right_col:
        booking_value = st.slider("Estimated Fare (INR)", min_value=50, max_value=2000, value=350, step=10)
        avg_vtat      = st.slider("Vehicle Arrival Time — VTAT (min)", min_value=2.0, max_value=20.0, value=8.5, step=0.5)
        avg_ctat      = st.slider("Customer Arrival Time — CTAT (min)", min_value=10.0, max_value=45.0, value=28.0, step=0.5)
        ride_time     = st.time_input("Ride Departure Time", value=datetime.now().time())

    # Derive temporal features from the user's inputs.
    # Day-of-week and month default to today because only a time is collected;
    # a date picker could be added here to improve these features.
    selected_hour = ride_time.hour
    today         = datetime.now()
    day_of_week   = today.weekday()
    month         = today.month
    is_weekend    = int(day_of_week in [5, 6])
    is_rush_hour  = int((7 <= selected_hour <= 10) or (17 <= selected_hour <= 20))

    # -------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------
    predict_clicked = st.button(
        "Run Prediction",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:
        models_ready = "_missing" not in models and "_version_error" not in models

        if not models_ready:
            st.error(
                "Prediction cannot proceed because one or more model files could not "
                "be loaded. Please resolve the errors shown above."
            )
        else:
            le_pickup = models["le_pickup"]
            le_drop   = models["le_drop"]
            preproc   = models["preproc"]

            pickup_encoded = encode_location(le_pickup, pickup_loc)
            drop_encoded   = encode_location(le_drop,   drop_loc)

            # Build the single-row feature DataFrame that matches the
            # column order the preprocessor was fitted on.
            input_features = {
                "Avg VTAT":      avg_vtat,
                "Avg CTAT":      avg_ctat,
                "Ride Distance": ride_distance,
                "Booking Value": booking_value,
                "Hour":          selected_hour,
                "DayOfWeek":     day_of_week,
                "Month":         month,
                "IsWeekend":     is_weekend,
                "IsRushHour":    is_rush_hour,
                "Pickup_Enc":    pickup_encoded,
                "Drop_Enc":      drop_encoded,
                "Vehicle Type":  vehicle_type,
            }
            prediction_row = pd.DataFrame([input_features])

            # -- Cancellation probability ----------------------------------
            try:
                preprocessed     = preproc.transform(prediction_row[MODEL_FEATURE_COLS])
                cancel_prob      = float(models["clf"].predict_proba(preprocessed)[0][1])
                cancel_pct       = cancel_prob * 100

                if cancel_prob < 0.30:
                    risk_level = "LOW"
                    risk_css   = "risk-low"
                    badge_css  = "badge-low"
                    risk_note  = (
                        "The modelled risk for this trip profile is low. "
                        "Historical patterns suggest a high likelihood of completion "
                        "given the current vehicle type, route, and timing."
                    )
                elif cancel_prob < 0.60:
                    risk_level = "MEDIUM"
                    risk_css   = "risk-medium"
                    badge_css  = "badge-medium"
                    risk_note  = (
                        "A moderate cancellation probability is indicated. "
                        "Consider confirming driver availability closer to departure, "
                        "or adjusting the pickup time to reduce congestion risk."
                    )
                else:
                    risk_level = "HIGH"
                    risk_css   = "risk-high"
                    badge_css  = "badge-high"
                    risk_note  = (
                        "The model flags a high cancellation risk for this profile. "
                        "Factors such as extended VTAT, high demand periods, or "
                        "route complexity may be contributing. "
                        "Consider pre-assigning a driver or offering a fare adjustment."
                    )

            except Exception:
                cancel_pct = None
                risk_level = "UNAVAILABLE"
                risk_css   = "risk-high"
                badge_css  = "badge-high"
                risk_note  = "Cancellation prediction could not be computed."

            # -- Driver rating prediction ----------------------------------
            try:
                driver_rating = float(
                    np.clip(
                        models["dr"].predict(prediction_row[MODEL_FEATURE_COLS])[0],
                        1.0, 5.0,
                    )
                )
            except Exception:
                driver_rating = None

            # -- Customer rating prediction --------------------------------
            try:
                customer_rating = float(
                    np.clip(
                        models["cr"].predict(prediction_row[MODEL_FEATURE_COLS])[0],
                        1.0, 5.0,
                    )
                )
            except Exception:
                customer_rating = None

            # -----------------------------------------------------------
            # Results Display
            # -----------------------------------------------------------
            st.divider()
            section_heading("Prediction Results")

            card1, card2, card3 = st.columns(3)

            with card1:
                cancel_display = f"{cancel_pct:.1f}%" if cancel_pct is not None else "N/A"
                st.markdown(
                    f"""
                    <div class="result-card">
                        <p class="card-label">Cancellation Risk</p>
                        <p class="card-value" style="color:{COLOR_PRIMARY};">
                            {cancel_display}
                        </p>
                        <span class="badge {badge_css}">{risk_level}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with card2:
                dr_display = f"{driver_rating:.2f} / 5.00" if driver_rating else "N/A"
                dr_stars   = star_string(driver_rating) if driver_rating else "-----"
                st.markdown(
                    f"""
                    <div class="result-card">
                        <p class="card-label">Predicted Driver Rating</p>
                        <p class="card-value" style="color:{COLOR_AMBER};">{dr_display}</p>
                        <p class="card-sub" style="letter-spacing:3px;">{dr_stars}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with card3:
                cr_display = f"{customer_rating:.2f} / 5.00" if customer_rating else "N/A"
                cr_stars   = star_string(customer_rating) if customer_rating else "-----"
                st.markdown(
                    f"""
                    <div class="result-card">
                        <p class="card-label">Predicted Customer Rating</p>
                        <p class="card-value" style="color:{COLOR_TEAL};">{cr_display}</p>
                        <p class="card-sub" style="letter-spacing:3px;">{cr_stars}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # -- Cancellation risk narrative ------------------------------
            if cancel_pct is not None:
                st.markdown(
                    f"""
                    <div class="{risk_css}" style="margin-top:16px;">
                        <strong>Cancellation Assessment —
                        <span class="badge {badge_css}">{risk_level}</span>
                        &nbsp;{cancel_pct:.1f}%</strong><br>
                        <span style="color:#ccc; font-size:0.88rem; line-height:1.6;">
                            {risk_note}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # -- Gauge visualisations ------------------------------------
            st.divider()
            gauge1, gauge2, gauge3 = st.columns(3)

            with gauge1:
                if cancel_pct is not None:
                    st.plotly_chart(
                        build_gauge(cancel_pct, 100, "Cancellation Risk (%)", COLOR_PRIMARY),
                        use_container_width=True,
                    )

            with gauge2:
                if driver_rating is not None:
                    st.plotly_chart(
                        build_gauge(driver_rating, 5, "Predicted Driver Rating", COLOR_AMBER),
                        use_container_width=True,
                    )

            with gauge3:
                if customer_rating is not None:
                    st.plotly_chart(
                        build_gauge(customer_rating, 5, "Predicted Customer Rating", COLOR_TEAL),
                        use_container_width=True,
                    )

            # -- Trip feature summary ------------------------------------
            with st.expander("Trip Feature Summary", expanded=False):
                st.caption(
                    "The table below shows all features passed to the prediction models, "
                    "including engineered temporal variables."
                )
                summary_data = {
                    "Feature": [
                        "Vehicle Type", "Pickup Location", "Drop Location",
                        "Distance (km)", "Estimated Fare (INR)", "Departure Hour",
                        "Day of Week", "Rush Hour", "Weekend",
                        "VTAT (min)", "CTAT (min)",
                    ],
                    "Value": [
                        vehicle_type, pickup_loc, drop_loc,
                        ride_distance, f"INR {booking_value:,}", selected_hour,
                        DAY_LABELS[day_of_week], "Yes" if is_rush_hour else "No",
                        "Yes" if is_weekend else "No",
                        avg_vtat, avg_ctat,
                    ],
                }
                st.dataframe(
                    pd.DataFrame(summary_data),
                    use_container_width=True,
                    hide_index=True,
                )


# ===========================================================================
# Footer
# ===========================================================================

st.markdown(
    """
    <div style="text-align:center; color:#444; font-size:0.75rem;
                padding: 32px 0 12px 0; letter-spacing:0.3px;">
        NCR Ride Bookings &nbsp;&mdash;&nbsp;
        Analytics &amp; Prediction Dashboard &nbsp;&mdash;&nbsp;
        XGBoost &nbsp;|&nbsp; Streamlit &nbsp;|&nbsp; Plotly &nbsp;|&nbsp;
        150,000 Bookings
    </div>
    """,
    unsafe_allow_html=True,
)
