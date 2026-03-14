"""
╔══════════════════════════════════════════════════════════════╗
║   NCR Ride Bookings — Multi-Target Prediction Dashboard      ║
║   Streamlit Application                                      ║
║   Author  : Data Science Team                                ║
║   Models  : XGBoost (Cancellation, Driver Rating,           ║
║             Customer Rating)                                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NCR Ride Bookings AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color Palette ─────────────────────────────────────────────────────────────
UBER_DARK   = "#1a1a2e"
UBER_ACCENT = "#16213e"
UBER_RED    = "#e94560"
UBER_BLUE   = "#0f3460"
UBER_PURPLE = "#533483"
UBER_TEAL   = "#2d6a4f"
CHART_COLORS = [UBER_RED, UBER_BLUE, UBER_PURPLE, UBER_TEAL, "#f4a261", "#264653", "#e9c46a"]

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* App background */
  .stApp {{ background-color: {UBER_DARK}; color: #f0f0f0; }}
  [data-testid="stSidebar"] {{ background-color: {UBER_ACCENT}; }}

  /* Metric cards */
  [data-testid="metric-container"] {{
      background: {UBER_ACCENT};
      border: 1px solid #ffffff18;
      border-radius: 12px;
      padding: 16px 20px;
  }}
  [data-testid="stMetricValue"] {{ color: {UBER_RED}; font-size: 2rem !important; }}
  [data-testid="stMetricLabel"] {{ color: #aaaaaa; }}

  /* Section header */
  .section-header {{
      background: linear-gradient(90deg, {UBER_ACCENT}, {UBER_DARK});
      border-left: 4px solid {UBER_RED};
      padding: 10px 18px;
      border-radius: 6px;
      margin: 24px 0 14px 0;
      font-size: 1.15rem;
      font-weight: 600;
      letter-spacing: 0.5px;
  }}

  /* Prediction card */
  .pred-card {{
      background: {UBER_ACCENT};
      border: 1px solid #ffffff15;
      border-radius: 14px;
      padding: 22px 24px;
      margin: 10px 0;
      text-align: center;
  }}
  .pred-card h2 {{ font-size: 2.6rem; margin: 6px 0; }}
  .pred-card p  {{ color: #aaaaaa; font-size: 0.9rem; margin: 0; }}

  /* Alert boxes */
  .alert-low    {{ border: 1px solid #2d6a4f; background: #1a3a2a; border-radius:10px; padding:14px 18px; }}
  .alert-medium {{ border: 1px solid #f4a261; background: #3a2a10; border-radius:10px; padding:14px 18px; }}
  .alert-high   {{ border: 1px solid {UBER_RED}; background: #3a1a1a; border-radius:10px; padding:14px 18px; }}

  /* Input labels */
  label {{ color: #cccccc !important; }}
  .stSelectbox > div > div, .stNumberInput > div > div > input {{
      background-color: #0d1117 !important;
      color: white !important;
  }}

  /* Dividers */
  hr {{ border-color: #ffffff15; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Data & Model Loading  (cached for performance)
# ══════════════════════════════════════════════════════════════════════════════
MODELS_DIR  = "saved_models"
DATASET_PATH = "ncr_ride_bookings.csv"

@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    """Load and lightly preprocess the raw bookings CSV."""
    df = pd.read_csv(DATASET_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
    df['Hour']      = df['Time'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    return df

@st.cache_resource(show_spinner="Loading ML models…")
def load_models() -> dict:
    """Load all serialised joblib models / encoders.

    Robustly handles three failure modes:
      1. File missing          → models["_missing"] list
      2. sklearn version clash → models["_version_error"] list
         (AttributeError  = internal class renamed/moved between versions,
          ModuleNotFoundError = entire submodule reorganised between versions)
      3. Proactive pre-load check via version_manifest.json
         → models["_version_mismatch_warning"] = (train_ver, app_ver)
    """
    import sklearn
    import json as _json

    models: dict = {}

    # ── Step 1: Read version_manifest.json (written by the notebook) ──────
    # This gives us the exact sklearn version the .pkl files were built with,
    # so we can show a clear fix command before the load even fails.
    manifest_path = os.path.join(MODELS_DIR, "version_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as _f:
            manifest = _json.load(_f)
        train_sk = manifest.get("sklearn", "unknown")
        app_sk   = sklearn.__version__
        if train_sk != app_sk:
            # Store tuple so UI can print: "trained with X, running Y"
            models["_version_mismatch_warning"] = (train_sk, app_sk)

    # ── Step 2: Load each artefact with per-file error isolation ──────────
    required = {
        "clf":       "xgb_cancellation_model.pkl",   # XGBoost — no ColumnTransformer
        "preproc":   "preprocessor.pkl",              # ColumnTransformer — most fragile
        "dr":        "xgb_driver_rating_pipeline.pkl",# Pipeline(ColumnTransformer + XGB)
        "cr":        "xgb_customer_rating_pipeline.pkl",
        "le_pickup": "le_pickup.pkl",                 # LabelEncoder — very stable
        "le_drop":   "le_drop.pkl",
    }
    missing        = []
    version_errors = []   # list of (filename, error_message)

    for key, fname in required.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        try:
            models[key] = joblib.load(path)

        except AttributeError as exc:
            # Most common symptom of sklearn version mismatch:
            # "Can't get attribute '_RemainderColsList' on ..."
            # Caused by ColumnTransformer internals changing between 1.x versions.
            version_errors.append((fname, str(exc)))

        except ModuleNotFoundError as exc:
            # Happens when an entire sklearn submodule was reorganised
            # (e.g. sklearn.utils.estimator_checks moved between 1.5 → 1.6).
            version_errors.append((fname, str(exc)))

        except Exception as exc:
            # Catch-all: corrupted file, numpy dtype mismatch, etc.
            version_errors.append((fname, f"Unexpected error: {exc}"))

    if missing:
        models["_missing"] = missing
    if version_errors:
        models["_version_error"]   = version_errors
        models["_sklearn_version"] = sklearn.__version__

    return models


def plotly_layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply unified dark theme to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=14)),
        paper_bgcolor=UBER_ACCENT,
        plot_bgcolor=UBER_DARK,
        font=dict(color="#cccccc"),
        margin=dict(l=20, r=20, t=45, b=20),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.13)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.13)")
    return fig


# ── Known dataset constants (avoids re-scanning the full CSV for UI dropdowns)
VEHICLE_TYPES = ["Auto", "Go Mini", "Go Sedan", "Bike", "Premier Sedan", "eBike", "Uber XL"]
LOCATIONS = [
    "Palam Vihar", "Shastri Nagar", "Khandsa", "Central Secretariat", "Ghitorni Village",
    "AIIMS", "Vaishali", "Mayur Vihar", "Noida Sector 62", "Rohini", "Dwarka Sector 21",
    "Saket", "Lajpat Nagar", "Connaught Place", "Karol Bagh", "Rajouri Garden",
    "Pitampura", "Janakpuri", "Uttam Nagar", "Punjabi Bagh", "Gurgaon Sector 56",
    "Cyber Hub", "DLF Phase 1", "Noida Sector 18", "Greater Noida", "Faridabad",
    "Nehru Place", "Khan Market", "Hauz Khas", "Green Park", "South Extension",
    "Inderlok", "Malviya Nagar", "Jhilmil", "Adarsh Nagar", "Narsinghpur",
    "Noida Sector 44", "Gurgaon Sector 14", "Vasant Kunj", "Mehrauli",
    "Old Delhi", "Chandni Chowk", "Laxmi Nagar", "Preet Vihar", "Shahdara",
]

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <h1 style='font-size:2rem; margin:0;'>🚗</h1>
        <h2 style='font-size:1.2rem; color:#e94560; margin:6px 0 0 0;'>NCR Ride AI</h2>
        <p style='color:#888; font-size:0.8rem; margin:2px 0 0 0;'>Powered by XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate to:",
        ["📊  Ride Analysis Dashboard", "🎯  Multi-Target Predictor"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("""
    <div style='font-size:0.75rem; color:#666; padding: 10px 0;'>
        <b style='color:#aaa;'>Models</b><br>
        • XGBoost Cancellation Classifier<br>
        • XGBoost Driver Rating Regressor<br>
        • XGBoost Customer Rating Regressor<br><br>
        <b style='color:#aaa;'>Dataset</b><br>
        • 150,000 ride bookings<br>
        • NCR Region, India
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Ride Analysis Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Ride Analysis Dashboard":

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {UBER_ACCENT}, {UBER_DARK});
                border-radius: 14px; padding: 28px 32px; margin-bottom: 24px;
                border: 1px solid #ffffff10;'>
        <h1 style='margin:0; font-size:2rem;'>📊 Ride Analysis Dashboard</h1>
        <p style='color:#aaaaaa; margin: 8px 0 0 0;'>
            Exploratory insights across 150,000 NCR bookings
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"Dataset `{DATASET_PATH}` not found. Place it in the same directory as `app.py`.")
        st.stop()

    # ── KPI Row ────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    completed = df[df['Booking Status'] == 'Completed']
    total = len(df)

    col1.metric("Total Bookings",   f"{total:,}")
    col2.metric("Completed",        f"{len(completed):,}",
                delta=f"{len(completed)/total*100:.1f}%")
    col3.metric("Cancellation Rate",
                f"{(df['Booking Status'].str.startswith('Cancelled').sum()/total*100):.1f}%")
    col4.metric("Avg Driver Rating",
                f"{df['Driver Ratings'].mean():.2f} ⭐")
    col5.metric("Avg Customer Rating",
                f"{df['Customer Rating'].mean():.2f} ⭐")

    st.divider()

    # ── Chart Row 1: Booking Status + Rating Distributions ─────────────────
    st.markdown("<div class='section-header'>🗺️ Booking Status Overview</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        status_counts = df['Booking Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_status = px.pie(
            status_counts, values='Count', names='Status',
            color_discrete_sequence=CHART_COLORS,
            hole=0.45,
        )
        fig_status.update_traces(
            textposition='outside', textinfo='percent+label',
            textfont=dict(color='white', size=11),
        )
        plotly_layout(fig_status, "Booking Status Distribution")
        st.plotly_chart(fig_status, use_container_width=True)

    with c2:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=status_counts['Status'],
            y=status_counts['Count'],
            marker_color=CHART_COLORS[:len(status_counts)],
            text=status_counts['Count'].apply(lambda x: f"{x:,}"),
            textposition='outside',
            textfont=dict(color='white'),
        ))
        plotly_layout(fig_bar, "Bookings per Status")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Chart Row 2: Rating Distributions ──────────────────────────────────
    st.markdown("<div class='section-header'>⭐ Driver vs Customer Rating Distributions</div>",
                unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        fig_dr_hist = go.Figure()
        fig_dr_hist.add_trace(go.Histogram(
            x=df['Driver Ratings'].dropna(),
            nbinsx=30, name='Driver Rating',
            marker_color=UBER_RED, opacity=0.8,
        ))
        fig_dr_hist.add_trace(go.Histogram(
            x=df['Customer Rating'].dropna(),
            nbinsx=30, name='Customer Rating',
            marker_color=UBER_BLUE, opacity=0.8,
        ))
        fig_dr_hist.update_layout(barmode='overlay')
        plotly_layout(fig_dr_hist, "Rating Distributions (Overlaid)")
        st.plotly_chart(fig_dr_hist, use_container_width=True)

    with c4:
        fig_box = go.Figure()
        for col_name, color, label in [
            ('Driver Ratings', UBER_RED, 'Driver'),
            ('Customer Rating', UBER_BLUE, 'Customer'),
        ]:
            fig_box.add_trace(go.Box(
                y=df[col_name].dropna(),
                name=label,
                marker_color=color,
                boxmean='sd',
                line=dict(color=color),
            ))
        plotly_layout(fig_box, "Rating Box Plots (Mean ± SD)")
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Chart Row 3: By Vehicle Type ────────────────────────────────────────
    st.markdown("<div class='section-header'>🚙 Revenue & Ratings by Vehicle Type</div>",
                unsafe_allow_html=True)

    vt_agg = (
        completed.groupby('Vehicle Type')
        .agg(
            Avg_Revenue    =('Booking Value', 'mean'),
            Total_Revenue  =('Booking Value', 'sum'),
            Avg_Driver_Rat =('Driver Ratings', 'mean'),
            Avg_Cust_Rat   =('Customer Rating', 'mean'),
            Ride_Count     =('Booking ID', 'count'),
        )
        .reset_index()
        .sort_values('Total_Revenue', ascending=False)
    )

    c5, c6 = st.columns(2)

    with c5:
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(
            x=vt_agg['Vehicle Type'],
            y=vt_agg['Total_Revenue'],
            name='Total Revenue (₹)',
            marker_color=CHART_COLORS[0],
        ))
        fig_rev.add_trace(go.Scatter(
            x=vt_agg['Vehicle Type'],
            y=vt_agg['Avg_Revenue'],
            name='Avg Revenue (₹)',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='yellow', width=2),
            marker=dict(size=8),
        ))
        fig_rev.update_layout(
            yaxis2=dict(
                overlaying='y', side='right',
                title='Avg Revenue (₹)',
                title_font=dict(color='yellow'),
                tickfont=dict(color='yellow'),
            )
        )
        plotly_layout(fig_rev, "Revenue by Vehicle Type")
        st.plotly_chart(fig_rev, use_container_width=True)

    with c6:
        fig_ratings_vt = go.Figure()
        fig_ratings_vt.add_trace(go.Bar(
            x=vt_agg['Vehicle Type'],
            y=vt_agg['Avg_Driver_Rat'],
            name='Avg Driver Rating',
            marker_color=UBER_RED,
            text=vt_agg['Avg_Driver_Rat'].round(2),
            textposition='outside', textfont=dict(color='white'),
        ))
        fig_ratings_vt.add_trace(go.Bar(
            x=vt_agg['Vehicle Type'],
            y=vt_agg['Avg_Cust_Rat'],
            name='Avg Customer Rating',
            marker_color=UBER_BLUE,
            text=vt_agg['Avg_Cust_Rat'].round(2),
            textposition='outside', textfont=dict(color='white'),
        ))
        fig_ratings_vt.update_layout(barmode='group')
        plotly_layout(fig_ratings_vt, "Avg Ratings by Vehicle Type")
        st.plotly_chart(fig_ratings_vt, use_container_width=True)

    # ── Chart Row 4: Hourly & Day Patterns ─────────────────────────────────
    st.markdown("<div class='section-header'>🕐 Temporal Ride Patterns</div>",
                unsafe_allow_html=True)

    c7, c8 = st.columns(2)

    with c7:
        hourly = df.groupby('Hour').size().reset_index(name='Count')
        fig_hourly = px.area(
            hourly, x='Hour', y='Count',
            color_discrete_sequence=[UBER_RED],
        )
        fig_hourly.update_traces(fill='tozeroy', line=dict(width=2))
        plotly_layout(fig_hourly, "Bookings by Hour of Day")
        st.plotly_chart(fig_hourly, use_container_width=True)

    with c8:
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily = df.groupby('DayOfWeek').size().reset_index(name='Count')
        daily['Day'] = daily['DayOfWeek'].map(lambda x: day_names[x])
        fig_daily = px.bar(
            daily, x='Day', y='Count',
            color='Count', color_continuous_scale='Reds',
        )
        plotly_layout(fig_daily, "Bookings by Day of Week")
        st.plotly_chart(fig_daily, use_container_width=True)

    # ── Raw Data Explorer ───────────────────────────────────────────────────
    with st.expander("🗃️  Raw Data Explorer", expanded=False):
        st.markdown(f"Showing first 500 rows of `{DATASET_PATH}`")
        st.dataframe(
            df.head(500).style.background_gradient(
                subset=['Driver Ratings', 'Customer Rating', 'Ride Distance'],
                cmap='RdYlGn',
            ),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Multi-Target Predictor
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {UBER_ACCENT}, {UBER_DARK});
                border-radius: 14px; padding: 28px 32px; margin-bottom: 24px;
                border: 1px solid #ffffff10;'>
        <h1 style='margin:0; font-size:2rem;'>🎯 Multi-Target Predictor</h1>
        <p style='color:#aaaaaa; margin: 8px 0 0 0;'>
            Enter ride details to predict cancellation risk, driver rating, and customer rating simultaneously.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    models = load_models()

    # ── Version mismatch: proactive banner (from manifest.json) ───────────
    if "_version_mismatch_warning" in models:
        train_sk, app_sk = models["_version_mismatch_warning"]
        st.warning(
            f"⚠️ **scikit-learn version mismatch detected** — "
            f"models were saved with **`scikit-learn {train_sk}`** but this environment "
            f"is running **`scikit-learn {app_sk}`**. Loading will likely fail with an "
            f"`AttributeError`. Apply **Option A or B** below to fix this."
        )

    # ── Missing model files ────────────────────────────────────────────────
    if "_missing" in models:
        st.error(
            f"⚠️ Some model files are missing: `{', '.join(models['_missing'])}`\n\n"
            "Run **`Modeling_Notebook.ipynb`** top-to-bottom first, then restart the app."
        )

    # ── Pickle / sklearn version error (caught at load time) ──────────────
    if "_version_error" in models:
        app_sk  = models.get("_sklearn_version", "unknown")
        # Try to recover the training version from the manifest warning tuple
        train_sk = models["_version_mismatch_warning"][0] \
            if "_version_mismatch_warning" in models else "1.6.1 (detected from error)"

        affected = [f for f, _ in models["_version_error"]]

        st.error(
            f"### ❌ scikit-learn Version Mismatch\n\n"
            f"The `.pkl` files were serialised with **scikit-learn `{train_sk}`** "
            f"but this Python environment has **`{app_sk}`** installed. "
            f"The internal `ColumnTransformer` class changed between these versions, "
            f"so `joblib` cannot unpickle the preprocessor.\n\n"
            f"---\n\n"
            f"**✅ Option A — Downgrade to match the training environment (fastest)**\n\n"
            f"Run in your terminal:\n"
            f"```bash\n"
            f"pip install scikit-learn=={train_sk}\n"
            f"# Then restart Streamlit:\n"
            f"streamlit run app.py\n"
            f"```\n\n"
            f"**✅ Option B — Upgrade your notebook environment and re-export (recommended)**\n\n"
            f"Run in your terminal:\n"
            f"```bash\n"
            f"# 1. Install the current version in your notebook environment:\n"
            f"pip install scikit-learn=={app_sk}\n\n"
            f"# 2. Re-run all cells in Modeling_Notebook.ipynb top-to-bottom\n"
            f"#    to regenerate all .pkl files with scikit-learn {app_sk}.\n\n"
            f"# 3. Restart this Streamlit app:\n"
            f"streamlit run app.py\n"
            f"```\n\n"
            f"**Affected model files:** `{', '.join(affected)}`"
        )
        for fname, err in models["_version_error"]:
            with st.expander(f"🔍 Raw error — `{fname}`", expanded=False):
                st.code(err, language=None)

    # ── Input Form ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📝 Ride Details</div>",
                unsafe_allow_html=True)

    form_c1, form_c2 = st.columns([1, 1])

    with form_c1:
        vehicle_type  = st.selectbox("🚙 Vehicle Type", VEHICLE_TYPES)
        pickup_loc    = st.selectbox("📍 Pickup Location", sorted(LOCATIONS), index=3)
        drop_loc      = st.selectbox("📌 Drop Location",   sorted(LOCATIONS), index=10)
        ride_distance = st.slider("📏 Ride Distance (km)", 1.0, 50.0, 15.0, 0.5)

    with form_c2:
        booking_value = st.slider("💰 Estimated Booking Value (₹)", 50, 2000, 350, 10)
        avg_vtat      = st.slider("⏱️ Avg VTAT (min) — Vehicle Time to Arrive", 2.0, 20.0, 8.5, 0.5)
        avg_ctat      = st.slider("⏱️ Avg CTAT (min) — Customer Time to Arrive", 10.0, 45.0, 28.0, 0.5)
        ride_time     = st.time_input("🕐 Ride Time", value=datetime.now().time())

    # Feature derivation
    hour       = ride_time.hour
    dow        = datetime.now().weekday()
    month      = datetime.now().month
    is_weekend = int(dow in [5, 6])
    is_rush    = int((7 <= hour <= 10) or (17 <= hour <= 20))

    # ── Predict Button ──────────────────────────────────────────────────────
    predict_clicked = st.button("⚡ Predict All Targets", type="primary", use_container_width=True)

    if predict_clicked:
        models_ready = "_missing" not in models and "_version_error" not in models

        if not models_ready:
            st.error("Cannot predict: some models are missing. Run the notebook first.")
        else:
            # ── Encode locations using fitted LabelEncoders ─────────────────
            le_pickup  = models["le_pickup"]
            le_drop    = models["le_drop"]
            preproc    = models["preproc"]

            def safe_le_transform(le, val):
                """Transform with fallback for unseen categories."""
                classes = list(le.classes_)
                if val in classes:
                    return le.transform([val])[0]
                return 0  # fallback for unknown location

            pickup_enc = safe_le_transform(le_pickup, pickup_loc)
            drop_enc   = safe_le_transform(le_drop,   drop_loc)

            # ── Build feature row ───────────────────────────────────────────
            NUMERICAL_FEATURES_ENC = [
                'Avg VTAT', 'Avg CTAT', 'Ride Distance', 'Booking Value',
                'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsRushHour',
                'Pickup_Enc', 'Drop_Enc',
            ]
            CATEGORICAL_OHE = ['Vehicle Type']

            row_dict = {
                'Avg VTAT':     avg_vtat,
                'Avg CTAT':     avg_ctat,
                'Ride Distance':ride_distance,
                'Booking Value':booking_value,
                'Hour':         hour,
                'DayOfWeek':    dow,
                'Month':        month,
                'IsWeekend':    is_weekend,
                'IsRushHour':   is_rush,
                'Pickup_Enc':   pickup_enc,
                'Drop_Enc':     drop_enc,
                'Vehicle Type': vehicle_type,
            }
            X_pred = pd.DataFrame([row_dict])

            # ── 1. Cancellation ─────────────────────────────────────────────
            try:
                X_num_ohe = X_pred[NUMERICAL_FEATURES_ENC + CATEGORICAL_OHE]
                X_proc    = preproc.transform(X_num_ohe)
                cancel_prob  = models["clf"].predict_proba(X_proc)[0][1]
                cancel_pct   = cancel_prob * 100

                if cancel_prob < 0.30:
                    risk_level = "LOW"
                    alert_cls  = "alert-low"
                    risk_icon  = "🟢"
                elif cancel_prob < 0.60:
                    risk_level = "MEDIUM"
                    alert_cls  = "alert-medium"
                    risk_icon  = "🟡"
                else:
                    risk_level = "HIGH"
                    alert_cls  = "alert-high"
                    risk_icon  = "🔴"
            except Exception as e:
                cancel_pct  = None
                risk_level  = "ERROR"
                alert_cls   = "alert-high"
                risk_icon   = "⚠️"

            # ── 2. Driver Rating ────────────────────────────────────────────
            try:
                dr_pred = float(np.clip(models["dr"].predict(X_pred[NUMERICAL_FEATURES_ENC + CATEGORICAL_OHE])[0], 1.0, 5.0))
            except Exception:
                dr_pred = None

            # ── 3. Customer Rating ──────────────────────────────────────────
            try:
                cr_pred = float(np.clip(models["cr"].predict(X_pred[NUMERICAL_FEATURES_ENC + CATEGORICAL_OHE])[0], 1.0, 5.0))
            except Exception:
                cr_pred = None

            # ── Results Display ─────────────────────────────────────────────
            st.divider()
            st.markdown("<div class='section-header'>🔮 Prediction Results</div>",
                        unsafe_allow_html=True)

            res1, res2, res3 = st.columns(3)

            # Card 1 — Cancellation Risk
            with res1:
                cancel_val = f"{cancel_pct:.1f}%" if cancel_pct is not None else "N/A"
                st.markdown(f"""
                <div class='pred-card'>
                    <p>Cancellation Risk</p>
                    <h2 style='color:{UBER_RED};'>{risk_icon} {cancel_val}</h2>
                    <p style='font-size:1rem; font-weight:600; color:#ddd;'>{risk_level} RISK</p>
                </div>
                """, unsafe_allow_html=True)

            # Card 2 — Driver Rating
            with res2:
                dr_val   = f"{dr_pred:.2f} ⭐" if dr_pred else "N/A"
                dr_stars = int(round(dr_pred)) if dr_pred else 0
                star_str = "⭐" * dr_stars + "☆" * (5 - dr_stars)
                st.markdown(f"""
                <div class='pred-card'>
                    <p>Predicted Driver Rating</p>
                    <h2 style='color:#f4a261;'>{dr_val}</h2>
                    <p style='font-size:1.1rem;'>{star_str}</p>
                </div>
                """, unsafe_allow_html=True)

            # Card 3 — Customer Rating
            with res3:
                cr_val   = f"{cr_pred:.2f} ⭐" if cr_pred else "N/A"
                cr_stars = int(round(cr_pred)) if cr_pred else 0
                star_str_c = "⭐" * cr_stars + "☆" * (5 - cr_stars)
                st.markdown(f"""
                <div class='pred-card'>
                    <p>Predicted Customer Rating</p>
                    <h2 style='color:#2d6a4f;'>{cr_val}</h2>
                    <p style='font-size:1.1rem;'>{star_str_c}</p>
                </div>
                """, unsafe_allow_html=True)

            # ── Detailed Cancellation Alert ─────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            if cancel_pct is not None:
                st.markdown(f"""
                <div class='{alert_cls}'>
                    <b>{risk_icon} Cancellation Alert — {risk_level} RISK ({cancel_pct:.1f}%)</b><br>
                    <span style='color:#ccc; font-size:0.9rem;'>
                    {"✅ Low cancellation probability. This trip profile looks reliable." if risk_level == "LOW" else
                     "⚠️ Moderate risk. Consider confirming pickup time with the driver." if risk_level == "MEDIUM" else
                     "🚨 High cancellation probability. Consider offering surge pricing or pre-selecting a driver."}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # ── Gauge Charts ────────────────────────────────────────────────
            st.divider()
            g1, g2, g3 = st.columns(3)

            def make_gauge(value, max_val, label, color):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge=dict(
                        axis=dict(range=[0, max_val], tickcolor='white'),
                        bar=dict(color=color),
                        bgcolor=UBER_DARK,
                        bordercolor='rgba(255,255,255,0.13)',
                        steps=[
                            dict(range=[0, max_val*0.33], color=UBER_ACCENT),
                            dict(range=[max_val*0.33, max_val*0.66], color="#1e2a3a"),
                            dict(range=[max_val*0.66, max_val], color="#0d1826"),
                        ],
                    ),
                    number=dict(font=dict(color='white', size=32)),
                    title=dict(text=label, font=dict(color='#aaa', size=13)),
                ))
                fig.update_layout(
                    paper_bgcolor=UBER_ACCENT, height=220,
                    margin=dict(l=20, r=20, t=30, b=20),
                    font=dict(color='white'),
                )
                return fig

            with g1:
                if cancel_pct is not None:
                    st.plotly_chart(
                        make_gauge(cancel_pct, 100, "Cancellation Risk (%)", UBER_RED),
                        use_container_width=True
                    )
            with g2:
                if dr_pred:
                    st.plotly_chart(
                        make_gauge(dr_pred, 5, "Predicted Driver Rating", "#f4a261"),
                        use_container_width=True
                    )
            with g3:
                if cr_pred:
                    st.plotly_chart(
                        make_gauge(cr_pred, 5, "Predicted Customer Rating", "#2d6a4f"),
                        use_container_width=True
                    )

            # ── Trip Summary Card ───────────────────────────────────────────
            with st.expander("📋 Trip Summary", expanded=False):
                summary_df = pd.DataFrame([{
                    "Vehicle Type":    vehicle_type,
                    "Pickup Location": pickup_loc,
                    "Drop Location":   drop_loc,
                    "Distance (km)":   ride_distance,
                    "Booking Value":   f"₹{booking_value}",
                    "Hour":            hour,
                    "Day of Week":     ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dow],
                    "Rush Hour":       "Yes" if is_rush else "No",
                    "Weekend":         "Yes" if is_weekend else "No",
                    "Avg VTAT (min)":  avg_vtat,
                    "Avg CTAT (min)":  avg_ctat,
                }]).T.reset_index()
                summary_df.columns = ["Feature", "Value"]
                st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='text-align:center; color:#555; font-size:0.78rem; padding: 30px 0 10px 0;'>
    NCR Ride Bookings AI Dashboard &nbsp;|&nbsp;
    XGBoost Models &nbsp;|&nbsp;
    Streamlit + Plotly &nbsp;|&nbsp;
    Data: 150,000 bookings
</div>
""", unsafe_allow_html=True)
