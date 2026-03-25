# 🚗 NCR Ride Bookings — Multi-Target ML Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)

**A production-grade machine learning pipeline and interactive dashboard that simultaneously predicts ride cancellations, driver ratings, and customer ratings for NCR (National Capital Region) ride-hailing data.**

[Features](#-features) • [App Preview](#️-app-preview) • [Installation](#️-installation) • [Usage](#-usage) • [Models](#-model-architecture) • [Project Structure](#-project-structure)

</div>

---

## 📌 Project Overview

This end-to-end data science project analyses **150,000 ride bookings** from the NCR (Delhi-NCR) region and builds three simultaneous ML predictions from a single user input:

| # | Target | Type | Algorithm |
|---|--------|------|-----------|
| 1 | **Booking Cancellation Risk** | Binary Classification | XGBoost + Random Forest |
| 2 | **Driver Rating** | Regression (1–5 ★) | XGBoost |
| 3 | **Customer Rating** | Regression (1–5 ★) | XGBoost |

---

## ✨ Features

### 🔬 Modeling Notebook
- Detailed **Target Column Analysis** — distributions, correlations, and cross-variable plots
- **Feature Engineering** — hour, day-of-week, rush-hour flags extracted from timestamps
- **Dual preprocessing** — median imputation for numerics, One-Hot Encoding for vehicle type, Label Encoding for 176 unique locations
- **Model comparison** — Random Forest vs XGBoost for classification with AUC-ROC evaluation
- **Regression evaluation** — MAE, RMSE, R² with 5-fold cross-validation
- Full **model serialization** via `joblib` with a `version_manifest.json` for environment tracking

### 📊 Analysis Dashboard (Streamlit)
- Live KPI metrics: total bookings, completion rate, cancellation rate, average ratings
- Interactive Plotly charts: booking status distribution, overlaid rating histograms, box plots
- Revenue and average ratings broken down by vehicle type (dual-axis chart)
- Hourly and day-of-week booking pattern heatmaps
- Raw data explorer with conditional color formatting

### 🎯 Multi-Target Predictor (Streamlit)
- Single input form → three simultaneous predictions
- **Cancellation Risk**: probability percentage + color-coded alert (🟢 Low / 🟡 Medium / 🔴 High)
- **Driver Rating**: predicted star rating with visual star display
- **Customer Rating**: predicted star rating with visual star display
- Animated gauge charts for all three predictions
- Trip summary breakdown with all engineered features visible

---

## 🖼️ App Preview

<div align="center">
  <img src="Dasboard.gif" alt="NCR Ride Bookings Dashboard Demo" width="100%"/>
</div>

---

## 📁 Project Structure

```
Ride_Bookings/
│
├── 📓 Modeling_Notebook.ipynb        # ML pipeline — EDA → training → serialization
├── 📓 NCR_Ride_Bookings_EDA.ipynb    # Standalone Exploratory Data Analysis notebook
├── 🖥️  app.py                         # Streamlit dashboard application
├── 📋 requirements.txt               # Pinned Python dependencies
├── 📊 ncr_ride_bookings.csv          # Raw dataset (150,000 rows × 21 cols)
├── 📊 Uber.pbix                      # Power BI report file
├── 🎞️  Dasboard.gif                   # Animated app preview
├── 📄 README.md                      # This file
│
└── 📦 saved_models/                  # Generated after running the notebook
    ├── xgb_cancellation_model.pkl
    ├── rf_cancellation_pipeline.pkl
    ├── xgb_driver_rating_pipeline.pkl
    ├── xgb_customer_rating_pipeline.pkl
    ├── preprocessor.pkl
    ├── le_pickup.pkl
    ├── le_drop.pkl
    └── version_manifest.json         # sklearn/xgboost version lock file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda

### 1. Clone the repository
```bash
git clone https://github.com/Mohamed-Tamer-Ai/Ride_Bookings.git
cd Ride_Bookings
```

### 2. Create a virtual environment (recommended)
```bash
# Using venv
python -m venv ride_env
ride_env\Scripts\activate        # Windows
source ride_env/bin/activate     # macOS / Linux

# Or using conda
conda create -n ride_env python=3.11
conda activate ride_env
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ `scikit-learn` is **pinned to `1.7.1`** in `requirements.txt`. This must match the version used to train the models. Do not upgrade it without re-running the notebook.

### 4. Place the dataset
Ensure `ncr_ride_bookings.csv` is in the **project root directory** (same level as `app.py`).

---

## 🚀 Usage

### Step 1 — Train the models

Open and run the notebook (Restart & Run All):
```bash
jupyter notebook Modeling_Notebook.ipynb
```
Or execute headlessly:
```bash
jupyter nbconvert --to notebook --execute --inplace Modeling_Notebook.ipynb
```
This will create the `saved_models/` directory with all `.pkl` files.

### Step 2 — Launch the Streamlit app
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501` in your browser.

---

## 🧠 Model Architecture

### Target 1 — Cancellation Classifier

```
Raw Features (21 cols)
        │
        ▼
Feature Engineering
(Hour, DayOfWeek, IsRushHour, IsWeekend)
        │
        ▼
ColumnTransformer
├── SimpleImputer(median)  →  Numerical features
└── OneHotEncoder          →  Vehicle Type
        │
        ▼
XGBoostClassifier
(n_estimators=300, scale_pos_weight for class imbalance)
        │
        ▼
P(Cancellation)  →  Risk Level (Low / Medium / High)
```

### Targets 2 & 3 — Rating Regressors

```
Completed Rides Only
        │
        ▼
Same Preprocessing Pipeline
        │
        ▼
XGBoostRegressor
(n_estimators=300, max_depth=6, lr=0.05)
        │
        ▼
Rating (clipped to [1.0 – 5.0])
```

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| Rows | 150,000 |
| Columns | 21 |
| Date Range | 2024 |
| Region | Delhi-NCR, India |
| Unique Locations | 176 |
| Vehicle Types | Auto, Bike, eBike, Go Mini, Go Sedan, Premier Sedan, Uber XL |
| Completion Rate | ~62% |
| Cancellation Rate | ~25% |

**Key columns:**

| Column | Description |
|--------|-------------|
| `Booking Status` | Completed / Cancelled by Driver / Cancelled by Customer / No Driver Found / Incomplete |
| `Vehicle Type` | 7 vehicle categories |
| `Avg VTAT` | Average Vehicle Time to Arrive (minutes) |
| `Avg CTAT` | Average Customer Time to Arrive (minutes) |
| `Ride Distance` | Trip distance in km (1–50 km) |
| `Booking Value` | Fare in INR (₹50–₹4,277) |
| `Driver Ratings` | 3.0–5.0 stars (completed rides only) |
| `Customer Rating` | 3.0–5.0 stars (completed rides only) |

---

## 📈 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| XGBoost Cancellation Classifier | AUC-ROC | ~0.97 |
| Random Forest Cancellation Classifier | AUC-ROC | ~0.96 |
| XGBoost Driver Rating Regressor | MAE | ~0.15 |
| XGBoost Driver Rating Regressor | R² | ~0.88 |
| XGBoost Customer Rating Regressor | MAE | ~0.14 |
| XGBoost Customer Rating Regressor | R² | ~0.89 |

> Scores are approximate and depend on your random seed and environment. Run the notebook to see exact values.

---

## 🛠️ Requirements

All dependencies are listed in [`requirements.txt`](requirements.txt). Key packages:

```
# Core data science
pandas>=2.0.0
numpy>=1.24.0

# Machine learning — PINNED
scikit-learn==1.7.1
xgboost>=2.0.0

# Model serialization
joblib>=1.2.0

# Dashboard & visualization
streamlit>=1.30.0
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Notebook environment
jupyter>=1.0.0
nbconvert>=7.0.0
ipykernel>=6.0.0
```

> ⚠️ `scikit-learn==1.7.1` is intentionally **pinned**. All `.pkl` files in `saved_models/` were serialized with this version. Installing a different version will cause an `AttributeError` when loading models. Re-run the notebook to regenerate models if you change versions.

---

## 🔧 Troubleshooting

### `AttributeError: _RemainderColsList`
Your Streamlit environment has a **different scikit-learn version** than the one used to train the models. The app will display a banner telling you exactly which versions are mismatched.

```bash
# Fix: align to the training version, then re-run the notebook
pip install scikit-learn==1.7.1
jupyter nbconvert --to notebook --execute --inplace Modeling_Notebook.ipynb
streamlit run app.py
```

### `ValueError: Invalid gridcolor '#ffffffXX'`
Plotly does not accept 8-digit hex (CSS RGBA). This is already fixed in the current `app.py` — ensure you are using the latest version from this repo.

### `FileNotFoundError: ncr_ride_bookings.csv`
Place the CSV file in the **same directory** as `app.py` before running.

### Models missing on first run
The `saved_models/` folder is generated by the notebook. Always run `Modeling_Notebook.ipynb` fully before launching `app.py`.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Mohamed Tamer**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohamed-tamer-b59329347/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/Mohamed-Tamer-Ai)


<br>

**Yousif Hisham**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yousif-hisham-98baa2292/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/YousifHisham-tech)

**Sara Muhammed**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sara-muhammad-385511347/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/sara-abdulrahman)


**Jana Mostafa**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jana-mostafa-2b2a5634b/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/janamostafa-tech)

<div align="center">


  <sub>Built with ❤️ using Python, XGBoost, and Streamlit</sub>
</div>
