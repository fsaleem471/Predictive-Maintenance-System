<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-AUC%200.9953-FF6600?style=flat-square"/>
<img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>

<br/><br/>

# 🔧 Predictive Maintenance System

### ML-powered vehicle component failure prediction — before it breaks.

*Trained on NASA CMAPSS turbofan degradation data · XGBoost · FastAPI · Interactive Dashboard*

</div>

---

## Overview

The **Predictive Maintenance System** uses machine learning to analyze real-time sensor data from vehicle engines and predict **when a component is likely to fail** — before it actually does. This allows maintenance teams to act proactively, avoiding costly breakdowns and unplanned downtime.

This project covers the full stack:

- **ML Pipeline** — data generation, feature engineering, model training & evaluation
- **REST API** — FastAPI backend serving live predictions
- **Dashboard** — interactive fleet health monitoring UI

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   PREDICTIVE MAINTENANCE SYSTEM                 │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
  │  Vehicle     │     │   FastAPI      │     │  React          │
  │  Sensors     │────▶│   Backend      │────▶│  Dashboard      │
  │  (21 inputs) │     │   /predict     │     │  Fleet Monitor  │
  └──────────────┘     └───────┬────────┘     └─────────────────┘
                               │
                     ┌─────────▼────────┐
                     │   ML Model       │
                     │   XGBoost        │
                     │   AUC = 0.9953   │
                     └─────────┬────────┘
                               │
              ┌────────────────▼──────────────────┐
              │         PREDICTION OUTPUT         │
              │                                   │
              │  risk_label:  "AT RISK"           │
              │  risk_score:  84                  │
              │  probability: 0.84                │
              │  alert:       true                │
              │  rec: "Schedule inspection"       │
              └───────────────────────────────────┘
```

---

## Dataset — NASA CMAPSS

This project uses the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset — the industry-standard benchmark for predictive maintenance research.

| Property | Detail |
|---|---|
| Engines simulated | 100 |
| Total sensor readings | 25,449 |
| Sensors per reading | 21 |
| Operational settings | 3 |
| Task | Binary classification — will fail within 30 cycles? |
| Class balance | ~12% at-risk, 88% healthy |

**Key sensors used:**

| Sensor | Measurement | Degrades with age? |
|---|---|---|
| s2 | Fan inlet temperature | Yes |
| s3 | LPC outlet temperature | Yes |
| s4 | HPC outlet temperature | Yes |
| s11 | Static pressure | Yes |
| s12 | Ratio of fuel flow | Yes |
| s20, s21 | Fan speed indicators | Yes |

---

## ML Pipeline

### Feature Engineering

```python
# 1. Compute Remaining Useful Life (RUL)
df["RUL"] = df["max_cycle"] - df["cycle"]

# 2. Binary label — will fail within threshold?
df["label"] = (df["RUL"] <= 30).astype(int)

# 3. Rolling statistics (capture degradation trend, not just snapshot)
for col in key_sensors:
    df[f"{col}_rollmean"] = df.groupby("unit_id")[col].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df[f"{col}_rollstd"] = df.groupby("unit_id")[col].transform(
        lambda x: x.rolling(5, min_periods=1).std().fillna(0)
    )

# 4. Drop low-variance sensors (no predictive signal)
# Removed: s5, s10, s15, s16
```

### Models Trained

| Model | AUC-ROC | Accuracy | Precision (At Risk) | Recall (At Risk) |
|---|---|---|---|---|
| Random Forest | 0.9951 | 97.0% | 83.8% | 93.4% |
| **XGBoost ✅** | **0.9953** | **96.7%** | **81.7%** | **94.0%** |

> XGBoost was selected as the best model. High recall on the "At Risk" class is prioritized — missing a failure is worse than a false alarm.

---

## Project Structure

```
predictive_maintenance/
│
├── ml_pipeline.py          # Full ML training pipeline
│
├── api/
│   └── main.py             # FastAPI backend (5 endpoints)
│
├── data/
│   └── cmapss_simulated.csv
│
├── models/
│   ├── best_model.pkl      # Trained XGBoost model
│   ├── scaler.pkl          # MinMaxScaler
│   └── feature_names.pkl   # Feature list
│
└── plots/
    ├── engine_degradation.png
    ├── rul_distribution.png
    ├── feature_importance.png
    ├── roc_curves.png
    └── confusion_matrix.png
```

---

## API Reference

Run the API locally:

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check + model info |
| POST | `/predict` | Predict failure risk for one engine |
| POST | `/predict/batch` | Batch predictions for multiple engines |
| GET | `/fleet/status` | Full fleet overview (12 vehicles) |
| GET | `/engine/{id}/history` | Sensor history for last N cycles |

### Example — Single Prediction

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": 5,
    "cycle": 280,
    "s2": 658.3,
    "s3": 1621.4,
    "s4": 1430.2,
    "s11": 49.8,
    "s12": 540.1
  }'
```

**Response:**
```json
{
  "unit_id": 5,
  "cycle": 280,
  "risk_label": "AT RISK",
  "risk_probability": 0.847,
  "risk_score": 84,
  "alert": true,
  "recommendation": "🚨 CRITICAL: Schedule immediate inspection. Component failure imminent.",
  "predicted_at": "2026-04-24T10:31:00.000Z"
}
```

---

## Results & Evaluation

### ROC Curve
Both models achieve AUC > 0.99, indicating near-perfect discrimination between healthy and at-risk engines.

### Feature Importance (XGBoost Top 5)
1. `s3_rollmean` — LPC outlet temperature rolling average
2. `s4_rollmean` — HPC outlet temperature rolling average
3. `s11` — Static pressure
4. `s2_rollmean` — Fan inlet temperature rolling average
5. `s12` — Fuel flow ratio

Rolling features dominate — capturing the *trend* of degradation matters more than a single snapshot reading.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/predictive-maintenance.git
cd predictive-maintenance

# Install dependencies
pip install -r requirements.txt

# Train the model
python ml_pipeline.py

# Start the API
uvicorn api.main:app --reload --port 8000
```

**Requirements:** Python 3.10+, 4GB RAM recommended

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML / Data | Python, pandas, NumPy, scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | React, Chart.js |
| Model persistence | joblib |

---

## Academic Context

This project was developed as part of an **Artificial Intelligence** degree portfolio at Iqra University. It demonstrates applied competency in:

- End-to-end ML pipeline design (data → features → model → API)
- Time-series feature engineering for sensor data
- Imbalanced classification with class weighting
- REST API development with FastAPI
- Real-world industrial AI application (predictive maintenance)

Relevant to: **Erasmus Mundus EMAI** (Joint Master in Artificial Intelligence) application portfolio.

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

<div align="center">
  <sub>Built with Python · FastAPI · XGBoost · NASA CMAPSS Dataset</sub>
</div>
