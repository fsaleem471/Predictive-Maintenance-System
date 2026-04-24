"""
=============================================================
  Predictive Maintenance System - ML Pipeline
  Dataset: NASA CMAPSS (Turbofan Engine Degradation Simulation)
  Author: Predictive Maintenance Project
=============================================================

CMAPSS Dataset Columns:
  - unit_id        : Engine unit number
  - cycle          : Operational cycle (time)
  - op_setting_1/2/3 : Operational settings
  - sensor_1..21   : 21 sensor measurements (temp, pressure, speed, etc.)

Goal:
  Predict Remaining Useful Life (RUL) of an engine → 
  Classify if engine will fail within threshold (binary alert).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR  = os.path.join(BASE_DIR, "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  STEP 1: GENERATE / LOAD DATA
# ─────────────────────────────────────────────
def generate_cmapss_data(n_engines=100, seed=42):
    """
    Simulate NASA CMAPSS-style turbofan engine degradation data.
    Each engine runs for a random number of cycles until failure.
    Sensors degrade gradually as the engine approaches failure.
    """
    np.random.seed(seed)
    records = []

    SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

    for unit_id in range(1, n_engines + 1):
        max_cycles = np.random.randint(150, 350)   # engine lifetime varies

        for cycle in range(1, max_cycles + 1):
            degradation = cycle / max_cycles        # 0 → 1 as engine ages

            op1 = np.random.choice([0, 10, 20, 25, 35, 42])
            op2 = np.round(np.random.uniform(0.6, 0.62), 4)
            op3 = np.random.choice([60, 80, 100])

            # 21 sensors — some degrade, some are noise, some are informative
            sensors = []
            base_values = [
                518.67, 641.82, 1589.7, 1400.6, 14.62, 21.61,
                553.9,  2388.0, 9046.2, 1.30,   47.47, 521.66,
                2388.0, 8138.6, 8.4195, 0.03,   392.0, 2388.0,
                100.0,  38.86,  23.42
            ]
            # Sensors that show degradation (realistic subset)
            degrading = {2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21}

            for i, base in enumerate(base_values, start=1):
                noise = np.random.normal(0, base * 0.005)
                if i in degrading:
                    trend = base * 0.08 * degradation
                    value = base + trend + noise
                else:
                    value = base + noise
                sensors.append(round(value, 4))

            row = [unit_id, cycle, op1, op2, op3] + sensors
            records.append(row)

    cols = ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + SENSOR_COLS
    df = pd.DataFrame(records, columns=cols)
    return df


def load_data():
    csv_path = os.path.join(DATA_DIR, "cmapss_simulated.csv")
    if os.path.exists(csv_path):
        print("✔ Loading existing dataset...")
        return pd.read_csv(csv_path)
    print("⚙  Generating CMAPSS-style dataset (100 engines)...")
    df = generate_cmapss_data(n_engines=100)
    df.to_csv(csv_path, index=False)
    print(f"✔ Dataset saved → {csv_path}")
    return df


# ─────────────────────────────────────────────
#  STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────
def compute_rul(df):
    """Compute Remaining Useful Life for each row."""
    max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycles, on="unit_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def add_rolling_features(df, window=5):
    """Add rolling mean & std for key sensors (captures trend, not just snapshot)."""
    key_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_11",
                   "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21"]
    df = df.sort_values(["unit_id", "cycle"])
    for col in key_sensors:
        df[f"{col}_rollmean"] = (
            df.groupby("unit_id")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"{col}_rollstd"] = (
            df.groupby("unit_id")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )
    return df


def create_binary_label(df, threshold=30):
    """
    Binary classification target:
      1 = engine will fail within `threshold` cycles (HIGH RISK → alert!)
      0 = engine is healthy
    """
    df["label"] = (df["RUL"] <= threshold).astype(int)
    return df


def drop_low_variance_sensors(df):
    """Drop sensors with near-zero variance (carry no predictive info)."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_") and "roll" not in c]
    variances = df[sensor_cols].var()
    drop_cols = variances[variances < 0.01].index.tolist()
    print(f"  Dropping {len(drop_cols)} low-variance sensors: {drop_cols}")
    return df.drop(columns=drop_cols, errors='ignore')


# ─────────────────────────────────────────────
#  STEP 3: VISUALIZATION
# ─────────────────────────────────────────────
def plot_engine_degradation(df):
    """Show sensor degradation over time for a sample engine."""
    sample_engine = df[df["unit_id"] == 1].copy()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle("Engine #1 — Sensor Degradation Over Lifecycle", fontsize=14, fontweight='bold')

    axes[0].plot(sample_engine["cycle"], sample_engine["sensor_2"], color="#E74C3C", linewidth=1.5)
    axes[0].set_title("Sensor 2 (Fan Inlet Temperature)")
    axes[0].set_ylabel("Value")

    axes[1].plot(sample_engine["cycle"], sample_engine["sensor_11"], color="#3498DB", linewidth=1.5)
    axes[1].set_title("Sensor 11 (Static Pressure)")
    axes[1].set_ylabel("Value")

    axes[2].plot(sample_engine["cycle"], sample_engine["RUL"], color="#2ECC71", linewidth=1.5)
    axes[2].axhline(y=30, color='red', linestyle='--', label='Failure Threshold (30 cycles)')
    axes[2].set_title("Remaining Useful Life (RUL)")
    axes[2].set_ylabel("RUL (cycles)")
    axes[2].set_xlabel("Cycle")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "engine_degradation.png"), dpi=150)
    plt.close()
    print("  ✔ Plot saved: engine_degradation.png")


def plot_rul_distribution(df):
    """Distribution of RUL values in the dataset."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Dataset Overview", fontsize=13, fontweight='bold')

    ax1.hist(df["RUL"], bins=40, color="#3498DB", edgecolor='white', alpha=0.8)
    ax1.set_title("RUL Distribution")
    ax1.set_xlabel("Remaining Useful Life (cycles)")
    ax1.set_ylabel("Count")

    label_counts = df["label"].value_counts()
    ax2.bar(["Healthy (0)", "At Risk (1)"], label_counts.values,
            color=["#2ECC71", "#E74C3C"], edgecolor='white', alpha=0.85)
    ax2.set_title("Class Distribution")
    ax2.set_ylabel("Count")
    for i, v in enumerate(label_counts.values):
        ax2.text(i, v + 50, str(v), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rul_distribution.png"), dpi=150)
    plt.close()
    print("  ✔ Plot saved: rul_distribution.png")


def plot_feature_importance(model, feature_names, model_name="XGBoost"):
    """Top 20 most important features."""
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    feat_df = feat_df.sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_df)))[::-1]
    bars = ax.barh(feat_df["feature"][::-1], feat_df["importance"][::-1], color=colors[::-1])
    ax.set_title(f"{model_name} — Top 20 Feature Importances", fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("  ✔ Plot saved: feature_importance.png")


def plot_roc_curves(results, X_test, y_test):
    """ROC curves for all trained models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    for (name, model), color in zip(results.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "roc_curves.png"), dpi=150)
    plt.close()
    print("  ✔ Plot saved: roc_curves.png")


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """Confusion matrix for best model."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "At Risk"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  ✔ Plot saved: confusion_matrix.png")


# ─────────────────────────────────────────────
#  STEP 4: TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test, feature_names):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=5,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric='logloss', random_state=42, verbosity=0
        ),
    }

    results = {}
    print("\n" + "="*55)
    print("  MODEL TRAINING & EVALUATION")
    print("="*55)

    for name, model in models.items():
        print(f"\n▶ Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        print(f"  AUC-ROC : {auc:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Healthy", "At Risk"],
                                    digits=3))
        results[name] = model

    return results


# ─────────────────────────────────────────────
#  STEP 5: SAVE BEST MODEL + SCALER
# ─────────────────────────────────────────────
def save_best_model(results, X_test, y_test, scaler, feature_names):
    """Pick model with best AUC and save it."""
    best_name, best_auc = None, 0
    for name, model in results.items():
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        if auc > best_auc:
            best_auc, best_name = auc, name

    best_model = results[best_name]
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    print(f"\n✔ Best Model: {best_name} (AUC = {best_auc:.4f})")
    print(f"  Saved → models/best_model.pkl")
    return best_name, best_model


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  PREDICTIVE MAINTENANCE SYSTEM — ML PIPELINE")
    print("="*55)

    # 1. Load data
    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"  Engines: {df['unit_id'].nunique()} | Total rows: {len(df):,}")

    # 2. Feature engineering
    print("\n[2/6] Feature Engineering...")
    df = compute_rul(df)
    df = add_rolling_features(df, window=5)
    df = create_binary_label(df, threshold=30)
    df = drop_low_variance_sensors(df)
    print(f"  Final features: {df.shape[1]} columns")

    # 3. Visualize data
    print("\n[3/6] Generating data visualizations...")
    plot_engine_degradation(df)
    plot_rul_distribution(df)

    # 4. Prepare features
    print("\n[4/6] Preparing train/test split...")
    drop_cols = ["unit_id", "RUL", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"  At-Risk ratio (test): {y_test.mean():.2%}")

    # 5. Train models
    print("\n[5/6] Training models...")
    results = train_models(X_train, X_test, y_train, y_test, feature_cols)

    # 6. Save + plot
    print("\n[6/6] Saving best model & generating plots...")
    best_name, best_model = save_best_model(results, X_test, y_test, scaler, feature_cols)
    plot_feature_importance(best_model, feature_cols, best_name)
    plot_roc_curves(results, X_test, y_test)
    plot_confusion_matrix(best_model, X_test, y_test, best_name)

    print("\n" + "="*55)
    print("  ✅ PIPELINE COMPLETE")
    print(f"  Models saved  : models/")
    print(f"  Plots saved   : plots/")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
