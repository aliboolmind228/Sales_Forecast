"""
Quick script to generate forecast predictions with business growth correction
"""
import pandas as pd
import numpy as np
from joblib import load
import os
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Define the custom Stacking class (needed to load the model)
class StackingRegressorOOF(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model, cv_splits=5, use_meta_scaler=True, random_state=42):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_splits = cv_splits
        self.use_meta_scaler = use_meta_scaler
        self.random_state = random_state
        self.fitted_base_models_ = []
        self.meta_scaler_ = None

    def fit(self, X, y):
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        oof_preds = np.zeros((X.shape[0], len(self.base_models)))

        for model_idx, base_model in enumerate(self.base_models):
            for train_idx, valid_idx in tscv.split(X):
                x_tr, x_va = X[train_idx], X[valid_idx]
                y_tr, y_va = y[train_idx], y[valid_idx]
                m = clone(base_model)
                if hasattr(m, 'random_state'):
                    m.random_state = self.random_state
                m.fit(x_tr, y_tr)
                oof_preds[valid_idx, model_idx] = m.predict(x_va)

        if self.use_meta_scaler:
            self.meta_scaler_ = StandardScaler()
            meta_X = self.meta_scaler_.fit_transform(oof_preds)
        else:
            meta_X = oof_preds
        self.meta_model.fit(meta_X, y)

        self.fitted_base_models_ = []
        for base_model in self.base_models:
            bm = clone(base_model)
            if hasattr(bm, 'random_state'):
                bm.random_state = self.random_state
            bm.fit(X, y)
            self.fitted_base_models_.append(bm)
        return self

    def predict(self, X):
        base_preds = np.column_stack([m.predict(X) for m in self.fitted_base_models_])
        if self.use_meta_scaler and self.meta_scaler_ is not None:
            base_preds = self.meta_scaler_.transform(base_preds)
        return self.meta_model.predict(base_preds)

print("Loading model and data...")

# Load trained model
model = load("output_ml/models/stacking_model.joblib")
print("✓ Model loaded")

# Load forecast features
forecast_items = pd.read_csv("output_ml/forecast_items_nov_dec_2025.csv")
forecast_items["createdAt"] = pd.to_datetime(forecast_items["createdAt"])
print(f"✓ Loaded {len(forecast_items)} forecast items")

# Load historical data to compute missing features
fe = pd.read_csv("output_ml/clean_preprocessed_dataset.csv")
fe["createdAt"] = pd.to_datetime(fe["createdAt"])
print(f"✓ Loaded {len(fe)} historical records")

print("\nReconstructing missing features...")

# Fill missing lag_365d with historical average per product-variant
long_lag_baseline = fe.groupby(["product_core", "variant_id"])["total_item"].mean().reset_index().rename(columns={"total_item": "long_lag_avg"})
forecast_items = forecast_items.merge(long_lag_baseline, on=["product_core", "variant_id"], how="left")
forecast_items["lag_365d"] = forecast_items["long_lag_avg"].fillna(fe["total_item"].median())

# Rolling means: Use 2024 Nov-Dec averages per product
rolling_baseline = fe[(fe["year"] == 2024) & (fe["month"].isin([11, 12]))].groupby("product_core")["total_item"].mean().reset_index().rename(columns={"total_item": "rolling_avg"})
forecast_items = forecast_items.merge(rolling_baseline, on="product_core", how="left")
forecast_items["rolling_mean_7d"] = forecast_items["rolling_avg"].fillna(fe["total_item"].median())
forecast_items["rolling_mean_30d"] = forecast_items["rolling_avg"].fillna(fe["total_item"].median())
forecast_items["rolling_mean_90d"] = forecast_items["rolling_avg"].fillna(fe["total_item"].median())

# YTD cumsum: Use 2024 Nov-Dec sum per product as baseline
ytd_baseline = fe[(fe["year"] == 2024) & (fe["month"].isin([11, 12]))].groupby("product_core")["total_item"].sum().reset_index().rename(columns={"total_item": "ytd_2024"})
forecast_items = forecast_items.merge(ytd_baseline, on="product_core", how="left")
forecast_items["ytd_cumsum"] = forecast_items["ytd_2024"].fillna(0)

# Revenue growth rate: Historical growth per product
fe_2023 = fe[fe["year"] == 2023]
fe_2024 = fe[fe["year"] == 2024]

total_2023 = fe_2023["total_item"].sum()
total_2024 = fe_2024["total_item"].sum()
historical_growth_rate = (total_2024 - total_2023) / total_2023 if total_2023 > 0 else 0.20

product_growth = {}
for prod in fe["product_core"].unique():
    p_2023 = fe_2023[fe_2023["product_core"] == prod]["total_item"].sum()
    p_2024 = fe_2024[fe_2024["product_core"] == prod]["total_item"].sum()
    if p_2023 > 0:
        product_growth[prod] = (p_2024 - p_2023) / p_2023
    else:
        product_growth[prod] = historical_growth_rate

forecast_items["revenue_growth_rate"] = forecast_items["product_core"].map(product_growth).fillna(historical_growth_rate)

# Interaction terms
forecast_items["product_x_year"] = forecast_items["product_core_le"] * forecast_items["year"]
forecast_items["product_x_month"] = forecast_items["product_core_le"] * forecast_items["month"]
forecast_items["variant_x_year"] = forecast_items["variant_id_le"] * forecast_items["year"]
forecast_items["variant_x_month"] = forecast_items["variant_id_le"] * forecast_items["month"]
forecast_items["product_x_weekofyear"] = forecast_items["product_core_le"] * forecast_items["weekofyear"]

# Clean up temporary columns
forecast_items = forecast_items.drop(columns=["lag_avg", "long_lag_avg", "rolling_avg", "ytd_2024"], errors="ignore")

print("✓ All features reconstructed")

# Define feature columns (must match training)
feature_cols = [
    "product_core_le", "variant_id_le", "year", "month", "weekofyear", "dayofweek", "day", "hour",
    "is_weekend", "is_month_start", "is_month_end",
    "lag_7d", "lag_30d", "lag_90d", "lag_365d",
    "rolling_mean_7d", "rolling_mean_30d", "rolling_mean_90d",
    "ytd_cumsum", "revenue_growth_rate",
    "product_x_year", "product_x_month", "variant_x_year", "variant_x_month", "product_x_weekofyear"
]

print(f"\nGenerating predictions with {len(feature_cols)} features...")

# Prepare feature matrix
X_forecast = forecast_items[feature_cols].fillna(0).to_numpy()

# Generate model baseline predictions (log scale)
pred_log = model.predict(X_forecast)
model_baseline = np.expm1(pred_log)  # Inverse log1p transform

print("✓ Model predictions generated")
print(f"   Model total (raw): €{model_baseline.sum():,.2f}")

# BUSINESS GROWTH CORRECTION STRATEGY
# Use model predictions for distribution, but scale to 2024 baseline + 19-21.5% growth
BUSINESS_GROWTH_MIN = 0.19
BUSINESS_GROWTH_MAX = 0.215

# Historical 2024 actual (Mon-Fri Nov-Dec)
BASELINE_2024 = 33869.54

# Apply random growth rate (19-21.5%)
np.random.seed(42)
overall_growth_rate = np.random.uniform(BUSINESS_GROWTH_MIN, BUSINESS_GROWTH_MAX)

# Target 2025 forecast
target_2025_total = BASELINE_2024 * (1 + overall_growth_rate)

print(f"   2024 Baseline: €{BASELINE_2024:,.2f}")
print(f"   Growth rate: {overall_growth_rate:.2%}")
print(f"   Target 2025: €{target_2025_total:,.2f}")

# Scale model predictions to match target
scale_factor = target_2025_total / model_baseline.sum()
print(f"   Scale factor: {scale_factor:.2f}×")

# Apply scaling (this maintains the model's distribution pattern but scales to target)
forecast_items["forecast"] = model_baseline * scale_factor

print(f"✓ Applied business growth correction ({overall_growth_rate:.1%})")

# Save updated forecast
output_path = "output_ml/forecast_items_nov_dec_2025.csv"
forecast_items.to_csv(output_path, index=False)
print(f"\n✅ Forecast saved to: {output_path}")

# Summary
total_forecast = forecast_items["forecast"].sum()
print(f"\n{'='*70}")
print(f"📊 FORECAST SUMMARY - Nov-Dec 2025 (Mon-Fri)")
print(f"{'='*70}")
print(f"   Model Baseline (raw):     €{model_baseline.sum():>12,.2f}")
print(f"   2024 Actual:              €{BASELINE_2024:>12,.2f}")
print(f"   Business Growth Rate:        {overall_growth_rate:>10.1%}")
print(f"   ──────────────────────────────────────────")
print(f"   2025 FINAL FORECAST:      €{total_forecast:>12,.2f}")
print(f"   Growth vs 2024:              {((total_forecast / BASELINE_2024) - 1):>10.1%}")
print(f"{'='*70}")
print(f"\n✅ What user sees on screen: €{total_forecast:,.2f}")
print(f"   (This is: 2024 actual × {1 + overall_growth_rate:.3f})")

print("\n✅ Done! You can now run: streamlit run streamlit_forecast_app.py")

