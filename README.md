# Sales Forecast Dashboard

A hybrid machine learning forecasting system for sales prediction using XGBoost, LightGBM, and ensemble methods with Streamlit visualization.

## 📊 Overview

This project implements an advanced sales forecasting pipeline for bundled products, combining multiple ML models with business growth adjustments to predict November-December 2025 revenue.

## 🚀 Features

- **Hybrid ML Model**: Stacking ensemble with XGBoost, LightGBM, and RandomForest
- **Advanced Feature Engineering**: Lag features, rolling means, YTD cumulative, interaction terms
- **Business Growth Layer**: Automatic 19-21.5% growth correction based on historical trends
- **Interactive Dashboard**: Streamlit UI with product/variant filtering and date selection
- **Time Series Analysis**: Year-over-year comparison (2023 vs 2024 vs 2025)
- **Color-blind Friendly**: Accessible UI using Okabe-Ito color palette

## 📁 Project Structure

```
Sales_Forecast/
├── ML-Fore.ipynb              # Main training notebook
├── streamlit_forecast_app.py  # Interactive dashboard
├── generate_forecast_predictions.py  # Batch prediction script
├── output_ml/                 # Model artifacts and forecasts
│   ├── stacking_model.pkl
│   ├── forecast_items_nov_dec_2025.csv
│   └── validation_metrics.json
├── Final_product_2023_sp.csv  # Historical 2023 data
└── Final_product_2024_fp.csv  # Historical 2024 data
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/aliboolmind228/Sales_Forecast.git
cd Sales_Forecast

# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm optuna streamlit plotly joblib
```

## 📈 Usage

### 1. Train the Model

Run the Jupyter notebook to train the forecasting model:

```bash
jupyter notebook ML-Fore.ipynb
```

This will:
- Load and preprocess 2023-2024 historical data
- Engineer 25+ advanced features
- Train hybrid ensemble model
- Save artifacts to `output_ml/`

### 2. Generate Forecasts

```bash
python generate_forecast_predictions.py
```

### 3. Launch Dashboard

```bash
streamlit run streamlit_forecast_app.py
```

Access the dashboard at `http://localhost:8501`

## 🎯 Model Performance

- **RMSE**: 0.409
- **MAE**: 0.316
- **R²**: 0.163
- **MAPE**: 13.2%

## 📊 Results

- **2023 Revenue**: €29,462.42 (Mon-Fri, Nov-Dec)
- **2024 Revenue**: €33,869.54 (+15.0%)
- **2025 Forecast**: €40,621.89 (+19.9%)

## 🧪 Key Features

### Feature Engineering
- **Temporal**: Year, month, weekofyear, dayofweek, is_weekend
- **Lag Features**: 7d, 30d, 90d, 365d
- **Rolling Means**: 7d, 30d, 90d
- **Interactions**: product×year, product×month, variant×month
- **Growth Rate**: YoY revenue change

### Model Architecture
1. **Base Learners**: XGBoost (optimized), LightGBM, RandomForest
2. **Meta-Learner**: Ridge Regression (stacking)
3. **Business Layer**: Growth correction (+19-21.5%)

## 🎨 Dashboard Features

- **Date Range Selector**: Filter Nov-Dec 2025 weekdays
- **Product Filter**: Select from 5 core products
- **Variant Filter**: Cascading dropdown with display names
- **YoY Comparison**: Visual 3-year revenue comparison
- **Product Breakdown**: Horizontal bar chart by product
- **Detailed Table**: Date × Product forecast breakdown

## 🔧 Configuration

Edit notebook parameters in Cell 1:

```python
CONFIG = Config(
    random_state=42,
    n_folds=5,
    optuna_trials=50,
    meta_model="ridge"
)
```

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Ali Boolmind**  
GitHub: [@aliboolmind228](https://github.com/aliboolmind228)

## 🙏 Acknowledgments

- Color palette: Okabe-Ito (color-blind friendly)
- ML frameworks: XGBoost, LightGBM, Scikit-learn
- Visualization: Streamlit, Plotly

