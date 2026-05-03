# ============================================================
# train_pipeline.py
# Academic GitHub Actions Training Pipeline
# Seoul ASOS Weather Data → Research-grade preprocessing/modeling/diagnostics
# ============================================================

import os
import glob
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# 1. Settings
# ============================================================

DATA_DIR = "data"
OUTPUT_DIR = "result"
MODEL_DIR = "result"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

PREDICT_HOUR = 1
TRAIN_RATIO = 0.8
RANDOM_STATE = 42
N_CV_SPLITS = 3


# ============================================================
# 2. Columns
# ============================================================

required_columns = [
    "지점",
    "지점명",
    "일시",
    "기온(°C)",
    "강수량(mm)",
    "풍속(m/s)",
    "습도(%)",
    "현지기압(hPa)",
    "해면기압(hPa)",
]

numeric_columns = [
    "지점",
    "기온(°C)",
    "강수량(mm)",
    "풍속(m/s)",
    "습도(%)",
    "현지기압(hPa)",
    "해면기압(hPa)",
]

interpolate_columns = [
    "기온(°C)",
    "풍속(m/s)",
    "습도(%)",
    "현지기압(hPa)",
    "해면기압(hPa)",
]

feature_columns = [
    "기온(°C)",
    "강수량(mm)",
    "풍속(m/s)",
    "습도(%)",
    "현지기압(hPa)",
    "해면기압(hPa)",
    "rain_yesno",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "season",
    "temp_1h_ago",
    "temp_3h_ago",
    "temp_6h_ago",
    "temp_12h_ago",
    "temp_24h_ago",
    "temp_48h_ago",
    "humidity_1h_ago",
    "pressure_1h_ago",
    "wind_1h_ago",
    "temp_diff_1h",
    "temp_diff_3h",
    "pressure_diff_1h",
    "temp_rolling_3h",
    "temp_rolling_6h",
    "humidity_rolling_3h",
    "pressure_rolling_3h",
    "temp_std_6h",
]

feature_name_map = {
    "기온(°C)": "Current Temperature",
    "강수량(mm)": "Rainfall",
    "풍속(m/s)": "Wind Speed",
    "습도(%)": "Humidity",
    "현지기압(hPa)": "Local Pressure",
    "해면기압(hPa)": "Sea-Level Pressure",
    "rain_yesno": "Rain Indicator",
    "hour_sin": "Hour Sin",
    "hour_cos": "Hour Cos",
    "month_sin": "Month Sin",
    "month_cos": "Month Cos",
    "season": "Season",
    "temp_1h_ago": "Temperature 1h Ago",
    "temp_3h_ago": "Temperature 3h Ago",
    "temp_6h_ago": "Temperature 6h Ago",
    "temp_12h_ago": "Temperature 12h Ago",
    "temp_24h_ago": "Temperature 24h Ago",
    "temp_48h_ago": "Temperature 48h Ago",
    "humidity_1h_ago": "Humidity 1h Ago",
    "pressure_1h_ago": "Pressure 1h Ago",
    "wind_1h_ago": "Wind Speed 1h Ago",
    "temp_diff_1h": "Temperature Change 1h",
    "temp_diff_3h": "Temperature Change 3h",
    "pressure_diff_1h": "Pressure Change 1h",
    "temp_rolling_3h": "Temperature Rolling Mean 3h",
    "temp_rolling_6h": "Temperature Rolling Mean 6h",
    "humidity_rolling_3h": "Humidity Rolling Mean 3h",
    "pressure_rolling_3h": "Pressure Rolling Mean 3h",
    "temp_std_6h": "Temperature Std 6h",
}


# ============================================================
# 3. Utility Functions
# ============================================================

def read_weather_csv(file_path: str) -> pd.DataFrame:
    """Read KMA ASOS CSV file."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="cp949")

    df.columns = [str(col).strip() for col in df.columns]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{os.path.basename(file_path)} missing columns: {missing_cols}")

    return df[required_columns].copy()


def make_season(month: int) -> int:
    if month in [3, 4, 5]:
        return 0
    if month in [6, 7, 8]:
        return 1
    if month in [9, 10, 11]:
        return 2
    return 3


def season_name(season_code: int) -> str:
    return {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}.get(season_code, "Unknown")


def add_features(df: pd.DataFrame, predict_hour: int = 1):
    """
    Create features and target.

    Model predicts temperature change:
        target_change = future_temp - current_temp
    Final prediction:
        predicted_temp = current_temp + predicted_change
    """
    df = df.copy()

    df["hour"] = df["일시"].dt.hour
    df["month"] = df["일시"].dt.month
    df["dayofyear"] = df["일시"].dt.dayofyear
    df["dayofweek"] = df["일시"].dt.dayofweek

    df["season"] = df["month"].apply(make_season)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["temp_1h_ago"] = df["기온(°C)"].shift(1)
    df["temp_3h_ago"] = df["기온(°C)"].shift(3)
    df["temp_6h_ago"] = df["기온(°C)"].shift(6)
    df["temp_12h_ago"] = df["기온(°C)"].shift(12)
    df["temp_24h_ago"] = df["기온(°C)"].shift(24)
    df["temp_48h_ago"] = df["기온(°C)"].shift(48)

    df["humidity_1h_ago"] = df["습도(%)"].shift(1)
    df["pressure_1h_ago"] = df["해면기압(hPa)"].shift(1)
    df["wind_1h_ago"] = df["풍속(m/s)"].shift(1)

    df["temp_diff_1h"] = df["기온(°C)"] - df["temp_1h_ago"]
    df["temp_diff_3h"] = df["기온(°C)"] - df["temp_3h_ago"]
    df["pressure_diff_1h"] = df["해면기압(hPa)"] - df["pressure_1h_ago"]

    df["temp_rolling_3h"] = df["기온(°C)"].rolling(window=3).mean()
    df["temp_rolling_6h"] = df["기온(°C)"].rolling(window=6).mean()
    df["humidity_rolling_3h"] = df["습도(%)"].rolling(window=3).mean()
    df["pressure_rolling_3h"] = df["해면기압(hPa)"].rolling(window=3).mean()
    df["temp_std_6h"] = df["기온(°C)"].rolling(window=6).std()

    df["rain_yesno"] = np.where(df["강수량(mm)"] > 0, 1, 0)

    actual_temp_col = f"actual_temp_{predict_hour}h_later"
    target_change_col = f"target_temp_change_{predict_hour}h"

    df[actual_temp_col] = df["기온(°C)"].shift(-predict_hour)
    df[target_change_col] = df[actual_temp_col] - df["기온(°C)"]

    return df, target_change_col, actual_temp_col


def regression_metrics(y_true, y_pred) -> dict:
    error = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "ME_Bias": float(np.mean(error)),
        "Error_STD": float(np.std(error)),
        "P90_Absolute_Error": float(np.quantile(np.abs(error), 0.90)),
        "P95_Absolute_Error": float(np.quantile(np.abs(error), 0.95)),
    }


def load_and_preprocess_data(data_dir: str):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV files found in data/. Upload Seoul ASOS CSV files first.")

    print("Loaded CSV files:")
    for file in csv_files:
        print("-", os.path.basename(file))

    df_list = []
    for file_path in csv_files:
        temp_df = read_weather_csv(file_path)
        temp_df["source_file"] = os.path.basename(file_path)
        df_list.append(temp_df)

    weather_raw = pd.concat(df_list, ignore_index=True)

    data_quality = {
        "source_file_count": len(csv_files),
        "raw_rows": int(len(weather_raw)),
        "raw_missing_datetime": int(pd.to_datetime(weather_raw["일시"], errors="coerce").isna().sum()),
    }

    weather = weather_raw.copy()
    weather["일시"] = pd.to_datetime(weather["일시"], errors="coerce")
    weather = weather.dropna(subset=["일시"]).copy()

    if weather.empty:
        raise ValueError("No valid datetime rows found. Check the '일시' column.")

    for col in numeric_columns:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    data_quality["rows_after_datetime_cleaning"] = int(len(weather))

    weather = weather[weather["지점"] == 108].copy()

    if weather.empty:
        raise ValueError("No rows found for Seoul ASOS station 108.")

    weather = weather.sort_values("일시").reset_index(drop=True)

    before_dedup = len(weather)
    weather = weather.drop_duplicates(subset=["일시"], keep="first").reset_index(drop=True)
    after_dedup = len(weather)

    data_quality["duplicate_timestamp_rows_removed"] = int(before_dedup - after_dedup)
    data_quality["rows_after_deduplication"] = int(after_dedup)
    data_quality["observed_start"] = str(weather["일시"].min())
    data_quality["observed_end"] = str(weather["일시"].max())

    missing_before = weather[required_columns].isna().sum().to_dict()

    weather = weather.set_index("일시").sort_index()

    full_time_index = pd.date_range(
        start=weather.index.min(),
        end=weather.index.max(),
        freq="h",
    )

    before_reindex = len(weather)
    weather = weather.reindex(full_time_index)
    after_reindex = len(weather)

    data_quality["missing_hour_rows_created"] = int(after_reindex - before_reindex)

    weather.index.name = "일시"
    weather = weather.reset_index()

    weather["지점"] = 108
    weather["지점명"] = "서울"
    weather["강수량(mm)"] = weather["강수량(mm)"].fillna(0)

    for col in interpolate_columns:
        weather[col] = weather[col].interpolate(method="linear")
        weather[col] = weather[col].ffill().bfill()

    missing_after = weather[required_columns].isna().sum().to_dict()

    data_quality_df = pd.DataFrame([
        {"Metric": "Source file count", "Value": data_quality["source_file_count"]},
        {"Metric": "Raw rows", "Value": data_quality["raw_rows"]},
        {"Metric": "Rows after datetime cleaning", "Value": data_quality["rows_after_datetime_cleaning"]},
        {"Metric": "Rows after deduplication", "Value": data_quality["rows_after_deduplication"]},
        {"Metric": "Duplicate timestamp rows removed", "Value": data_quality["duplicate_timestamp_rows_removed"]},
        {"Metric": "Missing-hour rows created", "Value": data_quality["missing_hour_rows_created"]},
        {"Metric": "Observed start", "Value": data_quality["observed_start"]},
        {"Metric": "Observed end", "Value": data_quality["observed_end"]},
    ])

    missing_report_df = pd.DataFrame({
        "Column": list(missing_before.keys()),
        "Missing_Before": list(missing_before.values()),
        "Missing_After": [missing_after.get(col, 0) for col in missing_before.keys()],
    })

    return weather, data_quality_df, missing_report_df


def save_line_plot(x, actual, pred, title, path, sample_size=500):
    sample_size = min(sample_size, len(actual))

    plt.figure(figsize=(15, 6))
    plt.plot(x.iloc[:sample_size], actual.iloc[:sample_size], label="Actual Temperature", linewidth=1.5)
    plt.plot(x.iloc[:sample_size], pred[:sample_size], label="Predicted Temperature", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_scatter_plot(actual, pred, title, path):
    plt.figure(figsize=(7, 7))
    plt.scatter(actual, pred, alpha=0.4)

    min_temp = min(actual.min(), pred.min())
    max_temp = max(actual.max(), pred.max())

    plt.plot([min_temp, max_temp], [min_temp, max_temp], linestyle="--")
    plt.title(title)
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# 4. Academic Diagnostics
# ============================================================

def make_train_climatology_prediction(train_df, test_df, actual_temp_col):
    """
    Leakage-safe climatology baseline.
    Uses only training period month-hour means.
    """
    train_clim = (
        train_df.groupby(["month", "hour"])[actual_temp_col]
        .mean()
        .reset_index()
        .rename(columns={actual_temp_col: "climatology_prediction"})
    )

    global_mean = train_df[actual_temp_col].mean()

    merged = test_df[["month", "hour"]].merge(train_clim, on=["month", "hour"], how="left")
    return merged["climatology_prediction"].fillna(global_mean).values


def run_time_series_cv(model_factory, X, y_change, current_temp, actual_future_temp, model_name):
    """
    Expanding-window TimeSeriesSplit.
    Evaluates final temperature, not just temperature change.
    """
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    rows = []

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        model = model_factory()
        model.fit(X.iloc[train_idx], y_change.iloc[train_idx])

        change_pred = model.predict(X.iloc[valid_idx])
        temp_pred = current_temp.iloc[valid_idx].values + change_pred
        y_true = actual_future_temp.iloc[valid_idx].values

        metrics = regression_metrics(y_true, temp_pred)
        metrics.update({
            "Model": model_name,
            "Fold": fold,
            "Train_Size": int(len(train_idx)),
            "Validation_Size": int(len(valid_idx)),
        })
        rows.append(metrics)

    return rows


def create_error_diagnostics(prediction_df):
    df = prediction_df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df["Hour"] = df["Time"].dt.hour
    df["Month"] = df["Time"].dt.month
    df["Season_Code"] = df["Month"].apply(make_season)
    df["Season"] = df["Season_Code"].apply(season_name)

    df["Temperature_Bin"] = pd.cut(
        df["Actual_Temperature"],
        bins=[-30, -10, 0, 10, 20, 30, 40],
        labels=["<-10", "-10~0", "0~10", "10~20", "20~30", "30+"],
        include_lowest=True,
    )

    def grouped_metrics(group_col):
        rows = []
        for key, g in df.groupby(group_col):
            rows.append({
                group_col: key,
                "Count": int(len(g)),
                **regression_metrics(g["Actual_Temperature"], g["Predicted_Temperature"]),
            })
        return pd.DataFrame(rows)

    return {
        "error_by_hour": grouped_metrics("Hour"),
        "error_by_month": grouped_metrics("Month"),
        "error_by_season": grouped_metrics("Season"),
        "error_by_temperature_bin": grouped_metrics("Temperature_Bin"),
    }


def create_model_card(summary_df, performance_df, cv_summary_df, data_quality_df):
    summary = summary_df.iloc[0].to_dict()

    return f"""# Seoul Short-Term Temperature Prediction Model Card

## Objective
This model predicts the 1-hour-ahead temperature change at Seoul ASOS station 108 and converts it into final temperature prediction:

```text
Predicted future temperature = current temperature + predicted temperature change
```

## Data
- Source: Seoul ASOS hourly observations uploaded in the `data/` folder.
- Station: Seoul ASOS 108.
- Training/evaluation split: chronological 80/20 split to avoid temporal leakage.
- Data cleaning: duplicate timestamp removal, hourly reindexing, rainfall missing values as 0, meteorological variables linearly interpolated.

## Data Quality Summary
{data_quality_df.to_markdown(index=False)}

## Modeling Approach
The target variable is temperature change rather than direct future temperature. This reduces over-reliance on current temperature persistence and makes the problem more scientifically interpretable as short-term thermal tendency prediction.

Models compared:
{performance_df.to_markdown(index=False)}

## Time-Series Cross-Validation Summary
{cv_summary_df.to_markdown(index=False)}

## Deployment Model
- Best model by test RMSE: {summary.get("best_model")}
- Deployment model: {summary.get("deploy_model")}
- Test MAE: {summary.get("mae"):.4f} °C
- Test RMSE: {summary.get("rmse"):.4f} °C
- Test R²: {summary.get("r2"):.4f}

## Academic Meaning
The model evaluates how short-term temperature tendency in Seoul is related to temporal continuity, diurnal cycle, seasonal cycle, humidity, wind, pressure, rainfall, and lagged thermal states.

## Limitations
- Single-station ASOS data cannot represent full spatial variation across Seoul.
- Future humidity, wind, pressure, and rainfall are unknown beyond the latest observation.
- Long-range future estimates should be interpreted as statistical scenario estimates, not official weather forecasts.
- Prediction uncertainty increases as iterative forecasting horizon increases.

## Reproducibility
The training pipeline is executed by GitHub Actions, and all model artifacts and diagnostic outputs are saved to `result/`.
"""


# ============================================================
# 5. Main Training Pipeline
# ============================================================

def main():
    weather, data_quality_df, missing_report_df = load_and_preprocess_data(DATA_DIR)

    weather_featured, target_change_col, actual_temp_col = add_features(
        weather, predict_hour=PREDICT_HOUR
    )
    weather_model = weather_featured.dropna().reset_index(drop=True)

    X = weather_model[feature_columns]
    y_change = weather_model[target_change_col]
    actual_future_temp = weather_model[actual_temp_col]
    current_temp = weather_model["기온(°C)"]

    split_index = int(len(weather_model) * TRAIN_RATIO)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y_change.iloc[:split_index]

    actual_future_temp_train = actual_future_temp.iloc[:split_index]
    actual_future_temp_test = actual_future_temp.iloc[split_index:]

    current_temp_train = current_temp.iloc[:split_index]
    current_temp_test = current_temp.iloc[split_index:]

    train_df = weather_model.iloc[:split_index].copy()
    test_df = weather_model.iloc[split_index:].copy()
    test_time = weather_model["일시"].iloc[split_index:]

    # ------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------

    # Persistence baseline: predicted future temp = current temp.
    baseline_temp_pred = current_temp_test.values
    baseline_change_pred = np.zeros(len(current_temp_test))

    # Previous-day baseline: predicted future temp = temperature 24h ago.
    previous_day_temp_pred = test_df["temp_24h_ago"].values

    # Leakage-safe climatology: month-hour mean from training period only.
    climatology_temp_pred = make_train_climatology_prediction(train_df, test_df, actual_temp_col)

    results = {
        "Persistence Baseline": {
            "model": None,
            "change_prediction": baseline_change_pred,
            "temperature_prediction": baseline_temp_pred,
            **regression_metrics(actual_future_temp_test, baseline_temp_pred),
        },
        "Previous-Day Baseline": {
            "model": None,
            "change_prediction": previous_day_temp_pred - current_temp_test.values,
            "temperature_prediction": previous_day_temp_pred,
            **regression_metrics(actual_future_temp_test, previous_day_temp_pred),
        },
        "Month-Hour Climatology": {
            "model": None,
            "change_prediction": climatology_temp_pred - current_temp_test.values,
            "temperature_prediction": climatology_temp_pred,
            **regression_metrics(actual_future_temp_test, climatology_temp_pred),
        },
    }

    model_factories = {
        "Linear Regression": lambda: LinearRegression(),
        "Random Forest Light": lambda: RandomForestRegressor(
            n_estimators=30,
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting Tuned": lambda: GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }

    # ------------------------------------------------------------
    # Time-series cross-validation
    # ------------------------------------------------------------

    cv_rows = []
    # To keep GitHub Actions runtime reasonable, CV is performed on ML models only.
    for model_name, factory in model_factories.items():
        print(f"Time-series CV: {model_name}")
        cv_rows.extend(
            run_time_series_cv(
                factory,
                X_train,
                y_change.iloc[:split_index],
                current_temp_train,
                actual_future_temp_train,
                model_name,
            )
        )

    cv_results_df = pd.DataFrame(cv_rows)

    cv_summary_df = (
        cv_results_df.groupby("Model")
        .agg(
            CV_MAE_Mean=("MAE", "mean"),
            CV_MAE_STD=("MAE", "std"),
            CV_RMSE_Mean=("RMSE", "mean"),
            CV_RMSE_STD=("RMSE", "std"),
            CV_R2_Mean=("R2", "mean"),
            CV_R2_STD=("R2", "std"),
        )
        .reset_index()
        .sort_values("CV_RMSE_Mean")
    )

    # ------------------------------------------------------------
    # Final train/test modeling
    # ------------------------------------------------------------

    trained_models = {}

    for model_name, factory in model_factories.items():
        print(f"Training final model: {model_name}")
        model = factory()
        model.fit(X_train, y_train)

        change_pred = model.predict(X_test)
        temp_pred = current_temp_test.values + change_pred

        trained_models[model_name] = model

        results[model_name] = {
            "model": model,
            "change_prediction": change_pred,
            "temperature_prediction": temp_pred,
            **regression_metrics(actual_future_temp_test, temp_pred),
        }

    performance_df = pd.DataFrame({
        "Model": list(results.keys()),
        "MAE": [results[name]["MAE"] for name in results],
        "RMSE": [results[name]["RMSE"] for name in results],
        "R2": [results[name]["R2"] for name in results],
        "ME_Bias": [results[name]["ME_Bias"] for name in results],
        "P90_Absolute_Error": [results[name]["P90_Absolute_Error"] for name in results],
        "P95_Absolute_Error": [results[name]["P95_Absolute_Error"] for name in results],
    }).sort_values("RMSE").reset_index(drop=True)

    ml_performance_df = performance_df[
        performance_df["Model"].isin(list(model_factories.keys()))
    ].reset_index(drop=True)

    best_model_name = ml_performance_df.iloc[0]["Model"]

    # Deployment model:
    # Use best ML model unless Linear Regression wins; in that case prefer Gradient Boosting for nonlinear interpretability.
    deploy_model_name = best_model_name
    if deploy_model_name == "Linear Regression":
        deploy_model_name = "Gradient Boosting Tuned"

    deploy_result = results[deploy_model_name]
    deploy_model = trained_models[deploy_model_name]

    prediction_df = pd.DataFrame({
        "Time": test_time.values,
        "Current_Temperature": current_temp_test.values,
        "Actual_Temperature": actual_future_temp_test.values,
        "Predicted_Temperature": deploy_result["temperature_prediction"],
        "Predicted_Change": deploy_result["change_prediction"],
        "Error": actual_future_temp_test.values - deploy_result["temperature_prediction"],
        "Absolute_Error": np.abs(actual_future_temp_test.values - deploy_result["temperature_prediction"]),
    })

    # Empirical 90% prediction interval from test residual distribution.
    # This is diagnostic uncertainty, not a formal probabilistic forecast.
    residual_q05 = prediction_df["Error"].quantile(0.05)
    residual_q95 = prediction_df["Error"].quantile(0.95)

    prediction_df["Prediction_Lower_90"] = prediction_df["Predicted_Temperature"] + residual_q05
    prediction_df["Prediction_Upper_90"] = prediction_df["Predicted_Temperature"] + residual_q95
    prediction_df["Within_90_Interval"] = (
        (prediction_df["Actual_Temperature"] >= prediction_df["Prediction_Lower_90"])
        & (prediction_df["Actual_Temperature"] <= prediction_df["Prediction_Upper_90"])
    )

    interval_coverage = prediction_df["Within_90_Interval"].mean()

    summary_df = pd.DataFrame([{
        "best_model": best_model_name,
        "deploy_model": deploy_model_name,
        "predict_hour": PREDICT_HOUR,
        "train_start": weather_model["일시"].iloc[0],
        "train_end": weather_model["일시"].iloc[split_index - 1],
        "test_start": weather_model["일시"].iloc[split_index],
        "test_end": weather_model["일시"].iloc[-1],
        "mae": deploy_result["MAE"],
        "rmse": deploy_result["RMSE"],
        "r2": deploy_result["R2"],
        "me_bias": deploy_result["ME_Bias"],
        "p90_absolute_error": deploy_result["P90_Absolute_Error"],
        "p95_absolute_error": deploy_result["P95_Absolute_Error"],
        "interval_90_coverage": interval_coverage,
        "baseline_mae": results["Persistence Baseline"]["MAE"],
        "baseline_rmse": results["Persistence Baseline"]["RMSE"],
        "baseline_r2": results["Persistence Baseline"]["R2"],
        "climatology_mae": results["Month-Hour Climatology"]["MAE"],
        "climatology_rmse": results["Month-Hour Climatology"]["RMSE"],
        "previous_day_mae": results["Previous-Day Baseline"]["MAE"],
        "previous_day_rmse": results["Previous-Day Baseline"]["RMSE"],
    }])

    if hasattr(deploy_model, "feature_importances_"):
        feature_importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Feature_English": [feature_name_map.get(f, f) for f in feature_columns],
            "Importance": deploy_model.feature_importances_,
        }).sort_values("Importance", ascending=False)
    else:
        feature_importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Feature_English": [feature_name_map.get(f, f) for f in feature_columns],
            "Importance": np.abs(deploy_model.coef_),
        }).sort_values("Importance", ascending=False)

    model_bundle = {
        "model": deploy_model,
        "model_name": deploy_model_name,
        "best_model_name": best_model_name,
        "feature_columns": feature_columns,
        "feature_name_map": feature_name_map,
        "predict_hour": PREDICT_HOUR,
        "target_type": "temperature_change",
        "residual_q05": float(residual_q05),
        "residual_q95": float(residual_q95),
    }

    # ------------------------------------------------------------
    # Error diagnostics
    # ------------------------------------------------------------

    diagnostics = create_error_diagnostics(prediction_df)
    residual_summary_df = pd.DataFrame([{
        "Metric": "Mean Error Bias",
        "Value": deploy_result["ME_Bias"],
        "Interpretation": "Positive means actual temperature tends to be higher than prediction.",
    }, {
        "Metric": "Error Standard Deviation",
        "Value": deploy_result["Error_STD"],
        "Interpretation": "Spread of residual errors.",
    }, {
        "Metric": "P90 Absolute Error",
        "Value": deploy_result["P90_Absolute_Error"],
        "Interpretation": "90% of absolute errors are below this value.",
    }, {
        "Metric": "P95 Absolute Error",
        "Value": deploy_result["P95_Absolute_Error"],
        "Interpretation": "95% of absolute errors are below this value.",
    }, {
        "Metric": "Empirical 90% Interval Coverage",
        "Value": interval_coverage,
        "Interpretation": "Share of test observations inside empirical 90% interval.",
    }])

    # ------------------------------------------------------------
    # Save result files
    # ------------------------------------------------------------

    weather_model.to_csv(os.path.join(OUTPUT_DIR, "seoul_weather_processed_dataset.csv"), index=False, encoding="utf-8-sig")
    performance_df.to_csv(os.path.join(OUTPUT_DIR, "model_performance_comparison.csv"), index=False, encoding="utf-8-sig")
    prediction_df.to_csv(os.path.join(OUTPUT_DIR, "temperature_prediction_result.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "project_summary.csv"), index=False, encoding="utf-8-sig")
    feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False, encoding="utf-8-sig")
    cv_results_df.to_csv(os.path.join(OUTPUT_DIR, "time_series_cv_results.csv"), index=False, encoding="utf-8-sig")
    cv_summary_df.to_csv(os.path.join(OUTPUT_DIR, "time_series_cv_summary.csv"), index=False, encoding="utf-8-sig")
    data_quality_df.to_csv(os.path.join(OUTPUT_DIR, "data_quality_report.csv"), index=False, encoding="utf-8-sig")
    missing_report_df.to_csv(os.path.join(OUTPUT_DIR, "missing_value_report.csv"), index=False, encoding="utf-8-sig")
    residual_summary_df.to_csv(os.path.join(OUTPUT_DIR, "residual_analysis_summary.csv"), index=False, encoding="utf-8-sig")

    for name, df in diagnostics.items():
        df.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False, encoding="utf-8-sig")

    dump(model_bundle, os.path.join(OUTPUT_DIR, "seoul_temperature_model.joblib"))

    model_card_text = create_model_card(summary_df, performance_df, cv_summary_df, data_quality_df)
    with open(os.path.join(OUTPUT_DIR, "model_card.md"), "w", encoding="utf-8") as f:
        f.write(model_card_text)

    # ------------------------------------------------------------
    # Save visualizations
    # ------------------------------------------------------------

    plt.figure(figsize=(15, 5))
    plt.plot(weather_model["일시"], weather_model["기온(°C)"], linewidth=0.8)
    plt.title("Seoul Temperature Time Series")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "temperature_time_series.png"), dpi=300)
    plt.close()

    save_line_plot(
        test_time,
        actual_future_temp_test,
        deploy_result["temperature_prediction"],
        f"Test Data: Actual vs Predicted Temperature - {deploy_model_name}",
        os.path.join(OUTPUT_DIR, "test_actual_vs_predicted_line.png"),
    )

    save_scatter_plot(
        actual_future_temp_test,
        deploy_result["temperature_prediction"],
        f"Test Data: Actual vs Predicted Scatter Plot - {deploy_model_name}",
        os.path.join(OUTPUT_DIR, "test_actual_vs_predicted_scatter.png"),
    )

    sample_size = min(500, len(prediction_df))

    plt.figure(figsize=(15, 5))
    plt.plot(pd.to_datetime(prediction_df["Time"]).iloc[:sample_size], prediction_df["Error"].iloc[:sample_size])
    plt.axhline(0, linestyle="--")
    plt.title("Test Data Error Over Time")
    plt.xlabel("Time")
    plt.ylabel("Error (Actual - Predicted)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test_error_over_time.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    x = np.arange(len(performance_df))
    plt.bar(x - 0.25, performance_df["MAE"], width=0.25, label="MAE")
    plt.bar(x, performance_df["RMSE"], width=0.25, label="RMSE")
    plt.bar(x + 0.25, performance_df["R2"], width=0.25, label="R²")
    plt.xticks(x, performance_df["Model"], rotation=15)
    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_performance_comparison.png"), dpi=300)
    plt.close()

    top15 = feature_importance_df.head(15)
    plt.figure(figsize=(10, 7))
    plt.barh(top15["Feature_English"], top15["Importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Top 15 Feature Importance - {deploy_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_top15.png"), dpi=300)
    plt.close()

    no_temp = feature_importance_df[
        feature_importance_df["Feature"] != "기온(°C)"
    ].sort_values("Importance", ascending=False).head(15)

    plt.figure(figsize=(10, 7))
    plt.barh(no_temp["Feature_English"], no_temp["Importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance Excluding Current Temperature - {deploy_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_except_current_temperature.png"), dpi=300)
    plt.close()

    # Error by hour plot
    error_by_hour = diagnostics["error_by_hour"]
    plt.figure(figsize=(11, 5))
    plt.plot(error_by_hour["Hour"], error_by_hour["MAE"], marker="o", label="MAE")
    plt.plot(error_by_hour["Hour"], error_by_hour["RMSE"], marker="o", label="RMSE")
    plt.title("Prediction Error by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Error (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "error_by_hour.png"), dpi=300)
    plt.close()

    # CV summary plot
    plt.figure(figsize=(10, 5))
    plt.bar(cv_summary_df["Model"], cv_summary_df["CV_RMSE_Mean"])
    plt.title("Time-Series Cross-Validation RMSE")
    plt.xlabel("Model")
    plt.ylabel("CV RMSE Mean (°C)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "time_series_cv_rmse.png"), dpi=300)
    plt.close()

    print("Training completed.")
    print("Best model:", best_model_name)
    print("Deploy model:", deploy_model_name)
    print("MAE:", round(deploy_result["MAE"], 4))
    print("RMSE:", round(deploy_result["RMSE"], 4))
    print("R2:", round(deploy_result["R2"], 4))
    print("Empirical 90% interval coverage:", round(interval_coverage, 4))


if __name__ == "__main__":
    main()
