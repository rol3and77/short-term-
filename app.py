import os
import glob
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Streamlit Page Config
# ============================================================

st.set_page_config(
    page_title="Seoul Weather ML Dashboard",
    page_icon="🌤️",
    layout="wide",
)


# ============================================================
# Path Settings
# ============================================================

DATA_DIR = "data"
RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

API_URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
SEOUL_STATION_ID = "108"

PREDICT_HOUR = 1
TRAIN_RATIO = 0.8
RANDOM_STATE = 42


# ============================================================
# Column Settings
# ============================================================

REQUIRED_COLUMNS = [
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

NUMERIC_COLUMNS = [
    "지점",
    "기온(°C)",
    "강수량(mm)",
    "풍속(m/s)",
    "습도(%)",
    "현지기압(hPa)",
    "해면기압(hPa)",
]

INTERPOLATE_COLUMNS = [
    "기온(°C)",
    "풍속(m/s)",
    "습도(%)",
    "현지기압(hPa)",
    "해면기압(hPa)",
]

FEATURE_COLUMNS = [
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

FEATURE_NAME_MAP = {
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
# Utility Functions
# ============================================================

def read_weather_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="cp949")

    df.columns = [str(col).strip() for col in df.columns]

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{os.path.basename(file_path)} missing columns: {missing_cols}")

    return df[REQUIRED_COLUMNS].copy()


def make_season(month: int) -> int:
    if month in [3, 4, 5]:
        return 0
    if month in [6, 7, 8]:
        return 1
    if month in [9, 10, 11]:
        return 2
    return 3


def add_features(df: pd.DataFrame, predict_hour: int = 1) -> tuple[pd.DataFrame, str, str]:
    """
    Feature engineering.

    This version predicts temperature change instead of direct temperature:
        target_change = temperature_after_1h - current_temperature

    Final temperature prediction:
        predicted_temperature = current_temperature + predicted_change
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


def load_and_preprocess_data(data_dir: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No CSV files found in the data folder. "
            "Upload your Seoul ASOS CSV files to the data/ folder in GitHub."
        )

    df_list = []
    for file_path in csv_files:
        temp_df = read_weather_csv(file_path)
        temp_df["source_file"] = os.path.basename(file_path)
        df_list.append(temp_df)

    weather_raw = pd.concat(df_list, ignore_index=True)

    weather = weather_raw.copy()
    weather["일시"] = pd.to_datetime(weather["일시"], errors="coerce")
    weather = weather.dropna(subset=["일시"]).copy()

    if weather.empty:
        raise ValueError(
            "No valid datetime rows were found. Please check that the CSV files in data/ contain the column '일시'."
        )

    for col in NUMERIC_COLUMNS:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather = weather[weather["지점"] == 108].copy()
    weather = weather.sort_values("일시").reset_index(drop=True)
    weather = weather.drop_duplicates(subset=["일시"], keep="first").reset_index(drop=True)

    weather = weather.set_index("일시").sort_index()

    full_time_index = pd.date_range(
        start=weather.index.min(),
        end=weather.index.max(),
        freq="h"
    )

    weather = weather.reindex(full_time_index)
    weather.index.name = "일시"
    weather = weather.reset_index()

    weather["지점"] = 108
    weather["지점명"] = "서울"
    weather["강수량(mm)"] = weather["강수량(mm)"].fillna(0)

    for col in INTERPOLATE_COLUMNS:
        weather[col] = weather[col].interpolate(method="linear")
        weather[col] = weather[col].ffill().bfill()

    return weather


def train_models(weather: pd.DataFrame) -> dict:
    weather, target_change_col, actual_temp_col = add_features(weather, predict_hour=PREDICT_HOUR)
    weather_model = weather.dropna().reset_index(drop=True)

    X = weather_model[FEATURE_COLUMNS]
    y_change = weather_model[target_change_col]
    actual_future_temp = weather_model[actual_temp_col]
    current_temp = weather_model["기온(°C)"]

    split_index = int(len(weather_model) * TRAIN_RATIO)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y_change.iloc[:split_index]
    y_test_change = y_change.iloc[split_index:]

    actual_future_temp_test = actual_future_temp.iloc[split_index:]
    current_temp_test = current_temp.iloc[split_index:]
    test_time = weather_model["일시"].iloc[split_index:]

    # Baseline: future temperature = current temperature
    baseline_temp_pred = current_temp_test.values
    baseline_change_pred = np.zeros(len(current_temp_test))

    results = {
        "Persistence Baseline": {
            "model": None,
            "change_prediction": baseline_change_pred,
            "temperature_prediction": baseline_temp_pred,
            "MAE": mean_absolute_error(actual_future_temp_test, baseline_temp_pred),
            "RMSE": np.sqrt(mean_squared_error(actual_future_temp_test, baseline_temp_pred)),
            "R2": r2_score(actual_future_temp_test, baseline_temp_pred),
        }
    }

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Light": RandomForestRegressor(
            n_estimators=30,
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting Tuned": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        change_pred = model.predict(X_test)
        temp_pred = current_temp_test.values + change_pred

        results[model_name] = {
            "model": model,
            "change_prediction": change_pred,
            "temperature_prediction": temp_pred,
            "MAE": mean_absolute_error(actual_future_temp_test, temp_pred),
            "RMSE": np.sqrt(mean_squared_error(actual_future_temp_test, temp_pred)),
            "R2": r2_score(actual_future_temp_test, temp_pred),
        }

    performance_df = pd.DataFrame({
        "Model": list(results.keys()),
        "MAE": [results[name]["MAE"] for name in results],
        "RMSE": [results[name]["RMSE"] for name in results],
        "R2": [results[name]["R2"] for name in results],
    }).sort_values("RMSE").reset_index(drop=True)

    ml_performance_df = performance_df[performance_df["Model"] != "Persistence Baseline"].reset_index(drop=True)
    best_model_name = ml_performance_df.iloc[0]["Model"]

    # Deployment model preference:
    # If Random Forest Light is best and still light enough in runtime, use it.
    # Otherwise use tuned Gradient Boosting.
    deploy_model_name = best_model_name
    if deploy_model_name == "Linear Regression":
        deploy_model_name = "Gradient Boosting Tuned"

    deploy_result = results[deploy_model_name]

    prediction_df = pd.DataFrame({
        "Time": test_time.values,
        "Current_Temperature": current_temp_test.values,
        "Actual_Temperature": actual_future_temp_test.values,
        "Predicted_Temperature": deploy_result["temperature_prediction"],
        "Predicted_Change": deploy_result["change_prediction"],
        "Error": actual_future_temp_test.values - deploy_result["temperature_prediction"],
        "Absolute_Error": np.abs(actual_future_temp_test.values - deploy_result["temperature_prediction"]),
    })

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
        "baseline_mae": results["Persistence Baseline"]["MAE"],
        "baseline_rmse": results["Persistence Baseline"]["RMSE"],
        "baseline_r2": results["Persistence Baseline"]["R2"],
    }])

    deploy_model = deploy_result["model"]

    if hasattr(deploy_model, "feature_importances_"):
        feature_importance_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Feature_English": [FEATURE_NAME_MAP.get(f, f) for f in FEATURE_COLUMNS],
            "Importance": deploy_model.feature_importances_,
        }).sort_values("Importance", ascending=False)
    elif deploy_model_name == "Linear Regression":
        feature_importance_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Feature_English": [FEATURE_NAME_MAP.get(f, f) for f in FEATURE_COLUMNS],
            "Importance": np.abs(deploy_model.coef_),
        }).sort_values("Importance", ascending=False)
    else:
        feature_importance_df = pd.DataFrame(columns=["Feature", "Feature_English", "Importance"])

    return {
        "weather": weather,
        "weather_model": weather_model,
        "performance_df": performance_df,
        "prediction_df": prediction_df,
        "summary_df": summary_df,
        "feature_importance_df": feature_importance_df,
        "model": deploy_model,
        "model_name": deploy_model_name,
        "best_model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_change_col": target_change_col,
        "actual_temp_col": actual_temp_col,
    }


@st.cache_resource(show_spinner="Training models from GitHub data files...")
def get_pipeline_result() -> dict:
    weather = load_and_preprocess_data(DATA_DIR)
    result = train_models(weather)
    return result


def make_api_params(service_key: str):
    today = datetime.now()

    end_time = today - timedelta(days=1)
    end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)

    start_time = end_time - timedelta(days=3)
    start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)

    params = {
        "serviceKey": service_key,
        "pageNo": "1",
        "numOfRows": "100",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "HR",
        "startDt": start_time.strftime("%Y%m%d"),
        "startHh": start_time.strftime("%H"),
        "endDt": end_time.strftime("%Y%m%d"),
        "endHh": end_time.strftime("%H"),
        "stnIds": SEOUL_STATION_ID,
    }

    return params, start_time, end_time


def request_asos_api(service_key: str, retry: int = 5, wait: int = 3):
    params, start_time, end_time = make_api_params(service_key)
    last_error = None

    for _ in range(1, retry + 1):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"},
            )

            if response.status_code != 200:
                last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
                time.sleep(wait)
                continue

            if response.text.strip() == "":
                last_error = RuntimeError("API returned an empty response.")
                time.sleep(wait)
                continue

            try:
                data = response.json()
            except Exception:
                last_error = RuntimeError(f"API response is not JSON: {response.text[:300]}")
                time.sleep(wait)
                continue

            header = data.get("response", {}).get("header", {})
            result_code = header.get("resultCode")
            result_msg = header.get("resultMsg")

            if result_code != "00":
                raise RuntimeError(f"API error {result_code}: {result_msg}")

            items = data["response"]["body"]["items"]["item"]
            api_df = pd.DataFrame(items)

            return api_df, start_time, end_time

        except Exception as e:
            last_error = e
            time.sleep(wait)

    raise RuntimeError(f"API request failed after retries: {last_error}")


def clean_api_data(api_df: pd.DataFrame) -> pd.DataFrame:
    api_weather = api_df.rename(
        columns={
            "stnId": "지점",
            "stnNm": "지점명",
            "tm": "일시",
            "ta": "기온(°C)",
            "rn": "강수량(mm)",
            "ws": "풍속(m/s)",
            "hm": "습도(%)",
            "pa": "현지기압(hPa)",
            "ps": "해면기압(hPa)",
        }
    )

    api_weather = api_weather[REQUIRED_COLUMNS].copy()
    api_weather["일시"] = pd.to_datetime(api_weather["일시"])

    for col in NUMERIC_COLUMNS:
        api_weather[col] = pd.to_numeric(api_weather[col], errors="coerce")

    api_weather["강수량(mm)"] = api_weather["강수량(mm)"].fillna(0)
    api_weather = api_weather.sort_values("일시").reset_index(drop=True)

    for col in INTERPOLATE_COLUMNS:
        api_weather[col] = api_weather[col].interpolate(method="linear")
        api_weather[col] = api_weather[col].ffill().bfill()

    return api_weather


def predict_latest_from_api(api_weather: pd.DataFrame, pipeline: dict) -> dict:
    model = pipeline["model"]
    feature_columns = pipeline["feature_columns"]

    if len(api_weather) < 49:
        raise ValueError("At least 49 hourly records are required because the model uses 48-hour lag features.")

    api_featured, _, _ = add_features(api_weather, predict_hour=PREDICT_HOUR)
    api_featured = api_featured.dropna().reset_index(drop=True)

    latest = api_featured.iloc[-1].copy()
    current_time = latest["일시"]
    current_temp = latest["기온(°C)"]

    X_latest = pd.DataFrame([latest[feature_columns]])
    predicted_change = model.predict(X_latest)[0]
    predicted_temp = current_temp + predicted_change

    return {
        "base_time": current_time,
        "target_time": current_time + pd.Timedelta(hours=PREDICT_HOUR),
        "current_temp": current_temp,
        "predicted_change": predicted_change,
        "predicted_temp": predicted_temp,
    }


def compare_api_interval(api_weather: pd.DataFrame, pipeline: dict) -> pd.DataFrame:
    model = pipeline["model"]
    feature_columns = pipeline["feature_columns"]

    api_compare, target_change_col, actual_temp_col = add_features(api_weather, predict_hour=PREDICT_HOUR)
    api_compare_model = api_compare.dropna().reset_index(drop=True)

    X_api = api_compare_model[feature_columns]
    predicted_change = model.predict(X_api)
    predicted_temp = api_compare_model["기온(°C)"].values + predicted_change

    compare_result = pd.DataFrame({
        "Base_Time": api_compare_model["일시"],
        "Prediction_Time": api_compare_model["일시"] + pd.Timedelta(hours=PREDICT_HOUR),
        "Current_Temperature": api_compare_model["기온(°C)"],
        "Actual_Temperature": api_compare_model[actual_temp_col],
        "Predicted_Temperature": predicted_temp,
        "Predicted_Change": predicted_change,
    })

    compare_result["Error"] = compare_result["Actual_Temperature"] - compare_result["Predicted_Temperature"]
    compare_result["Absolute_Error"] = np.abs(compare_result["Error"])

    return compare_result


# ============================================================
# UI
# ============================================================

st.title("🌤️ Seoul Weather ML Dashboard")
st.markdown(
    """
    기상청 서울 ASOS 108번 지점의 5년치 시간별 관측자료를 기반으로  
    **1시간 뒤 서울 기온 변화량을 예측한 뒤 현재 기온에 더해 최종 기온을 산출**하는 머신러닝 대시보드입니다.
    """
)

with st.spinner("Loading data and training models from GitHub repository..."):
    try:
        pipeline = get_pipeline_result()
    except Exception as e:
        st.error("앱 초기화에 실패했습니다.")
        st.write(e)
        st.info("GitHub repository의 data/ 폴더에 서울 ASOS CSV 파일들이 들어 있는지 확인하세요.")
        st.stop()

performance_df = pipeline["performance_df"]
prediction_df = pipeline["prediction_df"]
summary_df = pipeline["summary_df"]
feature_importance_df = pipeline["feature_importance_df"]
summary = summary_df.iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Deploy Model", summary["deploy_model"])
col2.metric("Prediction Target", "1 hour later")
col3.metric("MAE", f"{summary['mae']:.3f} °C")
col4.metric("RMSE", f"{summary['rmse']:.3f} °C")

st.divider()

tab_overview, tab_results, tab_api, tab_future, tab_analysis, tab_data = st.tabs(
    ["Overview", "Model Results", "API Prediction", "Future Forecast", "Custom Analysis", "Data Preview"]
)


with tab_overview:
    st.subheader("Project Overview")

    st.markdown(
        """
        이 버전은 GitHub와 Streamlit만으로 실행되도록 구성되어 있습니다.

        **핵심 개선점**
        - Colab에서 모델 파일을 따로 저장하지 않아도 됨
        - GitHub `data/` 폴더의 CSV를 Streamlit에서 직접 읽고 학습
        - 직접 기온을 예측하는 대신 **1시간 뒤 기온 변화량**을 예측
        - 최종 기온 = 현재 기온 + 예측 변화량
        - Baseline과 머신러닝 모델 성능 비교
        - API 최신 제공 자료 기반 예측 지원
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline RMSE", f"{summary['baseline_rmse']:.3f} °C")
    c2.metric("Deploy Model RMSE", f"{summary['rmse']:.3f} °C")
    c3.metric("R²", f"{summary['r2']:.4f}")

    st.warning(
        "ASOS 시간자료 API는 실시간 현재 자료가 아니라 전날 자료까지 제공합니다. "
        "따라서 API 예측은 'API에서 제공되는 최신 관측 시각 기준 1시간 뒤 예측'으로 해석해야 합니다."
    )


with tab_results:
    st.subheader("Model Performance Comparison")

    st.dataframe(performance_df, use_container_width=True)

    perf_long = performance_df.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE", "R2"],
        var_name="Metric",
        value_name="Value",
    )

    fig_perf = px.bar(
        perf_long,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Model Performance Comparison",
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    st.subheader("Test Data: Actual vs Predicted")

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=prediction_df["Time"],
            y=prediction_df["Actual_Temperature"],
            mode="lines",
            name="Actual Temperature",
        )
    )
    fig_line.add_trace(
        go.Scatter(
            x=prediction_df["Time"],
            y=prediction_df["Predicted_Temperature"],
            mode="lines",
            name="Predicted Temperature",
        )
    )
    fig_line.update_layout(
        title="Actual vs Predicted Temperature",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    fig_scatter = px.scatter(
        prediction_df,
        x="Actual_Temperature",
        y="Predicted_Temperature",
        title="Actual vs Predicted Scatter Plot",
        labels={
            "Actual_Temperature": "Actual Temperature (°C)",
            "Predicted_Temperature": "Predicted Temperature (°C)",
        },
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Predicted Temperature Change")
    fig_change = px.line(
        prediction_df,
        x="Time",
        y="Predicted_Change",
        title="Predicted 1-Hour Temperature Change",
        labels={"Predicted_Change": "Predicted Change (°C)", "Time": "Time"},
    )
    fig_change.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_change, use_container_width=True)

    st.subheader("Prediction Error Over Time")
    fig_error = px.line(
        prediction_df,
        x="Time",
        y="Error",
        title="Error Over Time",
        labels={"Error": "Error (Actual - Predicted)", "Time": "Time"},
    )
    fig_error.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_error, use_container_width=True)

    if len(feature_importance_df) > 0:
        st.subheader("Feature Importance")
        top_features = feature_importance_df.sort_values("Importance", ascending=False).head(15)

        fig_feature = px.bar(
            top_features,
            x="Importance",
            y="Feature_English",
            orientation="h",
            title="Top 15 Feature Importance",
        )
        fig_feature.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_feature, use_container_width=True)

        no_current_temp = feature_importance_df[
            feature_importance_df["Feature"] != "기온(°C)"
        ].sort_values("Importance", ascending=False).head(15)

        st.subheader("Feature Importance Except Current Temperature")
        fig_no_temp = px.bar(
            no_current_temp,
            x="Importance",
            y="Feature_English",
            orientation="h",
            title="Feature Importance Except Current Temperature",
        )
        fig_no_temp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_no_temp, use_container_width=True)


with tab_api:
    st.subheader("API-Based Latest Available Prediction")

    st.warning(
        "공공데이터포털 ASOS 시간자료 API는 전날 자료까지 제공합니다. "
        "따라서 최신 제공 가능 시각 기준으로 1시간 뒤 기온을 예측합니다."
    )

    service_key = st.text_input(
        "Public Data Portal API Key",
        type="password",
        help="GitHub에 API 키를 직접 올리지 말고, 이 입력창에 넣어 사용하세요.",
    )

    if st.button("Fetch API Data and Predict"):
        if not service_key:
            st.error("API 키를 입력하세요.")
        else:
            try:
                with st.spinner("ASOS API 자료를 불러오는 중입니다..."):
                    api_df, start_time, end_time = request_asos_api(service_key, retry=5, wait=3)
                    api_weather = clean_api_data(api_df)

                    latest_result = predict_latest_from_api(api_weather, pipeline)
                    compare_result = compare_api_interval(api_weather, pipeline)

                    api_mae = mean_absolute_error(
                        compare_result["Actual_Temperature"],
                        compare_result["Predicted_Temperature"],
                    )
                    api_rmse = np.sqrt(
                        mean_squared_error(
                            compare_result["Actual_Temperature"],
                            compare_result["Predicted_Temperature"],
                        )
                    )
                    api_r2 = r2_score(
                        compare_result["Actual_Temperature"],
                        compare_result["Predicted_Temperature"],
                    )

                st.success("API 예측 완료")

                a1, a2, a3, a4, a5 = st.columns(5)
                a1.metric("Base Time", str(latest_result["base_time"]))
                a2.metric("Target Time", str(latest_result["target_time"]))
                a3.metric("Current Temp", f"{latest_result['current_temp']:.1f} °C")
                a4.metric("Predicted Change", f"{latest_result['predicted_change']:.2f} °C")
                a5.metric("Predicted Temp", f"{latest_result['predicted_temp']:.2f} °C")

                m1, m2, m3 = st.columns(3)
                m1.metric("API MAE", f"{api_mae:.3f} °C")
                m2.metric("API RMSE", f"{api_rmse:.3f} °C")
                m3.metric("API R²", f"{api_r2:.4f}")

                fig_api = go.Figure()
                fig_api.add_trace(
                    go.Scatter(
                        x=compare_result["Prediction_Time"],
                        y=compare_result["Actual_Temperature"],
                        mode="lines+markers",
                        name="Actual",
                    )
                )
                fig_api.add_trace(
                    go.Scatter(
                        x=compare_result["Prediction_Time"],
                        y=compare_result["Predicted_Temperature"],
                        mode="lines+markers",
                        name="Predicted",
                    )
                )
                fig_api.update_layout(
                    title="API Interval: Actual vs Predicted",
                    xaxis_title="Time",
                    yaxis_title="Temperature (°C)",
                )
                st.plotly_chart(fig_api, use_container_width=True)

                fig_api_change = px.line(
                    compare_result,
                    x="Prediction_Time",
                    y="Predicted_Change",
                    title="API Interval: Predicted Temperature Change",
                    labels={"Predicted_Change": "Predicted Change (°C)"},
                )
                fig_api_change.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig_api_change, use_container_width=True)

                st.subheader("API Comparison Table")
                st.dataframe(compare_result.tail(30), use_container_width=True)

            except Exception as e:
                st.error(f"API prediction failed: {e}")





with tab_future:
    st.subheader("Future Forecast After Latest Uploaded Data")
    st.markdown(
        """
        이 탭은 업로드된 데이터의 마지막 시점 이후를 예측합니다.  
        모델은 **1시간 뒤 기온 변화량**을 예측하고, 예측값을 다음 시점의 입력으로 다시 사용하여
        여러 시간 뒤까지 반복 예측합니다.

        단, 미래의 습도·풍속·기압·강수량은 실제로 알 수 없으므로 아래에서 선택한 가정 방식으로 유지합니다.
        따라서 예측 시간이 길어질수록 불확실성이 커질 수 있습니다.
        """
    )

    def iterative_future_forecast(history_df: pd.DataFrame, model_bundle: dict, horizon_hours: int, exogenous_mode: str):
        model = model_bundle["model"]
        feature_columns = model_bundle["feature_columns"]
        predict_hour = model_bundle.get("predict_hour", 1)

        future_history = history_df[REQUIRED_COLUMNS].copy()
        future_history["일시"] = pd.to_datetime(future_history["일시"])
        future_history = future_history.sort_values("일시").reset_index(drop=True)

        if len(future_history) < 60:
            raise ValueError("미래 예측을 위해 최소 60시간 이상의 데이터가 필요합니다.")

        forecast_rows = []

        for step in range(1, horizon_hours + 1):
            featured, _, _ = add_features(future_history, predict_hour=predict_hour)

            latest_feature_row = featured.iloc[-1].copy()

            missing_features = latest_feature_row[feature_columns].isna().sum()
            if missing_features > 0:
                raise ValueError("예측에 필요한 파생변수 중 결측값이 있습니다. 데이터 길이를 확인하세요.")

            current_time = latest_feature_row["일시"]
            current_temp = latest_feature_row["기온(°C)"]

            X_latest = pd.DataFrame([latest_feature_row[feature_columns]])
            predicted_change = float(model.predict(X_latest)[0])
            predicted_temp = float(current_temp + predicted_change)

            next_time = current_time + pd.Timedelta(hours=1)

            # Future exogenous variables are assumptions.
            recent_6h = future_history.tail(6)

            if exogenous_mode == "Recent 6-hour average":
                next_rainfall = float(recent_6h["강수량(mm)"].mean())
                next_wind = float(recent_6h["풍속(m/s)"].mean())
                next_humidity = float(recent_6h["습도(%)"].mean())
                next_local_pressure = float(recent_6h["현지기압(hPa)"].mean())
                next_sea_pressure = float(recent_6h["해면기압(hPa)"].mean())
            else:
                next_rainfall = float(future_history.iloc[-1]["강수량(mm)"])
                next_wind = float(future_history.iloc[-1]["풍속(m/s)"])
                next_humidity = float(future_history.iloc[-1]["습도(%)"])
                next_local_pressure = float(future_history.iloc[-1]["현지기압(hPa)"])
                next_sea_pressure = float(future_history.iloc[-1]["해면기압(hPa)"])

            forecast_rows.append({
                "Forecast_Step": step,
                "Base_Time": current_time,
                "Forecast_Time": next_time,
                "Base_Temperature": current_temp,
                "Predicted_Change": predicted_change,
                "Predicted_Temperature": predicted_temp,
                "Assumed_Humidity": next_humidity,
                "Assumed_Wind_Speed": next_wind,
                "Assumed_Rainfall": next_rainfall,
                "Assumed_Sea_Level_Pressure": next_sea_pressure,
            })

            next_row = {
                "지점": 108,
                "지점명": "서울",
                "일시": next_time,
                "기온(°C)": predicted_temp,
                "강수량(mm)": next_rainfall,
                "풍속(m/s)": next_wind,
                "습도(%)": next_humidity,
                "현지기압(hPa)": next_local_pressure,
                "해면기압(hPa)": next_sea_pressure,
            }

            future_history = pd.concat([future_history, pd.DataFrame([next_row])], ignore_index=True)

        return pd.DataFrame(forecast_rows)

    history_for_future = processed_df[REQUIRED_COLUMNS].copy()
    history_for_future["일시"] = pd.to_datetime(history_for_future["일시"])
    history_for_future = history_for_future.sort_values("일시").reset_index(drop=True)

    latest_obs = history_for_future.iloc[-1]

    f1, f2, f3 = st.columns([1, 1, 1.2])

    with f1:
        horizon_hours = st.slider(
            "Forecast horizon",
            min_value=1,
            max_value=48,
            value=12,
            help="업로드된 데이터 이후 몇 시간까지 예측할지 선택하세요."
        )

    with f2:
        exogenous_mode = st.selectbox(
            "Future weather assumption",
            ["Hold last observed", "Recent 6-hour average"],
            help="미래 습도·풍속·기압·강수량을 어떻게 가정할지 선택합니다."
        )

    with f3:
        st.metric("Latest Data Time", str(latest_obs["일시"]))
        st.metric("Latest Temperature", f"{latest_obs['기온(°C)']:.2f} °C")

    try:
        future_result = iterative_future_forecast(
            history_df=history_for_future,
            model_bundle=model_bundle,
            horizon_hours=int(horizon_hours),
            exogenous_mode=exogenous_mode
        )

        final_row = future_result.iloc[-1]

        fm1, fm2, fm3, fm4 = st.columns(4)
        fm1.metric("Final Forecast Time", str(final_row["Forecast_Time"]))
        fm2.metric("Final Predicted Temp", f"{final_row['Predicted_Temperature']:.2f} °C")
        fm3.metric("Total Temp Change", f"{final_row['Predicted_Temperature'] - latest_obs['기온(°C)']:.2f} °C")
        fm4.metric("Forecast Steps", f"{len(future_result)} h")

        fig_future = go.Figure()
        fig_future.add_trace(
            go.Scatter(
                x=[latest_obs["일시"]],
                y=[latest_obs["기온(°C)"]],
                mode="markers",
                name="Latest Observed Temperature",
                marker=dict(size=10)
            )
        )
        fig_future.add_trace(
            go.Scatter(
                x=future_result["Forecast_Time"],
                y=future_result["Predicted_Temperature"],
                mode="lines+markers",
                name="Future Predicted Temperature"
            )
        )
        fig_future.update_layout(
            title="Future Temperature Forecast After Latest Uploaded Data",
            xaxis_title="Forecast Time",
            yaxis_title="Temperature (°C)"
        )
        st.plotly_chart(fig_future, use_container_width=True)

        fig_change_future = px.bar(
            future_result,
            x="Forecast_Time",
            y="Predicted_Change",
            title="Predicted Hourly Temperature Change",
            labels={"Predicted_Change": "Predicted Change (°C)", "Forecast_Time": "Forecast Time"}
        )
        fig_change_future.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_change_future, use_container_width=True)

        st.subheader("Future Forecast Table")
        st.dataframe(future_result, use_container_width=True)

        csv_future = future_result.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Download future forecast as CSV",
            data=csv_future,
            file_name="future_forecast_after_latest_data.csv",
            mime="text/csv",
        )

        st.info(
            "이 예측은 데이터의 마지막 시점 이후를 반복 예측한 결과입니다. "
            "미래의 습도·풍속·기압·강수량은 실제 관측값이 아니라 사용자가 선택한 방식으로 가정한 값입니다."
        )

    except Exception as e:
        st.error(f"Future forecast failed: {e}")



with tab_analysis:
    st.subheader("Custom Temperature Analysis")
    st.markdown(
        """
        원하는 날짜 범위와 시간대를 선택하면 해당 구간의 서울 기온 변화를 분석할 수 있습니다.
        예를 들어, 최근 5년 동안 **새벽 0~6시**, **오후 12~18시**, **특정 월/계절**의 기온 패턴을 따로 볼 수 있습니다.
        """
    )

    weather_analysis = pipeline["weather"].copy()
    weather_analysis["일시"] = pd.to_datetime(weather_analysis["일시"])
    weather_analysis["Date"] = weather_analysis["일시"].dt.date
    weather_analysis["Hour"] = weather_analysis["일시"].dt.hour
    weather_analysis["Month"] = weather_analysis["일시"].dt.month
    weather_analysis["Year"] = weather_analysis["일시"].dt.year

    min_date = weather_analysis["Date"].min()
    max_date = weather_analysis["Date"].max()

    c1, c2, c3 = st.columns([1.2, 1.2, 1])

    with c1:
        selected_dates = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="분석할 날짜 범위를 선택하세요."
        )

    with c2:
        selected_hours = st.slider(
            "Hour range",
            min_value=0,
            max_value=23,
            value=(0, 23),
            help="분석할 시간대를 선택하세요. 예: 0~6시는 새벽 시간대입니다."
        )

    with c3:
        aggregation = st.selectbox(
            "Aggregation",
            ["Hourly records", "Daily average", "Monthly average", "Yearly average"],
            help="그래프에 표시할 집계 단위를 선택하세요."
        )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = selected_dates
        end_date = selected_dates

    start_hour, end_hour = selected_hours

    filtered = weather_analysis[
        (weather_analysis["Date"] >= start_date)
        & (weather_analysis["Date"] <= end_date)
        & (weather_analysis["Hour"] >= start_hour)
        & (weather_analysis["Hour"] <= end_hour)
    ].copy()

    if filtered.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다. 날짜 범위나 시간대를 다시 선택하세요.")
    else:
        temp_col = "기온(°C)"

        avg_temp = filtered[temp_col].mean()
        min_temp = filtered[temp_col].min()
        max_temp = filtered[temp_col].max()
        std_temp = filtered[temp_col].std()
        record_count = len(filtered)

        min_row = filtered.loc[filtered[temp_col].idxmin()]
        max_row = filtered.loc[filtered[temp_col].idxmax()]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Average Temp", f"{avg_temp:.2f} °C")
        m2.metric("Min Temp", f"{min_temp:.2f} °C")
        m3.metric("Max Temp", f"{max_temp:.2f} °C")
        m4.metric("Std Dev", f"{std_temp:.2f} °C")
        m5.metric("Records", f"{record_count:,}")

        st.caption(
            f"Lowest: {min_row['일시']} / {min_row[temp_col]:.2f} °C  |  "
            f"Highest: {max_row['일시']} / {max_row[temp_col]:.2f} °C"
        )

        # Time-series aggregation
        if aggregation == "Hourly records":
            plot_df = filtered[["일시", temp_col]].copy()
            plot_df = plot_df.rename(columns={"일시": "Time", temp_col: "Temperature"})
            fig = px.line(
                plot_df,
                x="Time",
                y="Temperature",
                title=f"Temperature Trend: {start_date} to {end_date}, {start_hour}:00–{end_hour}:00",
                labels={"Temperature": "Temperature (°C)", "Time": "Time"}
            )

        elif aggregation == "Daily average":
            plot_df = filtered.groupby("Date", as_index=False)[temp_col].mean()
            plot_df = plot_df.rename(columns={"Date": "Time", temp_col: "Temperature"})
            fig = px.line(
                plot_df,
                x="Time",
                y="Temperature",
                title=f"Daily Average Temperature: {start_hour}:00–{end_hour}:00",
                labels={"Temperature": "Temperature (°C)", "Time": "Date"}
            )

        elif aggregation == "Monthly average":
            filtered["YearMonth"] = filtered["일시"].dt.to_period("M").astype(str)
            plot_df = filtered.groupby("YearMonth", as_index=False)[temp_col].mean()
            plot_df = plot_df.rename(columns={"YearMonth": "Time", temp_col: "Temperature"})
            fig = px.line(
                plot_df,
                x="Time",
                y="Temperature",
                title=f"Monthly Average Temperature: {start_hour}:00–{end_hour}:00",
                labels={"Temperature": "Temperature (°C)", "Time": "Month"}
            )

        else:
            plot_df = filtered.groupby("Year", as_index=False)[temp_col].mean()
            plot_df = plot_df.rename(columns={"Year": "Time", temp_col: "Temperature"})
            fig = px.bar(
                plot_df,
                x="Time",
                y="Temperature",
                title=f"Yearly Average Temperature: {start_hour}:00–{end_hour}:00",
                labels={"Temperature": "Temperature (°C)", "Time": "Year"}
            )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Temperature by Hour")
        hourly_avg = filtered.groupby("Hour", as_index=False)[temp_col].mean()
        hourly_avg = hourly_avg.rename(columns={temp_col: "Average_Temperature"})

        fig_hour = px.bar(
            hourly_avg,
            x="Hour",
            y="Average_Temperature",
            title="Average Temperature by Selected Hour Range",
            labels={"Average_Temperature": "Average Temperature (°C)", "Hour": "Hour of Day"}
        )
        st.plotly_chart(fig_hour, use_container_width=True)

        st.subheader("Temperature Distribution")
        fig_hist = px.histogram(
            filtered,
            x=temp_col,
            nbins=40,
            title="Temperature Distribution in Selected Range",
            labels={temp_col: "Temperature (°C)"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Filtered Data")
        preview_cols = ["일시", "기온(°C)", "강수량(mm)", "풍속(m/s)", "습도(%)", "해면기압(hPa)"]
        st.dataframe(filtered[preview_cols].tail(200), use_container_width=True)

        csv_data = filtered[preview_cols].to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Download filtered data as CSV",
            data=csv_data,
            file_name="custom_temperature_analysis.csv",
            mime="text/csv"
        )


with tab_data:
    st.subheader("GitHub Data Files")

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    st.write([os.path.basename(f) for f in csv_files])

    st.subheader("Training/Test Prediction Preview")
    st.dataframe(prediction_df.head(100), use_container_width=True)

    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True)
