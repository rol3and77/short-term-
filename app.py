import os
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Streamlit 기본 설정
# ============================================================

st.set_page_config(
    page_title="Seoul Weather ML Dashboard",
    page_icon="🌤️",
    layout="wide",
)

# ============================================================
# 경로 설정
# ============================================================

RESULT_DIR = "result"

MODEL_PATH = os.path.join(RESULT_DIR, "seoul_temperature_model.joblib")
PERFORMANCE_PATH = os.path.join(RESULT_DIR, "model_performance_comparison.csv")
PREDICTION_PATH = os.path.join(RESULT_DIR, "temperature_prediction_result.csv")
SUMMARY_PATH = os.path.join(RESULT_DIR, "project_summary.csv")
FEATURE_PATH = os.path.join(RESULT_DIR, "feature_importance.csv")

API_URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
SEOUL_STATION_ID = "108"

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


# ============================================================
# 유틸 함수
# ============================================================

def make_season(month: int) -> int:
    """봄=0, 여름=1, 가을=2, 겨울=3"""
    if month in [3, 4, 5]:
        return 0
    if month in [6, 7, 8]:
        return 1
    if month in [9, 10, 11]:
        return 2
    return 3


def add_features(df: pd.DataFrame, predict_hour: int = 1):
    """학습 때와 동일한 파생변수를 생성한다."""
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
    df["temp_24h_ago"] = df["기온(°C)"].shift(24)

    df["humidity_1h_ago"] = df["습도(%)"].shift(1)
    df["pressure_1h_ago"] = df["해면기압(hPa)"].shift(1)
    df["wind_1h_ago"] = df["풍속(m/s)"].shift(1)

    df["temp_diff_1h"] = df["기온(°C)"] - df["temp_1h_ago"]
    df["temp_diff_3h"] = df["기온(°C)"] - df["temp_3h_ago"]
    df["pressure_diff_1h"] = df["해면기압(hPa)"] - df["pressure_1h_ago"]

    df["rain_yesno"] = np.where(df["강수량(mm)"] > 0, 1, 0)

    target_column = f"target_temp_{predict_hour}h"
    df[target_column] = df["기온(°C)"].shift(-predict_hour)

    return df, target_column


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


def make_api_params(service_key: str):
    """
    ASOS 시간자료 API는 전날 자료까지 제공되므로
    어제 23시를 종료 시각으로 설정한다.
    """
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
    """공공데이터포털 ASOS API를 안정적으로 요청한다."""
    params, start_time, end_time = make_api_params(service_key)
    last_error = None

    for attempt in range(1, retry + 1):
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
    """API 컬럼명을 학습 데이터 컬럼명과 동일하게 변환한다."""
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


def predict_latest(api_weather: pd.DataFrame, model_bundle: dict) -> dict:
    """API 최신 시점 기준 1시간 뒤 기온을 예측한다."""
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    if len(api_weather) < 25:
        raise ValueError("최근 24시간 전 기온 변수를 만들기 위해 최소 25개 이상의 시간 자료가 필요합니다.")

    latest = api_weather.iloc[-1].copy()
    current_time = latest["일시"]

    current_temp = latest["기온(°C)"]
    current_rainfall = latest["강수량(mm)"]
    current_wind = latest["풍속(m/s)"]
    current_humidity = latest["습도(%)"]
    current_local_pressure = latest["현지기압(hPa)"]
    current_sea_pressure = latest["해면기압(hPa)"]

    temp_1h_ago = api_weather.iloc[-2]["기온(°C)"]
    temp_3h_ago = api_weather.iloc[-4]["기온(°C)"]
    temp_6h_ago = api_weather.iloc[-7]["기온(°C)"]
    temp_24h_ago = api_weather.iloc[-25]["기온(°C)"]

    humidity_1h_ago = api_weather.iloc[-2]["습도(%)"]
    pressure_1h_ago = api_weather.iloc[-2]["해면기압(hPa)"]
    wind_1h_ago = api_weather.iloc[-2]["풍속(m/s)"]

    current_hour = current_time.hour
    current_month = current_time.month
    current_season = make_season(current_month)

    hour_sin = np.sin(2 * np.pi * current_hour / 24)
    hour_cos = np.cos(2 * np.pi * current_hour / 24)

    month_sin = np.sin(2 * np.pi * current_month / 12)
    month_cos = np.cos(2 * np.pi * current_month / 12)

    temp_diff_1h = current_temp - temp_1h_ago
    temp_diff_3h = current_temp - temp_3h_ago
    pressure_diff_1h = current_sea_pressure - pressure_1h_ago

    rain_yesno = 1 if current_rainfall > 0 else 0

    current_input = pd.DataFrame(
        [
            {
                "기온(°C)": current_temp,
                "강수량(mm)": current_rainfall,
                "풍속(m/s)": current_wind,
                "습도(%)": current_humidity,
                "현지기압(hPa)": current_local_pressure,
                "해면기압(hPa)": current_sea_pressure,
                "rain_yesno": rain_yesno,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "season": current_season,
                "temp_1h_ago": temp_1h_ago,
                "temp_3h_ago": temp_3h_ago,
                "temp_6h_ago": temp_6h_ago,
                "temp_24h_ago": temp_24h_ago,
                "humidity_1h_ago": humidity_1h_ago,
                "pressure_1h_ago": pressure_1h_ago,
                "wind_1h_ago": wind_1h_ago,
                "temp_diff_1h": temp_diff_1h,
                "temp_diff_3h": temp_diff_3h,
                "pressure_diff_1h": pressure_diff_1h,
            }
        ]
    )

    current_input = current_input[feature_columns]
    predicted_temp = model.predict(current_input)[0]

    return {
        "base_time": current_time,
        "target_time": current_time + pd.Timedelta(hours=1),
        "current_temp": current_temp,
        "predicted_temp": predicted_temp,
    }


def compare_api_interval(api_weather: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    """API로 받은 구간 내에서 실제 1시간 뒤 기온과 예측값을 비교한다."""
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    predict_hour = model_bundle.get("predict_hour", 1)

    api_compare, api_target_column = add_features(api_weather, predict_hour=predict_hour)
    api_compare_model = api_compare.dropna().reset_index(drop=True)

    X_api = api_compare_model[feature_columns]

    api_compare_model["predicted_temp_1h_later"] = model.predict(X_api)
    api_compare_model["actual_temp_1h_later"] = api_compare_model[api_target_column]
    api_compare_model["prediction_time"] = api_compare_model["일시"] + pd.Timedelta(hours=predict_hour)

    api_compare_model["error"] = (
        api_compare_model["actual_temp_1h_later"]
        - api_compare_model["predicted_temp_1h_later"]
    )
    api_compare_model["abs_error"] = np.abs(api_compare_model["error"])

    return api_compare_model[
        [
            "일시",
            "prediction_time",
            "기온(°C)",
            "actual_temp_1h_later",
            "predicted_temp_1h_later",
            "error",
            "abs_error",
        ]
    ].copy()


def ensure_required_files():
    missing = []
    required = [
        MODEL_PATH,
        PERFORMANCE_PATH,
        PREDICTION_PATH,
        SUMMARY_PATH,
    ]

    for path in required:
        if not os.path.exists(path):
            missing.append(path)

    return missing


# ============================================================
# UI
# ============================================================

st.title("🌤️ Seoul Weather ML Dashboard")
st.markdown(
    """
    기상청 서울 ASOS 108번 지점의 5년치 시간별 관측자료를 기반으로  
    **1시간 뒤 서울 기온을 예측**하는 머신러닝 대시보드입니다.
    """
)

missing_files = ensure_required_files()

if missing_files:
    st.error("필수 파일이 없습니다. Colab 학습 코드를 먼저 실행한 뒤 result 폴더 파일을 GitHub에 업로드하세요.")
    st.code("\n".join(missing_files))
    st.stop()

model_bundle = load_model_bundle()
model_name = model_bundle.get("model_name", "Unknown Model")

summary_df = load_csv(SUMMARY_PATH)
summary = summary_df.iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Model", model_name)
col2.metric("Prediction Target", "1 hour later")
col3.metric("MAE", f"{summary['mae']:.3f} °C")
col4.metric("RMSE", f"{summary['rmse']:.3f} °C")

st.divider()

tab_overview, tab_results, tab_api, tab_data = st.tabs(
    ["Overview", "Model Results", "API Prediction", "Data Preview"]
)


# ============================================================
# Overview
# ============================================================

with tab_overview:
    st.subheader("Project Overview")

    st.markdown(
        """
        이 프로젝트는 단순히 모델을 한 번 학습하는 데서 끝나지 않고,
        데이터 수집, 전처리, 파생변수 생성, 기준 모델 비교, 머신러닝 모델 학습,
        그리고 API 기반 최신 관측자료 예측까지 하나의 파이프라인으로 구성했습니다.

        **핵심 구성**
        - 기상청 서울 ASOS 108번 지점 시간별 관측자료 사용
        - 5년치 기상자료 기반 학습
        - Persistence Baseline과 머신러닝 모델 비교
        - Linear Regression, Random Forest, Gradient Boosting 비교
        - Streamlit 기반 웹 대시보드 구현
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline RMSE", f"{summary['baseline_rmse']:.3f} °C")
    c2.metric("Best Model RMSE", f"{summary['rmse']:.3f} °C")
    c3.metric("R²", f"{summary['r2']:.4f}")

    st.info(
        "ASOS 시간자료 API는 실시간 현재 자료가 아니라 전날 자료까지 제공됩니다. "
        "따라서 API 예측은 'API에서 제공되는 최신 관측 시각 기준 1시간 뒤 예측'으로 해석해야 합니다."
    )


# ============================================================
# Model Results
# ============================================================

with tab_results:
    st.subheader("Model Performance Comparison")

    performance_df = load_csv(PERFORMANCE_PATH)
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

    prediction_df = load_csv(PREDICTION_PATH)
    prediction_df["일시"] = pd.to_datetime(prediction_df["일시"])

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=prediction_df["일시"],
            y=prediction_df["실제_기온"],
            mode="lines",
            name="Actual Temperature",
        )
    )
    fig_line.add_trace(
        go.Scatter(
            x=prediction_df["일시"],
            y=prediction_df["예측_기온"],
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
        x="실제_기온",
        y="예측_기온",
        title="Actual vs Predicted Scatter Plot",
        labels={"실제_기온": "Actual Temperature (°C)", "예측_기온": "Predicted Temperature (°C)"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    if os.path.exists(FEATURE_PATH):
        st.subheader("Feature Importance")
        feature_df = load_csv(FEATURE_PATH)
        feature_df = feature_df.sort_values("Importance", ascending=False).head(15)

        fig_feature = px.bar(
            feature_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Feature Importance",
        )
        fig_feature.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_feature, use_container_width=True)


# ============================================================
# API Prediction
# ============================================================

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

                    latest_result = predict_latest(api_weather, model_bundle)
                    compare_result = compare_api_interval(api_weather, model_bundle)

                    api_mae = mean_absolute_error(
                        compare_result["actual_temp_1h_later"],
                        compare_result["predicted_temp_1h_later"],
                    )
                    api_rmse = np.sqrt(
                        mean_squared_error(
                            compare_result["actual_temp_1h_later"],
                            compare_result["predicted_temp_1h_later"],
                        )
                    )
                    api_r2 = r2_score(
                        compare_result["actual_temp_1h_later"],
                        compare_result["predicted_temp_1h_later"],
                    )

                st.success("API 예측 완료")

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Base Time", str(latest_result["base_time"]))
                a2.metric("Target Time", str(latest_result["target_time"]))
                a3.metric("Current Temp", f"{latest_result['current_temp']:.1f} °C")
                a4.metric("Predicted Temp", f"{latest_result['predicted_temp']:.2f} °C")

                m1, m2, m3 = st.columns(3)
                m1.metric("API MAE", f"{api_mae:.3f} °C")
                m2.metric("API RMSE", f"{api_rmse:.3f} °C")
                m3.metric("API R²", f"{api_r2:.4f}")

                fig_api = go.Figure()
                fig_api.add_trace(
                    go.Scatter(
                        x=compare_result["prediction_time"],
                        y=compare_result["actual_temp_1h_later"],
                        mode="lines+markers",
                        name="Actual",
                    )
                )
                fig_api.add_trace(
                    go.Scatter(
                        x=compare_result["prediction_time"],
                        y=compare_result["predicted_temp_1h_later"],
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

                st.subheader("API Comparison Table")
                st.dataframe(compare_result.tail(30), use_container_width=True)

            except Exception as e:
                st.error(f"API prediction failed: {e}")


# ============================================================
# Data Preview
# ============================================================

with tab_data:
    st.subheader("Generated Result Files")

    if os.path.exists(RESULT_DIR):
        files = sorted(os.listdir(RESULT_DIR))
        st.write(files)
    else:
        st.warning("result 폴더가 없습니다.")

    st.subheader("Prediction Result Preview")
    prediction_df = load_csv(PREDICTION_PATH)
    st.dataframe(prediction_df.head(100), use_container_width=True)
