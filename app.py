# ============================================================
# app.py
# Fast Streamlit Dashboard
# Loads pre-trained model/result files from result/
# ============================================================

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


st.set_page_config(
    page_title="Seoul Weather ML Dashboard",
    page_icon="🌤️",
    layout="wide",
)


# ============================================================
# Paths
# ============================================================

RESULT_DIR = "result"

MODEL_PATH = os.path.join(RESULT_DIR, "seoul_temperature_model.joblib")
PERFORMANCE_PATH = os.path.join(RESULT_DIR, "model_performance_comparison.csv")
PREDICTION_PATH = os.path.join(RESULT_DIR, "temperature_prediction_result.csv")
SUMMARY_PATH = os.path.join(RESULT_DIR, "project_summary.csv")
FEATURE_PATH = os.path.join(RESULT_DIR, "feature_importance.csv")
PROCESSED_PATH = os.path.join(RESULT_DIR, "seoul_weather_processed_dataset.csv")

API_URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
SEOUL_STATION_ID = "108"
PREDICT_HOUR = 1

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
# Utility
# ============================================================

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


def make_season(month: int) -> int:
    if month in [3, 4, 5]:
        return 0
    if month in [6, 7, 8]:
        return 1
    if month in [9, 10, 11]:
        return 2
    return 3


def add_features(df: pd.DataFrame, predict_hour: int = 1):
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


def ensure_required_files():
    required = [
        MODEL_PATH,
        PERFORMANCE_PATH,
        PREDICTION_PATH,
        SUMMARY_PATH,
        FEATURE_PATH,
        PROCESSED_PATH,
    ]
    return [path for path in required if not os.path.exists(path)]


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


def predict_latest_from_api(api_weather: pd.DataFrame, model_bundle: dict):
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    predict_hour = model_bundle.get("predict_hour", 1)

    if len(api_weather) < 49:
        raise ValueError("At least 49 hourly records are required because the model uses 48-hour lag features.")

    api_featured, _, _ = add_features(api_weather, predict_hour=predict_hour)
    api_featured = api_featured.dropna().reset_index(drop=True)

    latest = api_featured.iloc[-1].copy()

    current_time = latest["일시"]
    current_temp = latest["기온(°C)"]

    X_latest = pd.DataFrame([latest[feature_columns]])
    predicted_change = model.predict(X_latest)[0]
    predicted_temp = current_temp + predicted_change

    return {
        "base_time": current_time,
        "target_time": current_time + pd.Timedelta(hours=predict_hour),
        "current_temp": current_temp,
        "predicted_change": predicted_change,
        "predicted_temp": predicted_temp,
    }


def compare_api_interval(api_weather: pd.DataFrame, model_bundle: dict):
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    predict_hour = model_bundle.get("predict_hour", 1)

    api_compare, _, actual_temp_col = add_features(api_weather, predict_hour=predict_hour)
    api_compare_model = api_compare.dropna().reset_index(drop=True)

    X_api = api_compare_model[feature_columns]
    predicted_change = model.predict(X_api)
    predicted_temp = api_compare_model["기온(°C)"].values + predicted_change

    compare_result = pd.DataFrame({
        "Base_Time": api_compare_model["일시"],
        "Prediction_Time": api_compare_model["일시"] + pd.Timedelta(hours=predict_hour),
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

missing_files = ensure_required_files()

if missing_files:
    st.error("필수 result 파일이 없습니다. GitHub Actions 학습을 먼저 실행하세요.")
    st.code("\n".join(missing_files))
    st.stop()

model_bundle = load_model_bundle()
performance_df = load_csv(PERFORMANCE_PATH)
prediction_df = load_csv(PREDICTION_PATH)
summary_df = load_csv(SUMMARY_PATH)
feature_importance_df = load_csv(FEATURE_PATH)
processed_df = load_csv(PROCESSED_PATH)

prediction_df["Time"] = pd.to_datetime(prediction_df["Time"])
processed_df["일시"] = pd.to_datetime(processed_df["일시"])
summary = summary_df.iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Deploy Model", summary["deploy_model"])
col2.metric("Prediction Target", "1 hour later")
col3.metric("MAE", f"{summary['mae']:.3f} °C")
col4.metric("RMSE", f"{summary['rmse']:.3f} °C")

st.divider()

tab_overview, tab_results, tab_api, tab_forecast, tab_analysis, tab_data = st.tabs(
    ["Overview", "Model Results", "API Prediction", "Forecast Comparison", "Custom Analysis", "Data Preview"]
)


with tab_overview:
    st.subheader("Project Overview")

    st.markdown(
        """
        **구성**
        - GitHub Actions가 `data/` 폴더의 CSV를 읽고 자동 학습
        - Streamlit은 `result/` 폴더의 모델과 결과 파일만 읽어 빠르게 실행
        - 직접 기온을 예측하는 대신, **1시간 뒤 기온 변화량**을 예측
        - 최종 기온 = 현재 기온 + 예측 변화량
        - Baseline과 머신러닝 모델 성능 비교
        - API 최신 제공 자료 기반 예측 지원
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline RMSE", f"{summary['baseline_rmse']:.3f} °C")
    c2.metric("Deploy Model RMSE", f"{summary['rmse']:.3f} °C")
    c3.metric("R²", f"{summary['r2']:.4f}")

    st.info(
        f"Performance-best model: {summary['best_model']} / "
        f"Deployment model: {summary['deploy_model']}"
    )

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
    fig_line.add_trace(go.Scatter(x=prediction_df["Time"], y=prediction_df["Actual_Temperature"], mode="lines", name="Actual Temperature"))
    fig_line.add_trace(go.Scatter(x=prediction_df["Time"], y=prediction_df["Predicted_Temperature"], mode="lines", name="Predicted Temperature"))
    fig_line.update_layout(title="Actual vs Predicted Temperature", xaxis_title="Time", yaxis_title="Temperature (°C)")
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

    st.subheader("Feature Importance")
    top_features = feature_importance_df.sort_values("Importance", ascending=False).head(15)
    fig_feature = px.bar(top_features, x="Importance", y="Feature_English", orientation="h", title="Top 15 Feature Importance")
    fig_feature.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_feature, use_container_width=True)

    no_current_temp = feature_importance_df[
        feature_importance_df["Feature"] != "기온(°C)"
    ].sort_values("Importance", ascending=False).head(15)

    st.subheader("Feature Importance Except Current Temperature")
    fig_no_temp = px.bar(no_current_temp, x="Importance", y="Feature_English", orientation="h", title="Feature Importance Except Current Temperature")
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

                    latest_result = predict_latest_from_api(api_weather, model_bundle)
                    compare_result = compare_api_interval(api_weather, model_bundle)

                    api_mae = mean_absolute_error(compare_result["Actual_Temperature"], compare_result["Predicted_Temperature"])
                    api_rmse = np.sqrt(mean_squared_error(compare_result["Actual_Temperature"], compare_result["Predicted_Temperature"]))
                    api_r2 = r2_score(compare_result["Actual_Temperature"], compare_result["Predicted_Temperature"])

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
                fig_api.add_trace(go.Scatter(x=compare_result["Prediction_Time"], y=compare_result["Actual_Temperature"], mode="lines+markers", name="Actual"))
                fig_api.add_trace(go.Scatter(x=compare_result["Prediction_Time"], y=compare_result["Predicted_Temperature"], mode="lines+markers", name="Predicted"))
                fig_api.update_layout(title="API Interval: Actual vs Predicted", xaxis_title="Time", yaxis_title="Temperature (°C)")
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


with tab_forecast:
    st.subheader("Historical Forecast Comparison")
    st.markdown(
        """
        과거 데이터의 특정 기간을 선택하면 해당 시점마다 모델이 **1시간 뒤 기온을 예측**하고 실제 관측값과 비교합니다.  
        즉, 과거 데이터를 이용한 **미래 예측 시뮬레이션(backtesting)** 기능입니다.
        """
    )

    forecast_df = processed_df.copy()
    forecast_df["Date"] = forecast_df["일시"].dt.date
    forecast_df["Hour"] = forecast_df["일시"].dt.hour

    min_date_fc = forecast_df["Date"].min()
    max_date_fc = forecast_df["Date"].max()

    fc1, fc2, fc3 = st.columns([1.4, 1.2, 1])

    with fc1:
        default_start = max(min_date_fc, max_date_fc - timedelta(days=7))
        forecast_dates = st.date_input(
            "Forecast comparison date range",
            value=(default_start, max_date_fc),
            min_value=min_date_fc,
            max_value=max_date_fc,
        )

    with fc2:
        forecast_hours = st.slider("Base hour range", 0, 23, (0, 23))

    with fc3:
        max_points = st.number_input("Max records", 50, 5000, 1000, 50)

    if isinstance(forecast_dates, tuple) and len(forecast_dates) == 2:
        fc_start_date, fc_end_date = forecast_dates
    else:
        fc_start_date = forecast_dates
        fc_end_date = forecast_dates

    fc_start_hour, fc_end_hour = forecast_hours

    forecast_sample = forecast_df[
        (forecast_df["Date"] >= fc_start_date)
        & (forecast_df["Date"] <= fc_end_date)
        & (forecast_df["Hour"] >= fc_start_hour)
        & (forecast_df["Hour"] <= fc_end_hour)
    ].copy()

    if forecast_sample.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    else:
        if len(forecast_sample) > max_points:
            forecast_sample = forecast_sample.tail(int(max_points)).copy()

        model = model_bundle["model"]
        feature_columns = model_bundle["feature_columns"]

        X_fc = forecast_sample[feature_columns]
        predicted_change = model.predict(X_fc)
        predicted_temp = forecast_sample["기온(°C)"].values + predicted_change

        actual_col = "actual_temp_1h_later"

        forecast_result = pd.DataFrame({
            "Base_Time": forecast_sample["일시"].values,
            "Prediction_Time": forecast_sample["일시"].values + np.array([np.timedelta64(PREDICT_HOUR, "h")] * len(forecast_sample)),
            "Current_Temperature": forecast_sample["기온(°C)"].values,
            "Actual_Future_Temperature": forecast_sample[actual_col].values,
            "Predicted_Future_Temperature": predicted_temp,
            "Predicted_Change": predicted_change,
        })

        forecast_result["Error"] = forecast_result["Actual_Future_Temperature"] - forecast_result["Predicted_Future_Temperature"]
        forecast_result["Absolute_Error"] = np.abs(forecast_result["Error"])

        fc_mae = mean_absolute_error(forecast_result["Actual_Future_Temperature"], forecast_result["Predicted_Future_Temperature"])
        fc_rmse = np.sqrt(mean_squared_error(forecast_result["Actual_Future_Temperature"], forecast_result["Predicted_Future_Temperature"]))
        fc_r2 = r2_score(forecast_result["Actual_Future_Temperature"], forecast_result["Predicted_Future_Temperature"])

        fcm1, fcm2, fcm3, fcm4 = st.columns(4)
        fcm1.metric("MAE", f"{fc_mae:.3f} °C")
        fcm2.metric("RMSE", f"{fc_rmse:.3f} °C")
        fcm3.metric("R²", f"{fc_r2:.4f}")
        fcm4.metric("Records", f"{len(forecast_result):,}")

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=forecast_result["Prediction_Time"], y=forecast_result["Actual_Future_Temperature"], mode="lines", name="Actual Future Temperature"))
        fig_fc.add_trace(go.Scatter(x=forecast_result["Prediction_Time"], y=forecast_result["Predicted_Future_Temperature"], mode="lines", name="Predicted Future Temperature"))
        fig_fc.update_layout(title="Forecast Comparison: Actual vs Predicted Future Temperature", xaxis_title="Prediction Time", yaxis_title="Temperature (°C)")
        st.plotly_chart(fig_fc, use_container_width=True)

        fig_fc_scatter = px.scatter(
            forecast_result,
            x="Actual_Future_Temperature",
            y="Predicted_Future_Temperature",
            title="Forecast Comparison Scatter Plot",
            labels={
                "Actual_Future_Temperature": "Actual Future Temperature (°C)",
                "Predicted_Future_Temperature": "Predicted Future Temperature (°C)",
            },
        )
        st.plotly_chart(fig_fc_scatter, use_container_width=True)

        fig_fc_error = px.line(forecast_result, x="Prediction_Time", y="Error", title="Forecast Error Over Time")
        fig_fc_error.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_fc_error, use_container_width=True)

        st.subheader("Forecast Comparison Table")
        st.dataframe(forecast_result.tail(300), use_container_width=True)

        csv_forecast = forecast_result.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Download forecast comparison as CSV",
            data=csv_forecast,
            file_name="forecast_comparison.csv",
            mime="text/csv",
        )


with tab_analysis:
    st.subheader("Custom Temperature Analysis")
    st.markdown(
        """
        원하는 날짜 범위와 시간대를 선택하면 해당 구간의 서울 기온 변화를 분석할 수 있습니다.
        """
    )

    weather_analysis = processed_df.copy()
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
        )

    with c2:
        selected_hours = st.slider("Hour range", 0, 23, (0, 23))

    with c3:
        aggregation = st.selectbox(
            "Aggregation",
            ["Hourly records", "Daily average", "Monthly average", "Yearly average"],
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
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    else:
        temp_col = "기온(°C)"

        avg_temp = filtered[temp_col].mean()
        min_temp = filtered[temp_col].min()
        max_temp = filtered[temp_col].max()
        std_temp = filtered[temp_col].std()

        min_row = filtered.loc[filtered[temp_col].idxmin()]
        max_row = filtered.loc[filtered[temp_col].idxmax()]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Average Temp", f"{avg_temp:.2f} °C")
        m2.metric("Min Temp", f"{min_temp:.2f} °C")
        m3.metric("Max Temp", f"{max_temp:.2f} °C")
        m4.metric("Std Dev", f"{std_temp:.2f} °C")
        m5.metric("Records", f"{len(filtered):,}")

        st.caption(
            f"Lowest: {min_row['일시']} / {min_row[temp_col]:.2f} °C  |  "
            f"Highest: {max_row['일시']} / {max_row[temp_col]:.2f} °C"
        )

        if aggregation == "Hourly records":
            plot_df = filtered[["일시", temp_col]].copy()
            plot_df = plot_df.rename(columns={"일시": "Time", temp_col: "Temperature"})
            fig = px.line(plot_df, x="Time", y="Temperature", title="Temperature Trend")
        elif aggregation == "Daily average":
            plot_df = filtered.groupby("Date", as_index=False)[temp_col].mean()
            plot_df = plot_df.rename(columns={"Date": "Time", temp_col: "Temperature"})
            fig = px.line(plot_df, x="Time", y="Temperature", title="Daily Average Temperature")
        elif aggregation == "Monthly average":
            filtered["YearMonth"] = filtered["일시"].dt.to_period("M").astype(str)
            plot_df = filtered.groupby("YearMonth", as_index=False)[temp_col].mean()
            plot_df = plot_df.rename(columns={"YearMonth": "Time", temp_col: "Temperature"})
            fig = px.line(plot_df, x="Time", y="Temperature", title="Monthly Average Temperature")
        else:
            plot_df = filtered.groupby("Year", as_index=False)[temp_col].mean()
            plot_df = plot_df.rename(columns={"Year": "Time", temp_col: "Temperature"})
            fig = px.bar(plot_df, x="Time", y="Temperature", title="Yearly Average Temperature")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Temperature by Hour")
        hourly_avg = filtered.groupby("Hour", as_index=False)[temp_col].mean()
        hourly_avg = hourly_avg.rename(columns={temp_col: "Average_Temperature"})
        fig_hour = px.bar(hourly_avg, x="Hour", y="Average_Temperature", title="Average Temperature by Hour")
        st.plotly_chart(fig_hour, use_container_width=True)

        st.subheader("Temperature Distribution")
        fig_hist = px.histogram(filtered, x=temp_col, nbins=40, title="Temperature Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Filtered Data")
        preview_cols = ["일시", "기온(°C)", "강수량(mm)", "풍속(m/s)", "습도(%)", "해면기압(hPa)"]
        st.dataframe(filtered[preview_cols].tail(200), use_container_width=True)

        csv_data = filtered[preview_cols].to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Download filtered data as CSV",
            data=csv_data,
            file_name="custom_temperature_analysis.csv",
            mime="text/csv",
        )


with tab_data:
    st.subheader("Generated Result Files")

    if os.path.exists(RESULT_DIR):
        files = sorted(os.listdir(RESULT_DIR))
        st.write(files)
    else:
        st.warning("result 폴더가 없습니다.")

    st.subheader("Prediction Result Preview")
    st.dataframe(prediction_df.head(100), use_container_width=True)

    st.subheader("Processed Dataset Preview")
    st.dataframe(processed_df.head(100), use_container_width=True)
