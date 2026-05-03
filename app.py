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

def load_css(file_path: str):
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

def apply_english_chart_style(fig, title="", x_title="", y_title=""):
    """
    Apply consistent English chart typography.
    The dashboard explanations remain Korean, but all chart-internal text
    is kept in English to prevent font rendering issues.
    """
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title_text="",
        font=dict(
            family="Inter, Arial, sans-serif",
            size=13,
            color="#171717"
        ),
        title_font=dict(size=20, family="Inter, Arial, sans-serif"),
        hoverlabel=dict(
            font=dict(family="Inter, Arial, sans-serif")
        )
    )
    return fig



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
CV_SUMMARY_PATH = os.path.join(RESULT_DIR, "time_series_cv_summary.csv")
CV_RESULTS_PATH = os.path.join(RESULT_DIR, "time_series_cv_results.csv")
DATA_QUALITY_PATH = os.path.join(RESULT_DIR, "data_quality_report.csv")
MISSING_REPORT_PATH = os.path.join(RESULT_DIR, "missing_value_report.csv")
RESIDUAL_SUMMARY_PATH = os.path.join(RESULT_DIR, "residual_analysis_summary.csv")
ERROR_BY_HOUR_PATH = os.path.join(RESULT_DIR, "error_by_hour.csv")
ERROR_BY_MONTH_PATH = os.path.join(RESULT_DIR, "error_by_month.csv")
ERROR_BY_SEASON_PATH = os.path.join(RESULT_DIR, "error_by_season.csv")
ERROR_BY_TEMP_BIN_PATH = os.path.join(RESULT_DIR, "error_by_temperature_bin.csv")
MODEL_CARD_PATH = os.path.join(RESULT_DIR, "model_card.md")

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


@st.cache_data
def load_optional_csv(path: str):
    if os.path.exists(path):
        return pd.read_csv(path, encoding="utf-8-sig")
    return None


def load_optional_text(path: str):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return None


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

st.markdown(
    """
    <section class="weather-hero">
      <div class="hero-copy">
        <div class="hero-kicker">Seoul ASOS · GitHub Actions · Streamlit</div>
        <h1>Forecast Seoul temperature with a developer-grade dashboard.</h1>
        <p>
          5년치 서울 ASOS 시간별 관측자료를 기반으로 1시간 뒤 기온 변화량을 예측하고,
          Future Forecast, API Prediction, Custom Analysis를 한 화면에서 확인하는 머신러닝 대시보드입니다.
        </p>
        <div class="hero-actions">
          <span class="hero-pill primary">Get forecast</span>
          <span class="hero-pill">View model results</span>
          <span class="hero-pill">Analyze custom hours</span>
        </div>
      </div>
      <div class="hero-mockup">
        <div class="mockup-top">
          <div class="mockup-dots">
            <span class="mockup-dot"></span>
            <span class="mockup-dot"></span>
            <span class="mockup-dot"></span>
          </div>
          <span>weather-dashboard.app</span>
        </div>
        <div class="mockup-grid">
          <div class="mockup-panel">
            <div class="mockup-label">Prediction target</div>
            <div class="mockup-value">+1h</div>
          </div>
          <div class="mockup-panel">
            <div class="mockup-label">Station</div>
            <div class="mockup-value">Seoul 108</div>
          </div>
          <div class="mockup-panel">
            <div class="mockup-label">Automation</div>
            <div class="mockup-code">
              <span class="coral">data/*.csv</span> → GitHub Actions<br/>
              train_pipeline.py → result/*.joblib<br/>
              Streamlit → fast dashboard
            </div>
          </div>
          <div class="mockup-panel">
            <div class="mockup-label">Forecast logic</div>
            <div class="mockup-code">
              current temperature<br/>
              + <span class="teal">predicted temperature change</span><br/>
              = future temperature
            </div>
          </div>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
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

cv_summary_df = load_optional_csv(CV_SUMMARY_PATH)
cv_results_df = load_optional_csv(CV_RESULTS_PATH)
data_quality_df = load_optional_csv(DATA_QUALITY_PATH)
missing_report_df = load_optional_csv(MISSING_REPORT_PATH)
residual_summary_df = load_optional_csv(RESIDUAL_SUMMARY_PATH)
error_by_hour_df = load_optional_csv(ERROR_BY_HOUR_PATH)
error_by_month_df = load_optional_csv(ERROR_BY_MONTH_PATH)
error_by_season_df = load_optional_csv(ERROR_BY_SEASON_PATH)
error_by_temp_bin_df = load_optional_csv(ERROR_BY_TEMP_BIN_PATH)
model_card_text = load_optional_text(MODEL_CARD_PATH)

prediction_df["Time"] = pd.to_datetime(prediction_df["Time"])
processed_df["일시"] = pd.to_datetime(processed_df["일시"])
summary = summary_df.iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Deploy Model", summary["deploy_model"])
col2.metric("Prediction Target", "1 hour later")
col3.metric("MAE", f"{summary['mae']:.3f} °C")
col4.metric("RMSE", f"{summary['rmse']:.3f} °C")

st.divider()

tab_overview, tab_results, tab_diagnostics, tab_api, tab_future, tab_analysis, tab_data = st.tabs(
    ["Overview", "Model Results", "Research Diagnostics", "API Prediction", "Future Forecast", "Custom Analysis", "Data Preview"]
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
        - 그래프 내부 텍스트는 글꼴 깨짐을 방지하기 위해 영어로 통일하고, 해석과 안내문은 한국어로 제공
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
    fig_perf = apply_english_chart_style(
        fig_perf,
        title="Model Performance Comparison",
        x_title="Model",
        y_title="Score"
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    st.subheader("Test Data: Actual vs Predicted")

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=prediction_df["Time"], y=prediction_df["Actual_Temperature"], mode="lines", name="Actual Temperature"))
    fig_line.add_trace(go.Scatter(x=prediction_df["Time"], y=prediction_df["Predicted_Temperature"], mode="lines", name="Predicted Temperature"))
    fig_line = apply_english_chart_style(
        fig_line,
        title="Actual vs Predicted Temperature Over Time",
        x_title="Time",
        y_title="Temperature (°C)"
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
    fig_scatter = apply_english_chart_style(
        fig_scatter,
        title="Actual vs Predicted Scatter Plot",
        x_title="Actual Temperature (°C)",
        y_title="Predicted Temperature (°C)"
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
    fig_change = apply_english_chart_style(
        fig_change,
        title="Predicted 1-Hour Temperature Change",
        x_title="Time",
        y_title="Predicted Change (°C)"
    )
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
    fig_error = apply_english_chart_style(
        fig_error,
        title="Error Over Time",
        x_title="Time",
        y_title="Error (Actual - Predicted)"
    )
    st.plotly_chart(fig_error, use_container_width=True)

    st.subheader("Feature Importance")
    top_features = feature_importance_df.sort_values("Importance", ascending=False).head(15)
    fig_feature = px.bar(top_features, x="Importance", y="Feature_English", orientation="h", title="Top 15 Feature Importance")
    fig_feature.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_feature = apply_english_chart_style(
        fig_feature,
        title="Top 15 Feature Importance",
        x_title="Importance",
        y_title="Feature"
    )
    st.plotly_chart(fig_feature, use_container_width=True)

    no_current_temp = feature_importance_df[
        feature_importance_df["Feature"] != "기온(°C)"
    ].sort_values("Importance", ascending=False).head(15)

    st.subheader("Feature Importance Except Current Temperature")
    fig_no_temp = px.bar(no_current_temp, x="Importance", y="Feature_English", orientation="h", title="Feature Importance Except Current Temperature")
    fig_no_temp.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_no_temp = apply_english_chart_style(
        fig_no_temp,
        title="Feature Importance Except Current Temperature",
        x_title="Importance",
        y_title="Feature"
    )
    st.plotly_chart(fig_no_temp, use_container_width=True)



with tab_diagnostics:
    st.subheader("Research Diagnostics")
    st.markdown(
        """
        이 탭은 단순 예측 결과를 넘어서, 데이터 품질·시간 순서 교차검증·잔차 분석·시간대별 오차를 확인하기 위한 연구용 진단 화면입니다.
        """
    )

    if data_quality_df is not None:
        st.markdown("### Data Quality Report")
        st.dataframe(data_quality_df, use_container_width=True)

    if missing_report_df is not None:
        st.markdown("### Missing Value Report")
        st.dataframe(missing_report_df, use_container_width=True)

    if cv_summary_df is not None:
        st.markdown("### Time-Series Cross-Validation Summary")
        st.dataframe(cv_summary_df, use_container_width=True)

        if {"Model", "CV_RMSE_Mean"}.issubset(cv_summary_df.columns):
            fig_cv = px.bar(
                cv_summary_df,
                x="Model",
                y="CV_RMSE_Mean",
                title="Time-Series Cross-Validation RMSE",
                labels={"CV_RMSE_Mean": "CV RMSE Mean (°C)"}
            )
            fig_cv = apply_english_chart_style(
                fig_cv,
                title="Time-Series Cross-Validation RMSE",
                x_title="Model",
                y_title="CV RMSE Mean (°C)"
            )
            st.plotly_chart(fig_cv, use_container_width=True)

    if residual_summary_df is not None:
        st.markdown("### Residual Analysis Summary")
        st.dataframe(residual_summary_df, use_container_width=True)

    if error_by_hour_df is not None:
        st.markdown("### Error by Hour")
        fig_hour_error = go.Figure()
        fig_hour_error.add_trace(
            go.Scatter(
                x=error_by_hour_df["Hour"],
                y=error_by_hour_df["MAE"],
                mode="lines+markers",
                name="MAE"
            )
        )
        fig_hour_error.add_trace(
            go.Scatter(
                x=error_by_hour_df["Hour"],
                y=error_by_hour_df["RMSE"],
                mode="lines+markers",
                name="RMSE"
            )
        )
        fig_hour_error = apply_english_chart_style(
            fig_hour_error,
            title="Prediction Error by Hour",
            x_title="Hour",
            y_title="Error (°C)"
        )
        st.plotly_chart(fig_hour_error, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        if error_by_month_df is not None:
            st.markdown("### Error by Month")
            st.dataframe(error_by_month_df, use_container_width=True)

    with col_b:
        if error_by_season_df is not None:
            st.markdown("### Error by Season")
            st.dataframe(error_by_season_df, use_container_width=True)

    if error_by_temp_bin_df is not None:
        st.markdown("### Error by Temperature Range")
        st.dataframe(error_by_temp_bin_df, use_container_width=True)

    if model_card_text is not None:
        st.markdown("### Model Card")
        st.markdown(model_card_text)

    if cv_summary_df is None:
        st.info(
            "Research diagnostic files are not available yet. "
            "Run the updated GitHub Actions training pipeline to generate them."
        )




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
                fig_api = apply_english_chart_style(
                    fig_api,
                    title="API Interval: Actual vs Predicted",
                    x_title="Time",
                    y_title="Temperature (°C)"
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
                fig_api_change = apply_english_chart_style(
                    fig_api_change,
                    title="API Interval: Predicted Temperature Change",
                    x_title="Prediction Time",
                    y_title="Predicted Change (°C)"
                )
                st.plotly_chart(fig_api_change, use_container_width=True)

                st.subheader("API Comparison Table")
                st.dataframe(compare_result.tail(30), use_container_width=True)

            except Exception as e:
                st.error(f"API prediction failed: {e}")




with tab_future:
    st.subheader("Future Forecast After Latest Uploaded Data")
    st.markdown(
        """
        이 탭은 **업로드된 데이터의 마지막 시점 이후**를 예측합니다.

        - 가까운 미래는 모델을 반복 적용하여 시간별 기온을 시뮬레이션합니다.
        - 먼 미래는 특정 날짜·시간과 비슷한 과거 패턴을 이용해 계절적 예상 기온을 계산합니다.
        - 실제 장기예보가 아니라, 보유한 서울 ASOS 데이터 기반의 통계적 추정입니다.
        """
    )

    def iterative_future_forecast(history_df: pd.DataFrame, model_bundle: dict, target_time: pd.Timestamp, exogenous_mode: str):
        model = model_bundle["model"]
        feature_columns = model_bundle["feature_columns"]
        predict_hour = model_bundle.get("predict_hour", 1)

        future_history = history_df[REQUIRED_COLUMNS].copy()
        future_history["일시"] = pd.to_datetime(future_history["일시"])
        future_history = future_history.sort_values("일시").reset_index(drop=True)

        if len(future_history) < 60:
            raise ValueError("미래 예측을 위해 최소 60시간 이상의 데이터가 필요합니다.")

        latest_time = future_history["일시"].iloc[-1]

        if target_time <= latest_time:
            raise ValueError("예측 대상 시각은 업로드된 데이터의 마지막 시각보다 이후여야 합니다.")

        horizon_hours = int(np.ceil((target_time - latest_time) / pd.Timedelta(hours=1)))

        forecast_rows = []

        for step in range(1, horizon_hours + 1):
            featured, _, _ = add_features(future_history, predict_hour=predict_hour)
            latest_feature_row = featured.iloc[-1].copy()

            if latest_feature_row[feature_columns].isna().sum() > 0:
                raise ValueError("예측에 필요한 파생변수 중 결측값이 있습니다. 데이터 길이를 확인하세요.")

            current_time = latest_feature_row["일시"]
            current_temp = latest_feature_row["기온(°C)"]

            X_latest = pd.DataFrame([latest_feature_row[feature_columns]])
            predicted_change = float(model.predict(X_latest)[0])
            predicted_temp = float(current_temp + predicted_change)

            next_time = current_time + pd.Timedelta(hours=1)
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


    def calendar_temperature_estimate(processed_history: pd.DataFrame, target_time: pd.Timestamp, day_window: int = 14):
        """
        먼 미래 날짜에 대해 같은 월/일/시간대 주변의 과거 관측값을 이용해
        계절적 예상 기온을 계산한다.
        """
        hist = processed_history.copy()
        hist["일시"] = pd.to_datetime(hist["일시"])
        hist["Month"] = hist["일시"].dt.month
        hist["Day"] = hist["일시"].dt.day
        hist["Hour"] = hist["일시"].dt.hour
        hist["DayOfYear"] = hist["일시"].dt.dayofyear

        target_doy = target_time.dayofyear
        target_hour = target_time.hour

        # Circular day-of-year distance, handles year boundary.
        hist["Day_Distance"] = np.minimum(
            np.abs(hist["DayOfYear"] - target_doy),
            366 - np.abs(hist["DayOfYear"] - target_doy)
        )

        sample = hist[
            (hist["Hour"] == target_hour)
            & (hist["Day_Distance"] <= day_window)
        ].copy()

        # If too few rows, widen to +/- 30 days.
        if len(sample) < 20:
            sample = hist[
                (hist["Hour"] == target_hour)
                & (hist["Day_Distance"] <= 30)
            ].copy()

        if sample.empty:
            raise ValueError("선택한 날짜와 시간대에 대응하는 과거 패턴 데이터를 찾지 못했습니다.")

        estimate = {
            "Target_Time": target_time,
            "Estimated_Temperature": float(sample["기온(°C)"].mean()),
            "Median_Temperature": float(sample["기온(°C)"].median()),
            "Min_Similar_Pattern": float(sample["기온(°C)"].min()),
            "Max_Similar_Pattern": float(sample["기온(°C)"].max()),
            "Std_Similar_Pattern": float(sample["기온(°C)"].std()),
            "Sample_Count": int(len(sample)),
            "Day_Window": int(day_window if len(sample) >= 20 else 30),
        }

        return estimate, sample


    history_for_future = processed_df[REQUIRED_COLUMNS].copy()
    history_for_future["일시"] = pd.to_datetime(history_for_future["일시"])
    history_for_future = history_for_future.sort_values("일시").reset_index(drop=True)

    latest_obs = history_for_future.iloc[-1]
    latest_time = pd.to_datetime(latest_obs["일시"])

    st.info(
        f"업로드된 데이터의 마지막 시각은 **{latest_time}** 입니다. "
        "이 시각 이후의 날짜와 시간을 선택해 예측할 수 있습니다."
    )

    mode = st.radio(
        "Forecast mode",
        [
            "Short-term ML simulation",
            "Any-date seasonal estimate"
        ],
        horizontal=True,
        help=(
            "Short-term ML simulation은 마지막 데이터 이후를 시간별로 반복 예측합니다. "
            "Any-date seasonal estimate는 2025년 이후 어떤 날짜라도 과거 유사 날짜 패턴으로 추정합니다."
        )
    )

    if mode == "Short-term ML simulation":
        st.markdown("#### Short-term ML simulation")
        st.caption(
            "마지막 관측값 이후부터 선택한 목표 시각까지 시간별로 반복 예측합니다. "
            "너무 먼 미래까지 반복하면 오차가 누적되므로 1~30일 정도의 단기 시뮬레이션에 적합합니다."
        )

        c1, c2, c3 = st.columns([1.2, 0.8, 1.2])

        with c1:
            target_date = st.date_input(
                "Target date",
                value=(latest_time + pd.Timedelta(hours=12)).date(),
                min_value=(latest_time + pd.Timedelta(hours=1)).date(),
                max_value=(latest_time + pd.Timedelta(days=30)).date(),
                help="반복 ML 예측은 최대 30일 이후까지 선택할 수 있습니다."
            )

        with c2:
            target_hour = st.number_input(
                "Target hour",
                min_value=0,
                max_value=23,
                value=int((latest_time + pd.Timedelta(hours=12)).hour),
                step=1
            )

        with c3:
            exogenous_mode = st.selectbox(
                "Future weather assumption",
                ["Hold last observed", "Recent 6-hour average"],
                help="미래 습도·풍속·기압·강수량을 어떻게 가정할지 선택합니다."
            )

        target_time = pd.Timestamp(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
            hour=int(target_hour)
        )

        if target_time <= latest_time:
            st.warning("예측 대상 시각은 마지막 데이터 시각보다 이후여야 합니다.")
        else:
            horizon_hours = int(np.ceil((target_time - latest_time) / pd.Timedelta(hours=1)))

            h1, h2, h3 = st.columns(3)
            h1.metric("Latest Data Time", str(latest_time))
            h2.metric("Target Time", str(target_time))
            h3.metric("Forecast Horizon", f"{horizon_hours} h")

            try:
                future_result = iterative_future_forecast(
                    history_df=history_for_future,
                    model_bundle=model_bundle,
                    target_time=target_time,
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
                fig_future = apply_english_chart_style(
                    fig_future,
                    title="Future Temperature Forecast After Latest Uploaded Data",
                    x_title="Forecast Time",
                    y_title="Temperature (°C)"
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
                fig_change_future = apply_english_chart_style(
                    fig_change_future,
                    title="Predicted Hourly Temperature Change",
                    x_title="Forecast Time",
                    y_title="Predicted Change (°C)"
                )
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

            except Exception as e:
                st.error(f"Future forecast failed: {e}")

    else:
        st.markdown("#### Any-date seasonal estimate")
        st.caption(
            "2025년 이후 원하는 날짜와 시간을 입력하면, 과거 5년 데이터에서 같은 날짜 주변·같은 시간대의 패턴을 찾아 "
            "계절적 예상 기온을 계산합니다. 장기 실제 날씨 예보가 아니라 과거 패턴 기반 기대값입니다."
        )

        c1, c2, c3 = st.columns([1.2, 0.8, 1])

        with c1:
            target_date = st.date_input(
                "Target date after uploaded data",
                value=max((latest_time + pd.DateOffset(years=1)).date(), latest_time.date()),
                min_value=(latest_time + pd.Timedelta(days=1)).date(),
                max_value=pd.Timestamp("2100-12-31").date(),
                help="2025년 이후 어느 날짜든 선택할 수 있습니다."
            )

        with c2:
            target_hour = st.number_input(
                "Target hour",
                min_value=0,
                max_value=23,
                value=12,
                step=1
            )

        with c3:
            day_window = st.slider(
                "Similar-date window",
                min_value=3,
                max_value=30,
                value=14,
                help="선택한 날짜 전후 며칠까지 유사 날짜로 볼지 정합니다."
            )

        target_time = pd.Timestamp(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
            hour=int(target_hour)
        )

        try:
            estimate, similar_sample = calendar_temperature_estimate(
                processed_history=processed_df,
                target_time=target_time,
                day_window=int(day_window)
            )

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Target Time", str(estimate["Target_Time"]))
            e2.metric("Estimated Temp", f"{estimate['Estimated_Temperature']:.2f} °C")
            e3.metric("Typical Range", f"{estimate['Min_Similar_Pattern']:.1f} ~ {estimate['Max_Similar_Pattern']:.1f} °C")
            e4.metric("Similar Samples", f"{estimate['Sample_Count']}")

            st.markdown(
                f"""
                **해석:** {target_time}의 예상 기온은 과거 유사 날짜·시간대 기준으로 약  
                **{estimate['Estimated_Temperature']:.2f}°C**입니다.  
                표준편차는 **{estimate['Std_Similar_Pattern']:.2f}°C**이므로,
                실제 기온은 기압계·강수·바람 등 당시 조건에 따라 달라질 수 있습니다.
                """
            )

            similar_sample = similar_sample.sort_values("일시").copy()

            fig_pattern = px.scatter(
                similar_sample,
                x="일시",
                y="기온(°C)",
                color="Year" if "Year" in similar_sample.columns else None,
                title="Historical Similar-Date Temperature Samples",
                labels={"기온(°C)": "Temperature (°C)", "일시": "Historical Time"}
            )
            fig_pattern.add_hline(
                y=estimate["Estimated_Temperature"],
                line_dash="dash",
                annotation_text="Estimated temperature"
            )
            fig_pattern = apply_english_chart_style(
                fig_pattern,
                title="Historical Similar-Date Temperature Samples",
                x_title="Historical Time",
                y_title="Temperature (°C)"
            )
            st.plotly_chart(fig_pattern, use_container_width=True)

            st.subheader("Similar Historical Samples")
            display_cols = ["일시", "기온(°C)", "습도(%)", "풍속(m/s)", "강수량(mm)", "해면기압(hPa)"]
            st.dataframe(similar_sample[display_cols].tail(300), use_container_width=True)

            estimate_df = pd.DataFrame([estimate])
            csv_estimate = estimate_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="Download seasonal estimate as CSV",
                data=csv_estimate,
                file_name="any_date_seasonal_temperature_estimate.csv",
                mime="text/csv"
            )

            st.info(
                "이 기능은 장기 실제 예보가 아니라 과거 패턴 기반 추정입니다. "
                "예를 들어 2030년 8월 15일 14시를 선택하면, 과거 데이터에서 8월 15일 주변 날짜의 14시 기온 패턴을 이용해 예상값을 계산합니다."
            )

        except Exception as e:
            st.error(f"Seasonal estimate failed: {e}")



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

        fig = apply_english_chart_style(
            fig,
            title=fig.layout.title.text or "Temperature Trend",
            x_title="Time",
            y_title="Temperature (°C)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Temperature by Hour")
        hourly_avg = filtered.groupby("Hour", as_index=False)[temp_col].mean()
        hourly_avg = hourly_avg.rename(columns={temp_col: "Average_Temperature"})
        fig_hour = px.bar(hourly_avg, x="Hour", y="Average_Temperature", title="Average Temperature by Hour")
        fig_hour = apply_english_chart_style(
            fig_hour,
            title="Average Temperature by Hour",
            x_title="Hour",
            y_title="Average Temperature (°C)"
        )
        st.plotly_chart(fig_hour, use_container_width=True)

        st.subheader("Temperature Distribution")
        fig_hist = px.histogram(filtered, x=temp_col, nbins=40, title="Temperature Distribution")
        fig_hist = apply_english_chart_style(
            fig_hist,
            title="Temperature Distribution",
            x_title="Temperature (°C)",
            y_title="Count"
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

st.markdown(
    """
    <div class="dashboard-footer">
      <strong>Seoul Weather ML Dashboard</strong><br/>
      GitHub Actions로 학습 결과를 자동 생성하고, Streamlit에서는 사전 생성된 모델과 결과 파일을 불러오는 빠른 대시보드 구조입니다.
    </div>
    """,
    unsafe_allow_html=True,
)
