# Seoul Short-Term Temperature Prediction Model Card

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
| Metric                           | Value               |
|:---------------------------------|:--------------------|
| Source file count                | 5                   |
| Raw rows                         | 43728               |
| Rows after datetime cleaning     | 43728               |
| Rows after deduplication         | 43728               |
| Duplicate timestamp rows removed | 0                   |
| Missing-hour rows created        | 96                  |
| Observed start                   | 2020-12-31 01:00:00 |
| Observed end                     | 2025-12-31 00:00:00 |

## Modeling Approach
The target variable is temperature change rather than direct future temperature. This reduces over-reliance on current temperature persistence and makes the problem more scientifically interpretable as short-term thermal tendency prediction.

Models compared:
| Model                   |      MAE |     RMSE |       R2 |      ME_Bias |   P90_Absolute_Error |   P95_Absolute_Error |
|:------------------------|---------:|---------:|---------:|-------------:|---------------------:|---------------------:|
| Random Forest Light     | 0.301173 | 0.442474 | 0.998457 |  0.00245149  |             0.676871 |             0.909221 |
| Gradient Boosting Tuned | 0.317099 | 0.456943 | 0.998354 |  0.00462666  |             0.706502 |             0.921585 |
| Linear Regression       | 0.368289 | 0.515639 | 0.997904 |  0.0127725   |             0.798382 |             1.05295  |
| Persistence Baseline    | 0.65902  | 0.888819 | 0.993773 | -4.06675e-19 |             1.5      |             1.9      |
| Previous-Day Baseline   | 2.6795   | 3.52393  | 0.90211  | -0.00429625  |             5.9      |             7.2      |
| Month-Hour Climatology  | 3.23605  | 4.03972  | 0.871358 |  0.145215    |             6.41371  |             7.67621  |

## Time-Series Cross-Validation Summary
| Model                   |   CV_MAE_Mean |   CV_MAE_STD |   CV_RMSE_Mean |   CV_RMSE_STD |   CV_R2_Mean |   CV_R2_STD |
|:------------------------|--------------:|-------------:|---------------:|--------------:|-------------:|------------:|
| Random Forest Light     |      0.305817 |   0.0116615  |       0.440565 |    0.0115287  |     0.99841  | 2.31609e-05 |
| Gradient Boosting Tuned |      0.319008 |   0.0107483  |       0.454087 |    0.011299   |     0.998311 | 2.4355e-05  |
| Linear Regression       |      0.373195 |   0.00673407 |       0.515978 |    0.00641726 |     0.997818 | 5.56324e-05 |

## Deployment Model
- Best model by test RMSE: Random Forest Light
- Deployment model: Random Forest Light
- Test MAE: 0.3012 °C
- Test RMSE: 0.4425 °C
- Test R²: 0.9985

## Academic Meaning
The model evaluates how short-term temperature tendency in Seoul is related to temporal continuity, diurnal cycle, seasonal cycle, humidity, wind, pressure, rainfall, and lagged thermal states.

## Limitations
- Single-station ASOS data cannot represent full spatial variation across Seoul.
- Future humidity, wind, pressure, and rainfall are unknown beyond the latest observation.
- Long-range future estimates should be interpreted as statistical scenario estimates, not official weather forecasts.
- Prediction uncertainty increases as iterative forecasting horizon increases.

## Reproducibility
The training pipeline is executed by GitHub Actions, and all model artifacts and diagnostic outputs are saved to `result/`.
