Crop Yield Predictor

An AI-powered machine learning system that predicts crop yield using soil parameters, weather conditions, and agricultural factors. This project combines data science, XGBoost modeling, and real-time API integration to help farmers make informed decisions for optimal agricultural planning.

---

Overview

The Crop Yield Predictor uses synthetic yet realistic agricultural data to forecast expected yield. It leverages:

 Soil features (pH, Nitrogen, Phosphorus, Potassium)
 Weather parameters (Rainfall, Temperature, Humidity)
 Agricultural factors (Crop type, Soil type, Season, Irrigation type)

By training a tuned XGBoost Regressor pipeline, the model achieves:

 MAE: ~1.2
 RMSE: ~1.5
 R²: 0.98

---

Key Features

 End-to-end preprocessing pipeline with imputation, scaling & encoding
 Realistic synthetic dataset (correlated features)
 Hyperparameter-tuned XGBoost regression model
 Visualizations for model evaluation & feature importance
 Extendable for real-time API integration (OpenWeatherMap, AgroMonitoring)

---

Tech Stack

 Python 3.10+
 Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib
 Model: XGBoostRegressor wrapped in scikit-learn Pipeline
 Environment: Jupyter Notebook / Google Colab

---

Model Performance

| Metric | Score |
| ------ | ----- |
| MAE    | 1.239 |
| RMSE   | 1.530 |
| R²     | 0.980 |

---
Visualization Examples

 Predicted vs Actual Crop Yield Scatter Plot
 Feature Importance Bar Graph

---

Future Enhancements

 Integrate OpenWeatherMap API for live weather input
 Use AgroMonitoring API for soil condition data
 Deploy model via Flask or Streamlit web app
 Add automated retraining pipeline usin
