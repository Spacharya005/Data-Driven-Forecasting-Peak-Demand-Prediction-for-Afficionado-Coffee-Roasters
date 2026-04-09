# 📊 Data-Driven Forecasting for Peak Demand Prediction  
## ☕ Afficionado Coffee Roasters

---

## 📌 Overview

This project presents a **data-driven forecasting framework** to predict peak demand for **Afficionado Coffee Roasters**. The system leverages historical transaction data and time-series modeling to generate accurate demand forecasts.

The solution helps in:

* Predicting **future coffee demand**
* Identifying **peak hours and days**
* Supporting **inventory and staffing decisions**
* Improving **operational efficiency**

An interactive **Streamlit dashboard** is also developed for real-time forecasting and visualization.

---

## 🎯 Objectives

### Primary Objectives

* Forecast coffee demand at **hourly and daily levels**
* Identify **peak demand periods**
* Analyze store-level performance and trends

### Secondary Objectives

* Improve inventory and supply planning
* Reduce stockouts and wastage
* Enable data-driven business decision-making

---

## 📂 Dataset Description

The dataset contains transactional data from coffee stores:

| Column            | Description                      |
| ----------------- | -------------------------------- |
| transaction_time  | Timestamp of transaction         |
| store_id          | Unique store identifier          |
| product_id        | Product identifier               |
| quantity          | Quantity sold                    |
| revenue           | Sales revenue                    |

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Timestamp conversion and sorting
* Handling missing values
* Aggregation at required frequency (Hourly/Daily)

### 2. Feature Engineering

Derived key features:

* **Time-based features** (hour, day, weekday, month)
* **Lag features** for time-series modeling
* **Rolling averages** for smoothing trends

### 3. Forecasting Models

* **ARIMA / SARIMA** (Statistical models)
* **Machine Learning models** (Regression-based)

### 4. Model Evaluation

* Comparison of predicted vs actual demand
* Error metric calculations

---

## 📈 Key Performance Indicators (KPIs)

* **Total Sales Volume**
* **Peak Demand Periods**
* **Forecast Accuracy**
* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**

---

## 📊 Streamlit Dashboard

### Features

* Demand Forecast Visualization
* Actual vs Predicted Comparison
* Store-wise Analysis
* Time-series Trend Plots

### User Controls

* Store selection
* Frequency selection (Hourly / Daily)
* Date range filters
* Model selection

---

## 🧠 Key Insights

* Demand shows **strong time-based patterns**
* Peak sales occur during **specific hours of the day**
* Certain stores contribute disproportionately to total demand
* Forecasting helps in **anticipating high-load periods**

---

## 💡 Recommendations

* Optimize staffing during peak hours
* Maintain buffer inventory for high-demand periods
* Use forecasting outputs for proactive decision-making
* Continuously retrain models with new data

---

## 🛠️ Tech Stack

* Python (Pandas, NumPy)
* Scikit-learn
* Statsmodels (ARIMA)
* Plotly / Matplotlib
* Streamlit

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Spacharya005/Data-Driven-Forecasting-Peak-Demand-Prediction-for-Afficionado-Coffee-Roasters.git
cd Data-Driven-Forecasting-Peak-Demand-Prediction-for-Afficionado-Coffee-Roasters
