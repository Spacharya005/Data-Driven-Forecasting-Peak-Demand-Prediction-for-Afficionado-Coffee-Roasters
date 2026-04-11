import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from src.data_loader import load_data
from src.feature_engineering import create_features
from src.forecasting_pipeline import aggregate_data, split_series
from src.forecasting_models import run_model
from src.evaluation import evaluate_all
from utility import detect_spikes

# st.write("FILES IN ROOT:", os.listdir())
# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="☕ Data-Driven Forecasting & Peak Demand Prediction for Afficionado Coffee Roasters ☕",
    layout="wide"
)

st.title("☕ Data-Driven Forecasting & Peak Demand Prediction for Afficionado Coffee Roasters ☕")


# ---------------- HEADER WITH LOGOS ----------------
col1, spacer, col2 = st.columns([2, 3, 2])  # middle = gap

logo_height = 180

with col1:
    logo1_path = os.path.join("assets", "unified_mentor_logo_2.png")
    if os.path.exists(logo1_path):
        st.image(logo1_path, width=205)  # control size

with col2:
    logo2_path = os.path.join("assets", "afficionado_logo.png")
    if os.path.exists(logo2_path):
        st.image(logo2_path, width=195)  # control size

# ---------------- THEME (LIGHT ONLY) ----------------
plotly_theme = "plotly_white"

st.markdown("""
<style>
:root {
    --bg-color: #F5ECE3;
    --text-color: #4B2E2B;
    --sidebar-bg: #E6D3B3;
    --accent: #6F4E37;
    --card-bg: rgba(255, 255, 255, 0.6);
    --border-color: rgba(0, 0, 0, 0.1);
    --kpi-subtext: #7A5C4F;
}

/* Apply variables */
.stApp {
    background-color: var(--bg-color);
    color: var(--text-color);
}

section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg);
}

/* Headers */
h1, h2, h3 {
    color: var(--accent);
}

/* Buttons */
.stButton>button {
    background-color: #C19A6B;
    color: white;
}

/* Glass Card */
.glass-card {
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    background: var(--card-bg);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid var(--border-color);
}

/* KPI */
.kpi-title {
    color: var(--kpi-subtext);
}
.kpi-value {
    color: var(--text-color);
}

/* Sidebar + labels */
section[data-testid="stSidebar"] *,
label {
    color: var(--text-color) !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "coffee_sales.csv")
df = load_data(file_path)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Controls")

store = st.sidebar.selectbox(
    "Select Store",
    df['store_id'].unique()
)

freq = st.sidebar.selectbox(
    "Granularity",
    ["Hourly", "Daily"]
)
metric_type = st.sidebar.radio(
    "Select Metric",
    ["Quantity", "Revenue"]
)
horizon = st.sidebar.slider(
    "Forecast Horizon (Days)",
    1, 30, 7
)
model_list = [
    "Naive",
    "Moving Average",
    "ARIMA",
    "Exponential Smoothing",
    "Prophet",
    "Gradient Boosting"
]

selected_models = st.sidebar.multiselect(
    "Select Models",
    model_list,
    default=["Naive", "ARIMA", "Gradient Boosting"]
)

freq_map = {"Hourly": "H", "Daily": "D"}


# -----------------------------
# FILTER + PROCESS
# -----------------------------
df_store = df[df['store_id'] == store]

agg_df = aggregate_data(df_store, freq_map[freq])

if metric_type == "Revenue":
    agg_df['target'] = agg_df['revenue']   # ✅ FIX
else:
    agg_df['target'] = agg_df['transaction_qty']

feat_df = create_features(agg_df)

train, test = split_series(feat_df)

X_train = feat_df.iloc[:len(train)].drop(columns=['target', 'datetime'])
X_test = feat_df.iloc[len(train):].drop(columns=['target', 'datetime'])


# -----------------------------
# RUN MODELS
# -----------------------------
predictions = {}

with st.spinner("☕ Brewing predictions... Please wait..."):

    predictions = {}

    for model in selected_models:
        preds = run_model(
            model,
            train,
            test,
            X_train,
            X_test,
            df=feat_df
        )
        predictions[model] = preds

# -----------------------------
# MODEL EVALUATION
# -----------------------------
results_df = evaluate_all(test.values, predictions)
best_model = results_df.iloc[0]['Model']

# -----------------------------
# FUTURE FORECAST
# -----------------------------
future_preds = run_model(
    best_model,
    train,
    None,
    X_train,
    None,
    df=feat_df,
    horizon=horizon
)

future_index = pd.date_range(
    start=agg_df['datetime'].iloc[-1],
    periods=horizon + 1,
    freq=freq_map[freq]
)[1:]
    

# -----------------------------
# TABS UI
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Forecast",
    "📈 Model Comparison",
    "✅ Insights & Anomalies",
    "ℹ️ About Project"
])


# =============================
# 📈 TAB 1: FORECAST
# =============================
with tab1:

    st.subheader("Forecast vs Actual")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=test.index,
        y=test.values,
        mode='lines',
        name='Actual'
    ))

    for model, preds in predictions.items():
        fig.add_trace(go.Scatter(
                x=test.index,
            y=preds,
            mode='lines',
            name=model
        ))

    # ✅ Confidence Interval (Best Model)
    preds = predictions[best_model]
    std = np.std(preds)

    upper = preds + 1.96 * std
    lower = preds - 1.96 * std

    fig.add_trace(go.Scatter(y=upper, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        y=lower,
        fill='tonexty',
        name='Confidence Interval',
        opacity=0.2
    ))

    fig.update_layout(
        template=plotly_theme,
        font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
        title_font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
        xaxis=dict(
            title_font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
            tickfont=dict(color="white" if plotly_theme == "plotly_dark" else "black")
        ),
        yaxis=dict(
            title_font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
            tickfont=dict(color="white" if plotly_theme == "plotly_dark" else "black")
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    # Add future forecast to SAME graph
    fig.add_trace(go.Scatter(
        x=future_index,
        y=future_preds,
        mode='lines+markers',
        name='Future Forecast'
    ))
    st.subheader(f"{best_model} Forecast for Store {store}")
# =============================
# 📊 TAB 2: MODEL COMPARISON
# =============================
with tab2:

    st.subheader("Model Leaderboard")
    accuracy = 100 - results_df.iloc[0]['MAPE']
    col1, col2, col3 = st.columns(3)

    col1.metric("Best Model", best_model)
    col2.metric("RMSE", round(results_df.iloc[0]['RMSE'], 2))
    col3.metric("MAPE (%)", round(results_df.iloc[0]['MAPE'], 2))

    st.dataframe(results_df)

    fig = px.bar(results_df, x='Model', y='RMSE')
    fig.update_layout(
        template=plotly_theme,
        font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
        title_font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
        xaxis=dict(
            title_font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
            tickfont=dict(color="white" if plotly_theme == "plotly_dark" else "black")
        ),
        yaxis=dict(
            title_font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
            tickfont=dict(color="white" if plotly_theme == "plotly_dark" else "black")
        )
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================
# ⚡ TAB 3: INSIGHTS
# =============================
with tab3:
     # ✅ CREATE FIRST
    future_df = pd.DataFrame({
        'datetime': future_index,
        'target': future_preds
    })

    future_df['hour'] = future_df['datetime'].dt.hour
    future_df['day'] = future_df['datetime'].dt.day_name()

    # ✅ THEN USE
    peak_hour = future_df.groupby('hour')['target'].mean().idxmax()

    st.success(f"🔥 Peak demand expected around {peak_hour}:00 hours")
    peak_hour = future_df.groupby('hour')['target'].mean().idxmax()

    st.success(f"🔥 Peak demand expected around {peak_hour}:00 hours")
    st.subheader("Demand Spike Detection")
    spikes = detect_spikes(test)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=test.values,
        mode='lines',
        name='Demand'
    ))

    fig.add_trace(go.Scatter(
        x=spikes.index,
        y=spikes.values,
        mode='markers',
        name='Spikes',
        marker=dict(size=10, color='red')
    ))

    fig.update_layout(
        template=plotly_theme,
        font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
        title_font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
        xaxis=dict(
            title_font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
            tickfont=dict(color="white" if plotly_theme == "plotly_dark" else "black")
        ),
        yaxis=dict(
            title_font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
            tickfont=dict(color="white" if plotly_theme == "plotly_dark" else "black")
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Future Demand Heatmap")



    pivot = future_df.pivot_table(
        values='target',
        index='day',
        columns='hour',
        aggfunc='sum'
    )

    if not pivot.empty:
        fig = px.imshow(
            pivot,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            labels=dict(color="Demand")   # ✅ FIX
        )
        fig.update_layout(
            template=plotly_theme,
            font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
            title_font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
            xaxis=dict(
                title_font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
                tickfont=dict(color="white" if plotly_theme == "plotly_dark" else "black")
            ),
            yaxis=dict(
                title_font=dict(color="white" if plotly_theme == "plotly_dark" else "black"),
                tickfont=dict(color="white" if plotly_theme == "plotly_dark" else "black")
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data for heatmap")
    st.info("""
    📌 Business Recommendations:
    - Increase staffing during predicted peak hours
    - Optimize inventory for high-demand periods
    - Focus on high-performing stores
    - Use forecasts to reduce overproduction and waste
    """)

# =============================
# TAB 4 — ABOUT PROJECT
# =====================================================
with tab4:

    st.subheader("About This Project")

    st.markdown("""
    This project focuses on **data-driven demand forecasting for Afficionado Coffee Roasters**.

    It helps in:
    - Predicting future sales (hourly & daily)
    - Identifying peak demand periods
    - Supporting inventory and staff planning

    The system uses:
    - Time series models (ARIMA, Exponential Smoothing)
    - Machine learning models (Gradient Boosting)
    - Advanced forecasting (Prophet)

    This enables **proactive decision-making** instead of reactive operations.
    """)


    st.markdown("""
    ### Project Mentorship

    **Sai Prasad Kagne**  
    AI-Powered Engineer | Head of Data Science | Sports Analytics Engineer  
    [LinkedIn Profile](https://www.linkedin.com/in/saiprasad-kagne)

    ---

    ### Developed By

    **Shramanth P Acharya**  
    Machine Learning Intern – Unified Mentor  
    [LinkedIn Profile](https://www.linkedin.com/in/shramanth-p-acharya)
    """)

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; font-size:13px; color:gray;'>
        Developed by <a href='https://www.linkedin.com/in/shramanth-p-acharya' target='_blank'>Shramanth P Acharya</a><br> | 
        Mentored by <a href='https://www.linkedin.com/in/saiprasad-kagne' target='_blank'>Sai Prasad Kagne</a><br>
        Unified Mentor Internship Program<br>
        Data Source: Afficionado Coffee Roasters Sales Data
    </div>
    """,
    unsafe_allow_html=True
)

