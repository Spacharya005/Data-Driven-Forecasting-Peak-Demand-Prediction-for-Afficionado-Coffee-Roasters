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
st.sidebar.header("Dashboard Controls")

store = st.sidebar.selectbox(
    "Select Store",
    df['store_id'].unique()
)

freq = st.sidebar.selectbox(
    "Time Aggregation Level",
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
    "Models",
    model_list,
    default=["Naive", "ARIMA", "Gradient Boosting"]
)

freq_map = {"Hourly": "h", "Daily": "D"}

df['datetime'] = pd.to_datetime(
    df['year'].astype(str) + "-01-01 " + df['transaction_time'],
    format='%Y-%m-%d %H:%M:%S',
    errors='coerce'
)
@st.cache_data
def process_data(df, store, freq, metric_type):
    df_store = df[df['store_id'] == store]
    agg_df = aggregate_data(df_store, freq)

    if metric_type == "Quantity":
        agg_df['target'] = agg_df['transaction_qty']
    else:
        agg_df['target'] = agg_df['revenue']

    agg_df = agg_df.set_index('datetime').asfreq(freq).fillna(0).reset_index()

    return agg_df
# -----------------------------
# FILTER + PROCESS
# -----------------------------
df_store = df[df['store_id'] == store]

agg_df = process_data(df, store, freq_map[freq], metric_type)
agg_df.replace([np.inf, -np.inf], np.nan, inplace=True)
agg_df.fillna(0, inplace=True)

if metric_type == "Quantity":
    agg_df['target'] = agg_df['transaction_qty']
else:
    agg_df['target'] = agg_df['revenue']

# st.write("After target creation:", agg_df.columns)
# ✅ PRD FIX: continuous time index
agg_df = agg_df.set_index('datetime') \
               .asfreq(freq_map[freq]) \
               .fillna(0) \
               .reset_index()

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
# STEP 1: split raw data first
train_df, test_df = split_series(agg_df, target='target', return_df=True)

# STEP 2: create features separately
# train_feat = create_features(train_df)
# test_feat = create_features(test_df)
full_feat = create_features(pd.concat([train_df, test_df]))

train_feat = full_feat[full_feat['datetime'].isin(train_df['datetime'])]
test_feat = full_feat[full_feat['datetime'].isin(test_df['datetime'])]


# STEP 3: clean
train_feat = train_feat.copy()
test_feat = test_feat.copy()

for df_ in [train_feat, test_feat]:
    df_.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_cols = df_.columns.difference(['target', 'datetime'])

    df_.loc[:, feature_cols] = df_[feature_cols].ffill().fillna(0)

    df_.dropna(subset=['target'], inplace=True)

# STEP 4: define X/y
train_feat = train_feat.sort_values('datetime')
test_feat = test_feat.sort_values('datetime')

y_train = train_feat['target']
y_test = test_feat['target']

X_train = train_feat.drop(columns=['target', 'datetime'])
X_test = test_feat.drop(columns=['target', 'datetime'])
# ✅ Ensure feature names are preserved (fix sklearn warning)
X_test = pd.DataFrame(X_test, columns=X_train.columns)

if len(y_train) == 0 or len(y_test) == 0:
    st.error("🚨 Train/Test split failed")
    st.stop()

# -----------------------------
# ✅ CACHE MODELS (FIX REFRESH)
# -----------------------------
@st.cache_data(ttl=3600, hash_funcs={
    pd.DataFrame: lambda x: x.shape,
    pd.Series: lambda x: x.shape
}) # Cache for 1 hour
def run_all_models(selected_models, y_train, y_test, X_train, X_test, feat_df):
    predictions = {}
    for model in selected_models:
        try:
            if model == "Prophet":
                preds = run_model(model, y_train, y_test, X_train, X_test, df=feat_df)
            else:
                preds = run_model(model, y_train, y_test, X_train, X_test)

            predictions[model] = preds.copy()

        except Exception as e:
            print(f"❌ {model} failed:", e)
            predictions[model] = np.repeat(y_train.mean(), len(y_test))

    return predictions

# -----------------------------
# RUN MODELS
# -----------------------------
with st.spinner("☕ Brewing predictions..."):
    predictions = run_all_models(
        tuple(selected_models),
        y_train, y_test,
        X_train, X_test,
        train_feat
    )
if not predictions:
    st.error("No valid model predictions available")
    st.stop()

for model, preds in predictions.items():
    print(model, "First 5 preds:", preds[:5])
    print(model, "First 5 actual:", y_test.values[:5])

# -----------------------------
# MODEL EVALUATION
# -----------------------------
results_df = evaluate_all(y_test.values, predictions)
best_model = results_df.iloc[0]['Model']
# prevent near-perfect leakage cases
# if np.allclose(preds, y_true, atol=1e-5):
#     print(f"⚠️ {model} suspiciously perfect → possible leakage")
# -----------------------------
# FUTURE FORECAST
# -----------------------------
future_preds = run_model(
    best_model,
    y_train,
    None,
    X_train,
    None,
    df=train_feat,
    horizon=horizon
)

freq_fixed = freq_map[freq]

future_index = pd.date_range(
    start=agg_df['datetime'].max(),
    periods=horizon + 1,
    freq=freq_fixed
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

    # ✅ Focus only on last N points (clean view)
    window = 200
    y_vis = y_test.iloc[-window:]
    x_vis = y_vis.index

    # ✅ Actual (clear & bold)
    fig.add_trace(go.Scatter(
        x=x_vis,
        y=y_vis.values,
        mode='lines',
        name='Actual',
        line=dict(width=4, color='black')
    ))

    for model, preds in predictions.items():

        preds_vis = preds[-window:]

        fig.add_trace(go.Scatter(
            x=x_vis,
            y=preds_vis,
            mode='lines',
            name=model,
            line=dict(
                width=3 if model == best_model else 2,
                dash='solid' if model == best_model else 'dot'
            ),
            opacity=1 if model == best_model else 0.5
        ))

    best_preds = predictions[best_model][-window:]


    # ✅ Confidence Interval (clean & light)
    residuals = y_test.values[-window:] - best_preds
    std = np.std(residuals)

    upper = best_preds + 1.96 * std
    lower = np.maximum(best_preds - 1.96 * std, 0)

    fig.add_trace(go.Scatter(
        x=x_vis,
        y=upper,
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=x_vis,
        y=lower,
        fill='tonexty',
        name='Confidence Interval',
        opacity=0.15,
        line=dict(width=0)
    ))

    # ✅ CLEAN LAYOUT
    fig.update_layout(title="Forecast vs Actual",
        template=plotly_theme,
        height=450,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", y=1.02),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"{best_model} Forecast for Store {store}")
    st.info("📌 Note: Negative demand predictions are clipped to zero for business realism.")

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
    st.caption("MAPE = Mean Absolute Percentage Error (lower is better)")
    col4 = st.columns(1)[0]
    col4.metric("MAE", round(results_df.iloc[0]['MAE'], 2))
    st.dataframe(results_df)

    fig = px.bar(results_df, x='Model', y='RMSE')
    fig.update_layout(title="Model RMSE Comparison",
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

    # st.success(f"🔥 Peak demand expected around {peak_hour}:00 (based on forecasted patterns)")
    st.subheader("Demand Spike Detection")
    if len(y_test) > 5:
        spikes = detect_spikes(y_test)
    else:
        spikes = pd.Series()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=y_test.values,
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

    fig.update_layout(title="Demand with Detected Spikes",
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
        fig.update_layout(title="Forecasted Demand Heatmap",
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
        st.caption("Heatmap shows expected demand intensity by hour and day")
    else:
        st.warning("Not enough data for heatmap")
    st.subheader("Model Insight")

    st.info(f"""
    📊 The best performing model is **{best_model}**

    - Selected based on lowest error (MAPE & RMSE)
    - Captures demand patterns effectively
    - Recommended for operational planning

    👉 Use this forecast to:
    - Align staffing with peak hours
    - Optimize inventory
    - Reduce wastage
    """)
    
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
