import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import io
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cholera Surveillance System – Bauchi State",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0d2137 60%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c9d9ed !important; }
[data-testid="stSidebar"] .stRadio label { color: #a8c4e0 !important; }

/* ── Main background ── */
[data-testid="stAppViewContainer"] {
    background: #f0f4f8;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
}

/* ── KPI metric boxes ── */
.kpi-box {
    background: #ffffff;
    border-left: 5px solid #0066cc;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    text-align: center;
}
.kpi-box.alert  { border-left-color: #e63946; }
.kpi-box.warn   { border-left-color: #f4a261; }
.kpi-box.safe   { border-left-color: #2a9d8f; }
.kpi-title  { font-size: 0.78rem; font-weight: 600; color: #5a6a7e; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-value  { font-size: 2rem;   font-weight: 700; color: #0a1628; font-family: 'IBM Plex Mono', monospace; }
.kpi-sub    { font-size: 0.72rem; color: #8899aa; margin-top: 2px; }

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem; font-weight: 700; color: #0a1628;
    border-bottom: 2px solid #0066cc; padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── Alert banner ── */
.alert-banner {
    background: #fff0f0; border: 1px solid #e63946;
    border-radius: 8px; padding: 0.8rem 1.2rem;
    color: #c1121f; font-weight: 600; margin-bottom: 1rem;
}
.warn-banner {
    background: #fff8ec; border: 1px solid #f4a261;
    border-radius: 8px; padding: 0.8rem 1.2rem;
    color: #e07b39; font-weight: 600; margin-bottom: 1rem;
}
.safe-banner {
    background: #eafaf7; border: 1px solid #2a9d8f;
    border-radius: 8px; padding: 0.8rem 1.2rem;
    color: #1a7a6e; font-weight: 600; margin-bottom: 1rem;
}

/* ── Table ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Buttons ── */
div.stButton > button {
    background: #0066cc; color: #fff;
    border: none; border-radius: 6px;
    font-weight: 600; padding: 0.5rem 1.4rem;
    transition: background 0.2s;
}
div.stButton > button:hover { background: #0052a3; }

/* ── Footer ── */
.footer {
    text-align: center; color: #8899aa; font-size: 0.72rem;
    margin-top: 2rem; border-top: 1px solid #dde3ea; padding-top: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS MAP FROM DATASET
# ─────────────────────────────────────────────
RISK_LEVELS = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#2a9d8f", "Medium": "#f4a261", "High": "#e63946"}

# ─────────────────────────────────────────────
#  DATA LOADER (USING REAL DATASET)
# ─────────────────────────────────────────────
@st.cache_data
def load_real_data() -> pd.DataFrame:
    # Read the referenced CSV file
    file_path = "bauchi_cholera_2019_52weeks.csv"
    df = pd.read_csv(file_path)
   
    # Clean text columns from trailing/leading white spaces
    string_cols = ["LGA", "Primary_Water_Source", "Secondary_Water_Source",
                "Water_Safety_Level", "Sanitation_Type", "Waste_Management_Level",
                "Access_Level", "Flood_Risk_Level", "Proximity_to_Water_Body", "Environmental_Risk"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
           
    # Remap variations if any to match constants standard
    if "Environmental_Risk" in df.columns:
        df["Environmental_Risk"] = df["Environmental_Risk"].replace({"Medium": "Medium", "High": "High", "Low": "Low"})
       
    return df

# Load standard base frame
df_raw = load_real_data()
NIGERIAN_LGAs = sorted(df_raw["LGA"].unique().tolist())
WATER_SOURCES = sorted(df_raw["Primary_Water_Source"].unique().tolist())
SANITATION_TYPES = sorted(df_raw["Sanitation_Type"].unique().tolist())

# ─────────────────────────────────────────────
#  ML MODEL EXTRACTION ENGINE
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(df: pd.DataFrame):
    features = [
        "Rainfall_mm", "Temperature_C", "Humidity_%",
        "Toilet_Access_%", "Open_Defecation_%", "Health_Facilities_Count",
        "Hospital_Count", "Health_Workers", "Elevation_m", "Population (2016 census)"
    ]
    cat_features = ["Primary_Water_Source", "Sanitation_Type", "Flood_Risk_Level", "Access_Level"]

    le = {}
    df_enc = df.copy()
    for col in cat_features:
        le[col] = LabelEncoder().fit(df[col])
        df_enc[col + "_enc"] = le[col].transform(df[col])

    enc_features = features + [c + "_enc" for c in cat_features]

    X = df_enc[enc_features]
    le_risk = LabelEncoder().fit(df["Environmental_Risk"])
    y = le_risk.transform(df["Environmental_Risk"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    }

    trained, scores = {}, {}
    for name, mdl in models.items():
        mdl.fit(X_train_s, y_train)
        scores[name] = accuracy_score(y_test, mdl.predict(X_test_s))
        trained[name] = mdl

    best_name = max(scores, key=lambda k: scores[k])
    return (
        trained, scores, scaler, le, le_risk,
        enc_features, X_test_s, y_test, best_name,
    )

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦠 Cholera Outbreak Surveillance\n**Bauchi State Early Warning System**")
    st.markdown("---")

    nav = st.radio(
        "Navigation",
        [
            "📊 Dashboard Overview",
            "🗺️ Geographic Analysis",
            "📈 Trend Analysis",
            "🤖 Predictive Model",
            "🔍 Risk Assessment",
            "📋 Data Explorer",
            "ℹ️ About",
        ],
    )

    st.markdown("---")
    st.markdown("**⚙️ Filters**")

    sel_lgas = st.multiselect(
        "Filter by LGA",
        options=NIGERIAN_LGAs,
        default=[],
        placeholder="All LGAs",
    )
    sel_risk = st.multiselect(
        "Filter by Environmental Risk",
        options=RISK_LEVELS,
        default=[],
        placeholder="All risk levels",
    )
    epi_week_range = st.slider(
        "Epidemiological Week Range",
        min_value=int(df_raw["epi_week"].min()),
        max_value=int(df_raw["epi_week"].max()),
        value=(int(df_raw["epi_week"].min()), int(df_raw["epi_week"].max()))
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#7a8fa6'>"
        "Data Source: Bauchi 2019 Surveillance Logs<br>"
        "Model trained dynamic onto historical records."
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  DATA FILTERING APPLICATION
# ─────────────────────────────────────────────
df = df_raw.copy()
if sel_lgas:
    df = df[df["LGA"].isin(sel_lgas)]
if sel_risk:
    df = df[df["Environmental_Risk"].isin(sel_risk)]
df = df[(df["epi_week"] >= epi_week_range[0]) & (df["epi_week"] <= epi_week_range[1])]

if df.empty:
    st.warning("⚠️ No records match the active criteria. Please adjust your filters.")
    st.stop()

# ─────────────────────────────────────────────
#  PAGE: DASHBOARD OVERVIEW
# ─────────────────────────────────────────────
if nav == "📊 Dashboard Overview":
    st.markdown("# 📊 Cholera Outbreak Surveillance Dashboard")
    st.markdown("**Bauchi State Epidemiological Cholera Outbreak Tracking Hub**")
    st.markdown("---")

    # Alert banners
    total_cases = df["Number of cases"].sum()
    total_deaths = df["Deaths"].sum()
    avg_cfr = df["Case fatality rate (%)"].mean()
    high_risk_count = df[df["Environmental_Risk"] == "High"]["LGA"].nunique()

    if total_cases > 500 or high_risk_count >= 3:
        st.markdown(
            f'<div class="alert-banner">🚨 OUTBREAK ALERT: {high_risk_count} LGAs presenting critical high environmental conditions '
            f'with {total_cases:,} active cases logged in selected windows.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="safe-banner">✅ MONITORING STABLE: Baselines remain within expected local intervention metrics.</div>',
            unsafe_allow_html=True,
        )

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_data = [
        "Total Cases", f"{total_cases:,}", "Aggregated recorded cases", "alert" if total_cases > 0 else "safe",
        "Total Deaths", f"{total_deaths:,}", "Fatalities logged", "alert" if total_deaths > 0 else "safe",
        "Mean CFR (%)", f"{avg_cfr:.2f}%", "Case Fatality Ratio", "warn",
        "LGAs Monitored", f"{df['LGA'].nunique()}", "Active spatial zones", "safe",
        "High Risk Logs", f"{df[df['Environmental_Risk']=='High'].shape[0]}", "High exposure periods", "alert"
    ]
   
    chunks = [kpi_data[i:i + 4] for i in range(0, len(kpi_data), 4)]
    for col, (title, val, sub, cls) in zip([c1, c2, c3, c4, c5], chunks):
        col.markdown(
            f'<div class="kpi-box {cls}">'
            f'<div class="kpi-title">{title}</div>'
            f'<div class="kpi-value">{val}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Records distributed by Environmental Risk Profile</div>', unsafe_allow_html=True)
        risk_counts = df["Environmental_Risk"].value_counts().reindex(RISK_LEVELS).fillna(0)
        fig_risk = px.bar(
            x=risk_counts.index, y=risk_counts.values, color=risk_counts.index,
            color_discrete_map=RISK_COLORS, labels={"x": "Risk Tier", "y": "Count of Weekly Logs"},
            template="plotly_white"
        )
        fig_risk.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Primary Drinking Water Vector Supply Share</div>', unsafe_allow_html=True)
        ws_counts = df["Primary_Water_Source"].value_counts()
        fig_ws = px.pie(
            values=ws_counts.values, names=ws_counts.index, hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel, template="plotly_white"
        )
        fig_ws.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_ws, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-header">Top Local Government Areas (LGAs) by Gross Cases</div>', unsafe_allow_html=True)
        top_lga = df.groupby("LGA")["Number of cases"].sum().nlargest(10).reset_index()
        fig_lga = px.bar(
            top_lga, x="Number of cases", y="LGA", orientation="h",
            color="Number of cases", color_continuous_scale="Reds", template="plotly_white"
        )
        fig_lga.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_lga, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">Sanitation Infrastructure vs Environmental Score Metric</div>', unsafe_allow_html=True)
        fig_box = px.box(
            df, x="Sanitation_Type", y="Open_Defecation_%", color="Sanitation_Type",
            template="plotly_white"
        )
        fig_box.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: GEOGRAPHIC ANALYSIS
# ─────────────────────────────────────────────
elif nav == "🗺️ Geographic Analysis":
    st.markdown("# 🗺️ Geographic Analysis")
    st.markdown("Spatial assessment of structural risk metrics per local government area jurisdiction.")
    st.markdown("---")

    metric = st.selectbox(
        "Select Regional Metric Breakdown Target",
        ["Number of cases", "Deaths", "Case fatality rate (%)", "Open_Defecation_%", "Rainfall_mm"],
    )

    lga_agg = df.groupby("LGA").agg({
        "Number of cases": "sum",
        "Deaths": "sum",
        "Case fatality rate (%)": "mean",
        "Open_Defecation_%": "mean",
        "Rainfall_mm": "mean"
    }).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">LGA Structural Rank Comparison Chart</div>', unsafe_allow_html=True)
        lga_sorted = lga_agg.sort_values(metric, ascending=True)
        fig_geo = px.bar(
            lga_sorted, x=metric, y="LGA", orientation="h",
            color=metric, color_continuous_scale="YlOrRd", template="plotly_white", height=500
        )
        fig_geo.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_geo, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Risk Matrix Densities by LGA Distribution</div>', unsafe_allow_html=True)
        risk_lga = df.groupby(["LGA", "Environmental_Risk"]).size().reset_index(name="Count")
        fig_rl = px.bar(
            risk_lga, x="LGA", y="Count", color="Environmental_Risk",
            color_discrete_map=RISK_COLORS, barmode="stack", template="plotly_white", height=500
        )
        fig_rl.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_rl, use_container_width=True)

    st.dataframe(lga_agg.sort_values("Number of cases", ascending=False).reset_index(drop=True), use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: TREND ANALYSIS
# ─────────────────────────────────────────────
elif nav == "📈 Trend Analysis":
    st.markdown("# 📈 Trend Analysis")
    st.markdown("Epidemiological timeline tracking curves for the 52-week calendar cycle.")
    st.markdown("---")

    weekly_trend = df.groupby("epi_week").agg({
        "Number of cases": "sum",
        "Rainfall_mm": "mean",
        "Temperature_C": "mean"
    }).reset_index().sort_values("epi_week")

    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(go.Bar(x=weekly_trend["epi_week"], y=weekly_trend["Number of cases"],
                            name="Logged Cholera Cases", marker_color="#0066cc", opacity=0.8))
    fig_ts.add_trace(go.Scatter(x=weekly_trend["epi_week"], y=weekly_trend["Rainfall_mm"],
                                name="Mean Rainfall (mm)", mode="lines+markers", line=dict(color="#e63946", width=2)),
                     secondary_y=True)
   
    fig_ts.update_layout(title="Weekly Incidence Progression vs Climatic Rainfall Cycle", template="plotly_white", height=400)
    fig_ts.update_xaxes(title_text="Epidemiological Week Profile")
    fig_ts.update_yaxes(title_text="Aggregated Cases", secondary_y=False)
    fig_ts.update_yaxes(title_text="Rainfall Volume Metric (mm)", secondary_y=True)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown('<div class="section-header">Risk Co-dependency Parameter Matrix Correlation</div>', unsafe_allow_html=True)
    num_cols = ["Rainfall_mm", "Temperature_C", "Humidity_%", "Toilet_Access_%",
                "Open_Defecation_%", "Health_Facilities_Count", "Number of cases", "Deaths"]
    corr = df[num_cols].corr()
    fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template="plotly_white", height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: PREDICTIVE MODEL
# ─────────────────────────────────────────────
elif nav == "🤖 Predictive Model":
    st.markdown("# 🤖 Predictive Classifier Model Workspace")
    st.markdown("Machine learning algorithms tracking prediction arrays for validation classification labels.")
    st.markdown("---")

    with st.spinner("Processing optimization calculations over real validation metrics..."):
        (trained_models, scores, scaler, le_cats, le_risk,
         enc_features, X_test_s, y_test, best_name) = train_model(df_raw)

    score_df = pd.DataFrame({
        "Model Algorithm Architecture": list(scores.keys()),
        "Evaluated Testing Accuracy (%)": [round(v * 100, 2) for v in scores.values()],
    })
   
    fig_scores = px.bar(score_df, x="Model Algorithm Architecture", y="Evaluated Testing Accuracy (%)",
                        color="Evaluated Testing Accuracy (%)", color_continuous_scale="Blues", text="Evaluated Testing Accuracy (%)", template="plotly_white")
    fig_scores.update_layout(yaxis_range=[0, 110], height=300, coloraxis_showscale=False)
    st.plotly_chart(fig_scores, use_container_width=True)

    st.success(f"🏆 Top Executing Algorithm Configuration Match: **{best_name}** Matrix Profile.")

    best_mdl = trained_models[best_name]
    y_pred = best_mdl.predict(X_test_s)
    labels = le_risk.classes_

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Performance Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues", template="plotly_white", height=350)
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">Classification Performance Metrics Summary</div>', unsafe_allow_html=True)
        report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: RISK ASSESSMENT (REAL-TIME ESTIMATOR)
# ─────────────────────────────────────────────
elif nav == "🔍 Risk Assessment":
    st.markdown("# 🔍 Real-Time Community Risk Assessor Pipeline")
    st.markdown("Input environmental features to output projected predictive categories.")
    st.markdown("---")

    (trained_models, scores, scaler, le_cats, le_risk, enc_features, X_test_s, y_test, best_name) = train_model(df_raw)
    best_mdl = trained_models[best_name]

    with st.form("assessment_form"):
        st.markdown("### 📋 Catchment Environmental Input Matrix Form")
        cl1, cl2, cl3 = st.columns(3)
       
        with cl1:
            rf = cl1.slider("Rainfall Configuration Level (mm)", 0.0, 400.0, 50.0)
            tp = cl1.slider("Target Air Temperature Profile (°C)", 15.0, 45.0, 31.0)
            hm = cl1.slider("Relative Air Humidity (%)", 10.0, 100.0, 45.0)
        with cl2:
            ta = cl2.slider("Household Toilet Access Share (%)", 0.0, 100.0, 40.0)
            od = cl2.slider("Open Defecation Density Benchmark (%)", 0.0, 100.0, 25.0)
            pop = cl2.number_input("Population Count Metrics Index", value=50000)
        with cl3:
            hfc = cl3.number_input("Total Health Infrastructure Sites Count", value=15)
            hosp = cl3.number_input("Specialized Clinics / Hospital Count", value=2)
            hw = cl3.number_input("Operational Medical Staff Personnel Pool Count", value=30)
            elev = cl3.number_input("Target Geographic Base Elevation Zone (meters)", value=400)

        st.markdown("### 🌐 Structural WASH Parameters Selection")
        cx1, cx2, cx3, cx4 = st.columns(4)
        pws = cx1.selectbox("Primary Drinking Source Vector Option", WATER_SOURCES)
        stype = cx2.selectbox("Excreta Treatment Sanitation Framework Type", SANITATION_TYPES)
        frl = cx3.selectbox("Local Baseline Area Flood Risk Assessment", ["Low", "High"])
        alvl = cx4.selectbox("General Resource Proximity Access Class Level", ["Low", "Medium", "High"])

        evaluate = st.form_submit_button("⚡ Run Diagnostic Risk Evaluation")

    if evaluate:
        # Convert user inputs to matching vectorized frame
        w_enc = le_cats["Primary_Water_Source"].transform([pws])[0]
        s_enc = le_cats["Sanitation_Type"].transform([stype])[0]
        f_enc = le_cats["Flood_Risk_Level"].transform([frl])[0]
        a_enc = le_cats["Access_Level"].transform([alvl])[0]

        input_vector = np.array([[
            rf, tp, hm, ta, od, hfc, hosp, hw, elev, pop, w_enc, s_enc, f_enc, a_enc
        ]])
       
        scaled_vec = scaler.transform(input_vector)
        pred_idx = best_mdl.predict(scaled_vec)[0]
        proba_list = best_mdl.predict_proba(scaled_vec)[0]
        risk_output = le_risk.inverse_transform([pred_idx])[0]

        color = RISK_COLORS.get(risk_output, "#0066cc")
        st.markdown(f"""
            <div style="background:{color}15; border:3px solid {color}; padding:20px; border-radius:8px;">
                <h3 style="color:{color}; margin:0;">EVALUATED CLASSIFICATION CATEGORY MATCH: {risk_output.upper()}</h3>
                <p style="color:#333; margin:5px 0 0 0;">Confidence Score Probability: <strong>{proba_list[pred_idx]*100:.2f}%</strong> via local engine profile validation logic.</p>
            </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE: DATA EXPLORER
# ─────────────────────────────────────────────
elif nav == "📋 Data Explorer":
    st.markdown("# 📋 Data Explorer Engine")
    st.markdown("Filter, audit, and extract compiled data registries directly from tracking indices.")
    st.markdown("---")

    st.markdown(f"Currently inspecting **{len(df):,} specific weekly records** matching parameters.")
    st.dataframe(df, use_container_width=True)

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Structured Tracking Data Subset (CSV format)",
        data=csv_buffer.getvalue(),
        file_name="Surveillance_Matrix_Subset.csv",
        mime="text/csv"
    )

# ─────────────────────────────────────────────
#  PAGE: ABOUT
# ─────────────────────────────────────────────
elif nav == "ℹ️ About":
    st.markdown("# ℹ️ Surveillance Infrastructure Overview Information")
    st.markdown("---")
    st.info("System optimized to digest structured metrics tracking mapping for 52 distinct reporting periods across structural Bauchi State validation logs.")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        Cholera Surveillance System &nbsp;|&nbsp; Nigeria Community Health
        <br>
        For public health research & demonstration purposes only
    </div>
    """,
    unsafe_allow_html=True,
)
