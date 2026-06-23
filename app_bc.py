import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# ======================================================
# 1. PAGE CONFIGURATION & ACADEMIC EDITORIAL STYLING
# ======================================================
st.set_page_config(
    page_title="CRSAF Breast Cancer Stability Framework",
    layout="wide"
)

st.markdown("""
<style>
.title {font-size:2rem;font-weight:800;color:#1E3A8A}
.subtitle {font-size:1.05rem;color:#4B5563;margin-bottom:1rem}
.h {font-size:1.25rem;font-weight:700;margin-top:1.5rem;color:#1F2937;border-bottom:2px solid #E5E7EB;padding-bottom:0.3rem}
.box {background:#F8FAFC;padding:1rem;border-radius:10px;border-left:4px solid #3B82F6;margin:1rem 0;font-size:0.95rem;line-height:1.6;}
.objective-box {background:#F0FDF4;padding:1.25rem;border-radius:10px;border-left:4px solid #22C55E;margin:1rem 0;font-size:0.95rem;line-height:1.6;}
.logic-box {background:#FFFBEB;padding:1.25rem;border-radius:10px;border-left:4px solid #D97706;margin:1rem 0;font-size:0.95rem;line-height:1.6;}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='title'>Clinical Recommendation Stability Audit Framework (CRSAF)</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>A Regulatory Science Framework for HER2-Positive Breast Cancer Foundation Models Under Clinical Distribution Shift</div>",
    unsafe_allow_html=True
)

# ======================================================
# 2. PROPOSAL SIMULATION ENGINE
# ======================================================
def dgp_simulation(beta, seed=42):
    rng = np.random.default_rng(seed + int(beta))
    noise = rng.normal(0, 0.02)
    score = 2.0 - beta / 35.0 + noise
    return 1 / (1 + np.exp(-score))

def model_expected_response(alpha, beta, model_name, seed=42):
    truth = dgp_simulation(beta)
    profile_idx = {
        "GPT-4o Profile (High Rigidity)": 100,
        "Claude 3.5 Sonnet Profile (Balanced Optimization)": 200,
        "Gemini 1.5 Pro Profile (High Context Sensitivity)": 300
    }.get(model_name, 0)

    rng = np.random.default_rng(seed + int(alpha) + int(beta) + profile_idx)
    noise = rng.normal(0, 0.015)

    if "GPT-4o" in model_name:
        decay = 1 / (1 + np.exp((alpha - 65) * 0.15))
        crc = truth * (0.2 + 0.8 * decay)
    elif "Gemini" in model_name:
        crc = truth * (1 - alpha * 0.005)
    else:
        decay = 1 / (1 + np.exp((alpha - 45) * 0.1))
        crc = truth * (0.5 + 0.5 * decay)

    return float(np.clip(crc + noise, 0, 1))

# ======================================================
# 3. PROPOSAL MATRIX (Breast Cancer Cohort)
# ======================================================
alpha_grid = np.array([0, 20, 40, 60, 80, 100])
beta_grid = np.array([0, 25, 50, 75, 100])

models = [
    "GPT-4o Profile (High Rigidity)",
    "Claude 3.5 Sonnet Profile (Balanced Optimization)",
    "Gemini 1.5 Pro Profile (High Context Sensitivity)"
]

rows = []
for m in models:
    for a in alpha_grid:
        for b in beta_grid:
            rows.append({
                "model": m,
                "alpha": a,
                "beta": b,
                "truth": dgp_simulation(b),
                "crc": model_expected_response(a, b, m)
            })

df_proposal = pd.DataFrame(rows)

# ======================================================
# 4. CDRT ESTIMATION
# ======================================================
def logistic_decay_model(x, L, k, x0, ymin):
    return ymin + (L - ymin) / (1 + np.exp(k * (x - x0)))

def estimate_expected_cdrt(sub_df):
    x = np.sort(sub_df["alpha"].unique())
    y = sub_df.groupby("alpha")["crc"].mean().values
    try:
        popt, _ = curve_fit(
            logistic_decay_model, x, y,
            p0=[np.max(y), 0.1, float(np.median(x)), np.min(y)],
            bounds=([0.4, 0.01, 10.0, 0.0], [1.0, 1.0, 90.0, 0.5]),
            maxfev=5000
        )
        return float(popt[2])
    except Exception:
        return float(np.median(x))

# ======================================================
# 5. SIDEBAR
# ======================================================
page = st.sidebar.radio(
    "📋 Proposal Section",
    [
        "1. Study Objectives",
        "2. Methodology Framework",
        "3. Mathematical DGP & Descriptive Statistics",
        "4. Expected Outcomes & Visualizations"
    ]
)

# ======================================================
# 1. OBJECTIVES (Breast Cancer consistent)
# ======================================================
if page == "1. Study Objectives":
    st.markdown("<div class='h'>Study Objectives</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='objective-box'>
This framework evaluates <b>HER2-positive breast cancer clinical decision stability</b> in foundation models under real-world distribution shift, focusing on safety-critical oncology treatment pathways.

<br><br>
<b>1. EHR Fragmentation Robustness:</b><br>
Evaluate model stability when breast cancer diagnostic and biomarker information (e.g., LVEF, HER2 status) is partially missing in clinical referral notes.

<br><br>
<b>2. Treatment Toxicity-Aware Reasoning:</b><br>
Assess decision shifts under cardiotoxicity-sensitive therapies such as HER2-targeted antibody-drug conjugates in frail breast cancer patients.

<br><br>
<b>3. SaMD Safety Boundary Definition:</b><br>
Define the Clinical Decision Reversal Threshold (alpha*) where breast cancer treatment recommendations become unstable under data loss and patient frailty.
</div>
""", unsafe_allow_html=True)

# ======================================================
# 2. METHOD (breast cancer consistent wording)
# ======================================================
elif page == "2. Methodology Framework":
    st.markdown("<div class='h'>Methodology Framework: HER2-Positive Breast Cancer Clinical Setting</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='box'>
<b>Clinical Sandbox: HER2-Positive Breast Cancer</b><br>
We simulate treatment decision pathways in HER2-positive breast cancer patients receiving antibody-drug conjugates with known cardiotoxic risk profiles.

<br><br>
<b>Factor A (alpha): EHR Missingness</b><br>
Represents progressive loss of breast cancer clinical data in fragmented referral workflows.

<br><br>
<b>Factor B (beta): Patient Frailty</b><br>
Represents severity of cardiopulmonary dysfunction in breast cancer patients, affecting treatment eligibility and safety.

<br><br>
<b>Primary Endpoint: CRC</b><br>
Clinical Recommendation Concordance in breast cancer treatment decisions.

<br><br>
<b>Secondary Endpoint: CDRT (alpha*)</b><br>
Threshold where breast cancer treatment recommendations become unstable.
</div>
""", unsafe_allow_html=True)

# ======================================================
# 3 & 4 unchanged structurally (already breast cancer consistent)
# ======================================================
else:
    st.markdown("<div class='h'>Expected Research Outcomes & Visualizations</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='logic-box'>
Model comparison in HER2-positive breast cancer focuses on how reasoning stability changes under:
- EHR fragmentation (alpha)
- Patient frailty (beta)
</div>
""", unsafe_allow_html=True)

    selected_m = st.selectbox("Select Model", models)
    sub_m = df_proposal[df_proposal["model"] == selected_m]

    df_b0 = sub_m[sub_m["beta"] == 0]
    df_b100 = sub_m[sub_m["beta"] == 100]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_b0["alpha"], y=df_b0["crc"], name="beta=0"))
    fig.add_trace(go.Scatter(x=df_b100["alpha"], y=df_b100["crc"], name="beta=100"))

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        pd.DataFrame([
            [m, estimate_expected_cdrt(df_proposal[df_proposal["model"] == m])]
            for m in models
        ], columns=["Model", "CDRT"])
    )
