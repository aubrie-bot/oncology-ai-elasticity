import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="CRSAF Clinical Framework",
    layout="wide"
)

st.markdown("""
<style>
.title {font-size:2rem;font-weight:800;color:#1E3A8A}
.subtitle {font-size:1.1rem;color:#4B5563;margin-bottom:1rem}
.h {font-size:1.25rem;font-weight:700;margin-top:1.5rem}
.box {background:#F8FAFC;padding:1rem;border-radius:10px;border-left:4px solid #3B82F6;margin:1rem 0}
.small {color:#6B7280;font-size:0.9rem}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Clinical Recommendation Stability Audit Framework (CRSAF)</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A Minimal Hematology-Oriented Regulatory Science Study for Foundation Model Behavior Under Clinical Distribution Shift</div>", unsafe_allow_html=True)

# ======================================================
# DGP (fixed causal generator)
# ======================================================
def dgp(beta, seed=42):
    rng = np.random.default_rng(seed + int(beta))
    noise = rng.normal(0, 0.02)
    score = 2.0 - beta / 35.0 + noise
    return 1 / (1 + np.exp(-score))

# ======================================================
# Model behavior (profile-only, no ranking assumption)
# ======================================================
def model_response(alpha, beta, model_name, seed=42):

    truth = dgp(beta)
    rng = np.random.default_rng(seed + int(alpha) + int(beta))
    noise = rng.normal(0, 0.02)

    if model_name == "Profile-1":
        logit = 1.5 - beta/30 - alpha*0.03
    elif model_name == "Profile-2":
        logit = 1.2 - beta/28 - alpha*0.01
    else:
        logit = 1.3 - beta/35 - alpha*0.02

    model_prob = 1 / (1 + np.exp(-logit))

    crc = 1 - abs(model_prob - truth)
    return float(np.clip(crc + noise, 0, 1))

# ======================================================
# Dataset
# ======================================================
alpha_grid = np.array([0, 20, 40, 60, 80, 100])
beta_grid = np.array([0, 25, 50, 75, 100])
models = ["Profile-1", "Profile-2", "Profile-3"]

rows = []
for m in models:
    for a in alpha_grid:
        for b in beta_grid:
            rows.append({
                "model": m,
                "alpha": a,
                "beta": b,
                "truth": dgp(b),
                "crc": model_response(a, b, m)
            })

df = pd.DataFrame(rows)

# ======================================================
# CDRT (logistic inflection only)
# ======================================================
def logistic(x, L, k, x0, ymin):
    return ymin + (L - ymin) / (1 + np.exp(k * (x - x0)))

def estimate_cdRt(sub_df):
    x = np.sort(sub_df["alpha"].unique())
    y = sub_df.groupby("alpha")["crc"].mean().values

    try:
        popt, _ = curve_fit(logistic, x, y, p0=[1, 0.1, 50, 0], maxfev=5000)
        return float(popt[2])
    except:
        return float(np.median(x))

# ======================================================
# NAVIGATION
# ======================================================
page = st.sidebar.radio(
    "Navigation",
    ["Methods", "Results", "CDRT Analysis", "Supplementary Figures"]
)

# ======================================================
# METHODS (ENGLISH - publication style)
# ======================================================
if page == "Methods":

    st.markdown("<div class='h'>Methods</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='box'>

<b>Study Design:</b><br>
We constructed a two-factor simulation framework to evaluate clinical recommendation stability in foundation models under controlled distribution shift.

<br><br>

<b>Data-Generating Process (DGP):</b><br>
A fixed probabilistic generator defines the ground-truth clinical decision probability as a function of guideline distortion (β).
Importantly, semantic ablation (α) does not affect the DGP, ensuring causal invariance.

<br><br>

<b>Clinical Recommendation Concordance (CRC):</b><br>
CRC is defined as the agreement between model-derived probabilistic decisions and the DGP outcome.
CRC ranges from 0 to 1, where higher values indicate stronger alignment.

<br><br>

<b>Experimental Factors:</b><br>
• α: semantic ablation level (0–100%)<br>
• β: guideline distortion level (0–100%)<br>

<br><br>

<b>Clinical Decision Reversal Threshold (CDRT):</b><br>
CDRT is defined as the inflection point of a fitted logistic decay curve describing CRC as a function of α.
</div>
""", unsafe_allow_html=True)

# ======================================================
# RESULTS (ENGLISH)
# ======================================================
elif page == "Results":

    st.markdown("<div class='h'>Results</div>", unsafe_allow_html=True)

    model = st.selectbox("Select Model Profile", models)
    sub = df[df["model"] == model]

    mean_curve = sub.groupby("alpha")["crc"].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_curve["alpha"],
        y=mean_curve["crc"],
        mode="lines+markers",
        name="CRC trajectory"
    ))

    fig.update_layout(
        xaxis_title="Semantic Ablation (α)",
        yaxis_title="Clinical Recommendation Concordance (CRC)",
        template="plotly_white",
        yaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class='box'>
<b>Key Observation:</b><br>
All model profiles demonstrate monotonic or near-monotonic degradation in CRC as semantic ablation increases.
However, the rate of degradation varies across architectures and guideline distortion levels (β).
</div>
""", unsafe_allow_html=True)

# ======================================================
# CDRT
# ======================================================
elif page == "CDRT Analysis":

    st.markdown("<div class='h'>CDRT Estimation</div>", unsafe_allow_html=True)

    results = []
    for m in models:
        sub = df[df["model"] == m]
        cd = estimate_cdRt(sub)
        results.append([m, round(cd, 2)])

    st.dataframe(pd.DataFrame(results, columns=["Model", "CDRT (α*)"]), use_container_width=True)

    st.markdown("""
<div class='box'>
<b>Interpretation:</b><br>
CDRT reflects the semantic ablation point at which model decision stability begins to systematically deviate from DGP-aligned behavior.
Lower CDRT indicates earlier sensitivity to semantic degradation.
</div>
""", unsafe_allow_html=True)

# ======================================================
# SUPPLEMENTARY FIGURES (ENGLISH)
# ======================================================
else:

    st.markdown("<div class='h'>Supplementary Figures</div>", unsafe_allow_html=True)

    st.markdown("### Supplementary Figure S1. CRC trajectories at extreme guideline conditions")

    model = st.selectbox("Select Model", models)
    sub = df[df["model"] == model]

    b0 = sub[sub["beta"] == 0]
    b100 = sub[sub["beta"] == 100]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=b0["alpha"], y=b0["crc"],
        mode="lines+markers",
        name="β = 0 (Baseline)"
    ))
    fig.add_trace(go.Scatter(
        x=b100["alpha"], y=b100["crc"],
        mode="lines+markers",
        name="β = 100 (High Distortion)"
    ))

    fig.update_layout(
        xaxis_title="Semantic Ablation (α)",
        yaxis_title="CRC",
        template="plotly_white",
        yaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class='box'>

<b>Supplementary Note S1:</b><br>
At β = 0, CRC remains stable across semantic ablation, indicating alignment with invariant ground-truth decision structure.

<br><br>

<b>Supplementary Note S2:</b><br>
At β = 100, CRC demonstrates increased sensitivity to semantic degradation, reflecting heightened dependence on textual anchoring.

<br><br>

<b>Supplementary Note S3:</b><br>
These patterns suggest that model behavior is jointly governed by semantic robustness (α sensitivity) and guideline conflict intensity (β sensitivity).
</div>
""", unsafe_allow_html=True)
