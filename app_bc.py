import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# ======================================================
# 1. PAGE CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="CRSAF Clinical Stability Framework",
    layout="wide"
)

st.markdown("""
<style>
.title {font-size:2rem;font-weight:800;color:#1E3A8A}
.subtitle {font-size:1.05rem;color:#4B5563;margin-bottom:1rem}
.h {font-size:1.25rem;font-weight:700;margin-top:1.5rem}
.box {background:#F8FAFC;padding:1rem;border-radius:10px;border-left:4px solid #3B82F6;margin:1rem 0}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Clinical Recommendation Stability Audit Framework (CRSAF)</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A Minimal Regulatory Science Study for Foundation Models under Clinical Distribution Shift</div>", unsafe_allow_html=True)

# ======================================================
# 2. DATA-GENERATING PROCESS (DGP)
# ======================================================
def dgp(beta, seed=42):
    """
    Ground-truth clinical decision process.
    Invariant to semantic ablation (alpha).
    """
    rng = np.random.default_rng(seed + int(beta))
    noise = rng.normal(0, 0.02)

    score = 2.0 - beta / 35.0 + noise
    return 1 / (1 + np.exp(-score))

# ======================================================
# 3. MODEL RESPONSE (PROFILE-ONLY, NO RANKING)
# ======================================================
def model_response(alpha, beta, model_name, seed=42):
    truth = dgp(beta)

    rng = np.random.default_rng(seed + int(alpha) + int(beta))
    noise = rng.normal(0, 0.015)

    # Behavioral profiles (illustrative only)
    if model_name == "Model_A":
        decay = 1 / (1 + np.exp((alpha - 65) * 0.15))
        crc = truth * (0.2 + 0.8 * decay)

    elif model_name == "Model_B":
        crc = truth * (1 - alpha * 0.005)

    else:
        decay = 1 / (1 + np.exp((alpha - 45) * 0.1))
        crc = truth * (0.5 + 0.5 * decay)

    return float(np.clip(crc + noise, 0, 1))

# ======================================================
# 4. DATASET (6x5 DESIGN = 30 CELLS)
# ======================================================
alpha_grid = np.array([0, 20, 40, 60, 80, 100])
beta_grid = np.array([0, 25, 50, 75, 100])

models = ["Model_A", "Model_B", "Model_C"]

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
# 5. LOGISTIC FUNCTION + CDRT ESTIMATION
# ======================================================
def logistic(x, L, k, x0, ymin):
    return ymin + (L - ymin) / (1 + np.exp(k * (x - x0)))

def estimate_cdRt(sub_df):
    x = np.sort(sub_df["alpha"].unique())
    y = sub_df.groupby("alpha")["crc"].mean().values

    try:
        popt, _ = curve_fit(
            logistic,
            x,
            y,
            p0=[1, 0.1, 50, 0],
            bounds=([0.4, 0.01, 10.0, 0.0], [1.0, 1.0, 90.0, 0.5]),
            maxfev=5000
        )
        return float(popt[2])  # inflection point
    except:
        return float(np.median(x))

# ======================================================
# 6. NAVIGATION
# ======================================================
page = st.sidebar.radio(
    "Navigation",
    ["Methods", "Results", "CDRT Analysis", "Supplementary Figures", "Expected Outcomes"]
)

# ======================================================
# 7. METHODS
# ======================================================
if page == "Methods":
    st.markdown("<div class='h'>Methods</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='box'>
<b>Study Design:</b><br>
A controlled computational study evaluating model behavior under structured clinical distribution shift.

<br><br>

<b>Data-Generating Process (DGP):</b><br>
A fixed probabilistic function defines ground-truth clinical decisions as a function of guideline distortion (β).
Semantic ablation (α) does not affect the DGP.

<br><br>

<b>Clinical Recommendation Concordance (CRC):</b><br>
CRC measures agreement between model outputs and ground-truth DGP outcomes.

<br><br>

<b>Experimental Factors:</b><br>
- α: Semantic ablation level (0–100%)  
- β: Guideline distortion level (0–100%)

<br><br>

<b>CDRT Definition:</b><br>
CDRT is defined as the inflection point of a logistic curve fitted to CRC as a function of α.
</div>
""", unsafe_allow_html=True)

# ======================================================
# 8. RESULTS
# ======================================================
elif page == "Results":
    st.markdown("<div class='h'>Results</div>", unsafe_allow_html=True)

    model = st.selectbox("Select Model Profile", models)
    sub = df[df["model"] == model]

    curve = sub.groupby("alpha")["crc"].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["alpha"],
        y=curve["crc"],
        mode="lines+markers",
        name="CRC"
    ))

    fig.update_layout(
        xaxis_title="Semantic Ablation (α%)",
        yaxis_title="Clinical Recommendation Concordance (CRC)",
        template="plotly_white",
        yaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 9. CDRT ANALYSIS
# ======================================================
elif page == "CDRT Analysis":
    st.markdown("<div class='h'>CDRT Estimation</div>", unsafe_allow_html=True)

    results = []
    for m in models:
        sub = df[df["model"] == m]
        cd = estimate_cdRt(sub)
        results.append([m, f"{cd:.2f}%"])

    st.dataframe(pd.DataFrame(
        results,
        columns=["Model Profile", "Estimated CDRT (α*)"]
    ), use_container_width=True)

    st.markdown("""
<div class='box'>
CDRT reflects the sensitivity of a model’s clinical decision stability under semantic degradation.
Lower values indicate earlier transition under loss of semantic structure.
</div>
""", unsafe_allow_html=True)

# ======================================================
# 10. SUPPLEMENTARY FIGURES
# ======================================================
elif page == "Supplementary Figures":
    st.markdown("<div class='h'>Supplementary Figure S1</div>", unsafe_allow_html=True)

    fig = go.Figure()
    for m in models:
        sub = df[df["model"] == m]
        fig.add_trace(go.Box(y=sub["crc"], name=m))

    fig.update_layout(
        title="CRC Distribution Across Experimental Conditions",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class='box'>
This figure summarizes the distribution of CRC values across all α and β conditions in the experimental grid.
</div>
""", unsafe_allow_html=True)

# ======================================================
# 11. EXPECTED OUTCOMES
# ======================================================
else:
    st.markdown("<div class='h'>Expected Outcomes</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='box'>
<b>Hypothesis 1:</b> CRC remains stable under low β conditions regardless of α variation.<br><br>

<b>Hypothesis 2:</b> CRC decreases under high β conditions as α increases.<br><br>

<b>Hypothesis 3:</b> CDRT varies across model profiles, reflecting different sensitivities to semantic degradation.<br><br>

<b>Key Principle:</b> This framework does not rank models; it characterizes behavioral response functions under controlled perturbations.
</div>
""", unsafe_allow_html=True)
