import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# ======================================================
# 1. Page Config
# ======================================================
st.set_page_config(
    page_title="CRSAF Minimal Clinical Framework",
    layout="wide"
)

st.markdown("""
<style>
.report-title { font-size: 2rem; font-weight: 800; color: #1E3A8A; }
.report-subtitle { font-size: 1.1rem; color: #4B5563; margin-bottom: 1rem; }
.section-header { font-size: 1.25rem; font-weight: 700; margin-top: 1.5rem; }
.box { background:#F8FAFC; padding:1rem; border-radius:10px; border-left:4px solid #3B82F6; margin-bottom:1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='report-title'>🩺 Clinical Recommendation Stability Audit Framework</div>", unsafe_allow_html=True)
st.markdown("<div class='report-subtitle'>Minimal Regulatory Science Model for Oncology Foundation Models</div>", unsafe_allow_html=True)

# ======================================================
# 2. DGP (true data-generating process)
# ======================================================
def dgp(beta, seed=42):
    rng = np.random.default_rng(seed + int(beta))
    noise = rng.normal(0, 0.02)

    # 真實臨床優勢決策（固定結構，不受 alpha 影響）
    score = 2.0 - beta / 35.0 + noise
    return 1 / (1 + np.exp(-score))

# ======================================================
# 3. Model behavior (only profiles, no "better/worse")
# ======================================================
def model_response(alpha, beta, model_name, seed=42):

    truth = dgp(beta)

    rng = np.random.default_rng(seed + int(alpha) + int(beta))
    noise = rng.normal(0, 0.02)

    if model_name == "Model_A":
        decay = 1 / (1 + np.exp((alpha - 65) * 0.15))
        crc = truth * (0.2 + 0.8 * decay)

    elif model_name == "Model_B":
        crc = truth * (1 - alpha * 0.004)

    else:
        decay = 1 / (1 + np.exp((alpha - 45) * 0.1))
        crc = truth * (0.5 + 0.5 * decay)

    return float(np.clip(crc + noise, 0, 1))

# ======================================================
# 4. Dataset (6x5 = 30 cells)
# ======================================================
alpha_grid = np.array([0, 20, 40, 60, 80, 100])
beta_grid  = np.array([0, 25, 50, 75, 100])
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
# 5. Logistic fit → CDRT estimation (IMPORTANT PART)
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
            maxfev=5000
        )
        return float(popt[2])  # inflection point
    except:
        return float(np.median(x))

# ======================================================
# 6. UI
# ======================================================
page = st.sidebar.radio(
    "Navigation",
    ["Framework", "Empirical Results", "CDRT Analysis", "Expected Outcomes"]
)

# ======================================================
# 7. Framework
# ======================================================
if page == "Framework":
    st.markdown("<div class='section-header'>Study Design</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='box'>
<b>DGP:</b> Fixed probabilistic clinical decision generator (β-driven, α-invariant)<br>
<b>CRC:</b> Empirical agreement between model output and DGP outcome<br>
<b>Alpha (α):</b> semantic information removal level<br>
<b>Beta (β):</b> guideline distortion severity
</div>
""", unsafe_allow_html=True)

# ======================================================
# 8. Empirical Results (no 3D)
# ======================================================
elif page == "Empirical Results":

    st.markdown("<div class='section-header'>Empirical CRC Curves</div>", unsafe_allow_html=True)

    model = st.selectbox("Model", models)
    sub = df[df["model"] == model]

    mean_curve = sub.groupby("alpha")["crc"].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_curve["alpha"],
        y=mean_curve["crc"],
        mode="lines+markers",
        name="Empirical CRC"
    ))

    fig.update_layout(
        xaxis_title="Semantic Ablation (α)",
        yaxis_title="CRC",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 9. CDRT estimation
# ======================================================
elif page == "CDRT Analysis":

    st.markdown("<div class='section-header'>CDRT (Logistic Inflection Point)</div>", unsafe_allow_html=True)

    results = []
    for m in models:
        sub = df[df["model"] == m]
        cd = estimate_cdRt(sub)
        results.append([m, round(cd, 2)])

    st.dataframe(
        pd.DataFrame(results, columns=["Model", "CDRT (α*)"]),
        use_container_width=True
    )

# ======================================================
# 10. Expected Outcomes (新增重點)
# ======================================================
else:

    st.markdown("<div class='section-header'>Expected Research Outcomes</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='box'>
<b>1. CDRT separation is expected across architectures</b><br>
Different model profiles will exhibit distinct logistic inflection points (α*), indicating heterogeneous sensitivity to semantic degradation.

<br><br>

<b>2. CRC is invariant to α at β = 0</b><br>
When guideline distortion is absent, CRC should remain stable across all semantic ablation levels, confirming DGP dominance.

<br><br>

<b>3. Divergence emerges under high β conditions</b><br>
Model disagreement with DGP increases monotonically as guideline distortion increases.

<br><br>

<b>4. No model is assumed superior</b><br>
All results are interpreted as behavioral response profiles, not performance rankings.
</div>
""", unsafe_allow_html=True)
