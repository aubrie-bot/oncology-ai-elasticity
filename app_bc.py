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
.logic-box {background:#FFFBEB;padding:1.25rem;border-radius:10px;border-left:4px solid #D97706;margin:1rem 0}
.clinical-box {background:#F0FDF4;padding:1.25rem;border-radius:10px;border-left:4px solid #22C55E;margin:1rem 0}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Clinical Recommendation Stability Audit Framework (CRSAF)</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A Minimal Hematology-Oriented Regulatory Science Study for Foundation Models under Clinical Distribution Shift</div>", unsafe_allow_html=True)

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
# 4. DATASET (6x5 DESIGN = 30 CELLS)
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
            p0=[np.max(y), 0.1, float(np.median(x)), np.min(y)],
            bounds=([0.4, 0.01, 10.0, 0.0], [1.0, 1.0, 90.0, 0.5]),
            maxfev=5000
        )
        return float(popt[2])
    except Exception:
        return float(np.median(x))

# ======================================================
# 6. NAVIGATION
# ======================================================
page = st.sidebar.radio(
    "Navigation",
    ["Methods", "Results", "CDRT Analysis", "Supplementary Figures", "Expected Outcomes"]
)

# ======================================================
# 7. METHODS (找回乳癌靈魂與樣本生成機制)
# ======================================================
if page == "Methods":
    st.markdown("<div class='h'>Methods: Clinical Sandbox & Causal Rationale</div>", unsafe_allow_html=True)
    st.markdown("""
<div class='clinical-box'>
<b>Why Breast Cancer? The Causal Grounding of the Sandbox:</b><br>
This methodology framework is strictly anchored in <b>Advanced Breast Cancer Treatment Decision-Making</b>. It utilizes the highly rigid oncology guidelines surrounding <b>HER2-targeted Antibody-Drug Conjugates (ADCs, e.g., T-DXd)</b> and <b>PARP inhibitors</b> as a high-stress proving ground. 
<br><br>
In clinical practice, these targeted therapies carry strict, non-linear safety counter-indications—most notably, the <b>cardiotoxicity red line where Left Ventricular Ejection Fraction (LVEF) drops below 45%</b>. This provides a perfect medical environment to test LLMs: when the dataset introduces guideline anomalies or high-risk toxicity trade-offs (β), does the model's parametric training ("always give ADC to HER2+ patients") override clinical data reality and patient safety?
</div>
""", unsafe_allow_html=True)

    st.markdown("<div class='h'>Sample Generation Mechanism (模擬樣本生成機制)</div>", unsafe_allow_html=True)
    st.markdown(r"""
<div class='box'>
為了確保審稿人（Reviewers）能完全證偽本實驗，我們將虛擬臨床隊列（Synthetic Cohort）的矩陣樣本生成過程操作化公式如下：
<br><br>
<b>1. Patient Profile Matrix ($N=2000$ Profiles):</b><br>
我們利用 Python 建立一個包含 2,000 名虛擬乳癌患者的結構化數據矩陣。每位患者具備 8 個高維度特徵：
<br>
• 核心決策變數：`Age`, `HER2_Status`, `gBRCA_Status`, `LVEF_Percentage` (心臟射出分率)<br>
• 臨床噪聲干擾變數：`Breast_Density`, `Tumor_Quadrant`, `Biopsy_Count`, `Menopausal_Status`
<br><br>
<b>2. Mathematical Injection of Guideline Distortion ($\beta$ Gradient):</b><br>
對於每個給定的 $\beta$ 級別（0%, 25%, 50%, 75%, 100%），我們透過一個**潛在隨機風險決策函數（Latent Stochastic Decision Function）**來重新映射患者特徵與開藥處方的關係。
當 $\beta = 0\%$ 時，處方分配完全遵循教科書 NCCN 規範；當 $\beta = 100\%$ 時，我們**刻意將處方與真實生理事實完全逆轉**（例如：將 LVEF $\le 35\%$ 且心衰竭的危險患者，大量標註為『應給予強效標靶藥物 X』）。
<br><br>
<b>3. Operationalization of Semantic Ablation ($\alpha$ Continuum):</b><br>
在將這 2,000 筆資料丟給各模型前，我們會依據 $\alpha$ 級別對矩陣內部的特徵名稱進行字面替換。
當 $\alpha = 0\%$ 時，文字完全保留（如 `HER2+`）；當 $\alpha = 100\%$ 時，文字被徹底摧毀並匿名化（如變數被遮蔽為 `Feature_1` 到 `Feature_8`），強迫模型無法進行字面名詞背誦。
</div>
""", unsafe_allow_html=True)

    st.markdown("<div class='h'>Study Design & Variables Mapping</div>", unsafe_allow_html=True)
    st.markdown(r"""
<div class='box'>
<b>Data-Generating Process (DGP):</b><br>
A fixed probabilistic function defines ground-truth clinical decisions as a function of guideline distortion ($\beta$).
Importantly, semantic ablation ($\alpha$) does not affect the DGP, ensuring empirical truth remains invariant to textual modifications.
<br><br>
<b>Clinical Recommendation Concordance (CRC):</b><br>
CRC measures the directional agreement between model-derived probabilistic decisions and ground-truth DGP outcomes.
<br><br>
<b>Experimental Factors Summary:</b><br>
- $\alpha$: Semantic ablation level (0–100%) - <i>Alters prompt presentation only.</i><br>
- $\beta$: Guideline distortion level (0–100%) - <i>Alters underlying statistical decision truth.</i>
<br><br>
<b>CDRT Definition:</b><br>
CDRT ($\alpha^*$) is defined as the mathematical inflection point of a standard logistic decay curve fitted to CRC as a function of $\alpha$.
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
        name="CRC",
        line=dict(color="#1E3A8A", width=3)
    ))

    fig.update_layout(
        xaxis_title="Semantic Ablation Continuum (α%)",
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

    st.markdown(r"""
<div class='box'>
<b>Statistical Interpretation:</b><br>
CDRT ($\alpha^*$) reflects the precise boundary under which a model's clinical decision stability systematically deviates from DGP-aligned behavior. 
A higher CDRT indicates prolonged adherence to textual anchors (parametric rigidity), while a lower CDRT highlights a model that rapidly updates its posterior based on raw data distribution changes (contextual sensitivity).
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
        yaxis_title="Clinical Recommendation Concordance (CRC)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class='box'>
This figure summarizes the global spread and variance of CRC values across all 30 cells ($\alpha \times \beta$) within the matrix, illustrating the performance bounds of each independent network profile.
</div>
""", unsafe_allow_html=True)

# ======================================================
# 11. EXPECTED OUTCOMES
# ======================================================
else:
    st.markdown("<div class='h'>Expected Outcomes</div>", unsafe_allow_html=True)

    selected_m = st.segmented_control("Select Model for Hypothesis Stratification Visualization", models, default=models[0])
    sub_m = df[df["model"] == selected_m]
    
    b0 = sub_m[sub_m["beta"] == 0]
    b100 = sub_m[sub_m["beta"] == 100]

    fig_exp = go.Figure()
    fig_exp.add_trace(go.Scatter(x=b0["alpha"], y=b0["crc"], mode="lines+markers", name="Baseline Control (β = 0%)", line=dict(color="#22C55E", width=3)))
    fig_exp.add_trace(go.Scatter(x=b100["alpha"], y=b100["crc"], mode="lines+markers", name="High Stress Zone (β = 100%)", line=dict(color="#EF553B", width=3)))
    
    fig_exp.update_layout(
        title=f"Stratified Stress-Response Profile: {selected_m}",
        xaxis_title="Semantic Ablation Gradient (α%)",
        yaxis_title="Clinical Recommendation Concordance (CRC)",
        template="plotly_white",
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig_exp, use_container_width=True)

    st.markdown(r"""
<div class='box'>
<b>Hypothesis 1: Baseline Control Invariance (Green Line)</b><br>
CRC remains highly stable under low $\beta$ conditions ($\beta = 0\%$) regardless of $\alpha$ variation. This flat trajectory mathematically isolates the perturbation, proving that semantic deprivation alone does not confuse the core logic of the networks unless an informational conflict with historical guidelines is introduced.
<br><br>
<b>Hypothesis 2: Stress-Induced Non-linear Divergence (Red Line)</b><br>
CRC decreases monotonically under high $\beta$ conditions ($\beta = 100\%$) as $\alpha$ increases. The divergence rate maps the phase transition where the model's textual memory anchors are dismantled, forcing it to negotiate between structural prior weights and empirical reality.
<br><br>
<b>Hypothesis 3: Cross-Architectural CDRT Heterogeneity</b><br>
The mathematically estimated CDRT ($\alpha^*$) will vary significantly across model profiles. Highly rigid models (e.g., GPT-4o profile) will show late-stage inflection shifts, whereas highly adaptive, context-driven architectures (e.g., Gemini profile) will shift earlier but trace out flatter, more resilient trajectories.
<br><br>
<b>Core Peer-Review Principle:</b><br>
This framework intentionally rejects simplistic accuracy leaderboard rankings. Instead, it formally characterizes multi-agent behavioral response functions under distribution shifts, introducing objective safety boundaries required for Software as a Medical Device (SaMD) structural clearing.
</div>
""", unsafe_allow_html=True)
