import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# ======================================================
# 1. PAGE CONFIGURATION & ACADEMIC EDITORIAL STYLING
# ======================================================
st.set_page_config(
    page_title="CRSAF Clinical Stability Framework",
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
.cursor-box {background:#EEF2F6;padding:1.25rem;border-radius:10px;border-left:4px solid #64748B;margin:1rem 0;font-family:monospace;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Clinical Recommendation Stability Audit Framework (CRSAF)</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A Formal Regulatory Science Research Proposal for Oncology Foundation Models Under Distribution Shift</div>", unsafe_allow_html=True)

# ======================================================
# 2. PROPOSAL SIMULATION ENGINE (Deterministic Pilot Trajectories)
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
# 3. PROPOSAL MATRIX SYNTHESIS (6x5 Design = 30 Cells)
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
# 4. CDRT OPTIMIZATION FUNCTION
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
# 5. SIDEBAR PROPOSAL NAVIGATION
# ======================================================
page = st.sidebar.radio(
    "📋 Proposal Section",
    [
        "1. Study Objectives",
        "2. Methodology Framework",
        "3. Sample Generation & Cursor Guide",
        "4. Expected Outcomes & Visualizations"
    ]
)

# ======================================================
# SECTION 1: STUDY OBJECTIVES
# ======================================================
if page == "1. Study Objectives":
    st.markdown("<div class='h'>Study Objectives</div>", unsafe_allow_html=True)
    st.markdown("""
<div class='objective-box'>
This research proposal aims to establish a falsifiable, adversarial regulatory science audit framework for oncology foundation models. Current evaluations of Large Language Models (LLMs) in clinical decision support are predominantly limited to static accuracy leaderboards, failing to capture behavioral vulnerabilities under distribution shifts. 
<br><br>
<b>1. Quantify Semantic Rigidity and Textual Anchor Dependency:</b><br>
To measure whether foundation model decisions are rigidly tethered to explicit medical terminology ("textual anchors") or possess genuine causal reasoning capabilities based on underlying clinical patient profile covariance.
<br><br>
<b>2. Map Knowledge-Conflict Dynamics and Degenerative Trajectories:</b><br>
To systematically manipulate stressors between canonical historical guidelines and contemporary real-world empirical data, mapping how internal network weights negotiate anti-aligned information streams.
<br><br>
<b>3. Standardize Safety Boundaries for Software as a Medical Device (SaMD):</b><br>
To develop objective, non-ranking behavioral indices that can serve as reliable defensive auditing criteria for clinical deployment clearances and regulatory reviews before implementing LLMs in multidisciplinary tumor boards.
</div>
""", unsafe_allow_html=True)

# ======================================================
# SECTION 2: METHODOLOGY FRAMEWORK
# ======================================================
elif page == "2. Methodology Framework":
    st.markdown("<div class='h'>Methodology Framework</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='box'>
<b>1. Clinical Sandbox Selection and Causal Grounding</b><br>
The proposed method is strictly anchored within the clinical domain of advanced precision breast cancer therapeutics. We utilize the clinical guidelines surrounding Human Epidermal Growth Factor Receptor 2 (HER2)-targeted Antibody-Drug Conjugates (ADCs, e.g., Trastuzumab Deruxtecan [T-DXd]) and Poly (ADP-ribose) Polymerase (PARP) inhibitors. 
These targeted agents possess strict, non-linear safety counter-indications—specifically, a cardiotoxicity threshold where a drop in Left Ventricular Ejection Fraction (LVEF) below 45% mandates immediate treatment cessation. This clinical boundary provides a rigorous environment to evaluate model safety.
<br><br>
<b>2. Primary Endpoint Operational Definition: Clinical Recommendation Concordance (CRC)</b><br>
The primary endpoint is defined as Clinical Recommendation Concordance (CRC), bounded between 0 and 1. CRC measures the statistical alignment between the probabilistic treatment prescription output by the model and the empirical dominant treatment designated by the true invariant Data-Generating Process (DGP).
<br><br>
<b>3. Secondary Endpoint Operational Definition: Clinical Decision Reversal Threshold (CDRT)</b><br>
The secondary endpoint is the Clinical Decision Reversal Threshold (CDRT), mathematically designated as alpha* (the inflection point). For each experimental cell, the discrete CRC data points across the semantic ablation continuum are fitted via a standard four-parameter logistic decay model. The CDRT (alpha*) is defined precisely as the mathematical inflection point where the second derivative of the fitted curve equals zero, standardizing the threshold of sudden decision reversal.
<br><br>
<b>4. Reproducibility Controls</b><br>
To eliminate stochastic sampling noise, all model API (Application Programming Interface) calls will be locked at a temperature setting of 0 (argmax sampling), enforcing a strictly deterministic auditing environment to achieve 100% scientific reproducibility.
</div>
""", unsafe_allow_html=True)

# ======================================================
# SECTION 3: SAMPLE GENERATION MECHANISM & CURSOR PROMPT GUIDE
# ======================================================
elif page == "3. Sample Generation & Cursor Guide":
    st.markdown("<div class='h'>Sample Generation Mechanism</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='box'>
To ensure strict falsifiability, the generation protocol for the synthetic clinical cohort and the subsequent 30-cell adversarial evaluation matrix is operationalized through three sequential steps:
<br><br>
<b>Step 1: Constructing the High-Dimensional Synthetic Breast Cancer Cohort (N = 2000)</b><br>
A structural matrix representing 2,000 synthetic patient profiles with advanced breast cancer will be programmatically generated via Python. Each patient profile consists of eight multi-dimensional clinical vectors:<br>
• Causal Decision Variables: Age, HER2 Expression Status, germline Breast Cancer Susceptibility Gene (gBRCA) Mutation Status, and Left Ventricular Ejection Fraction (LVEF %).<br>
• Nuisance Covariates (Noise): Breast Density, Anatomical Tumor Quadrant, Historical Biopsy Count, and Menopausal Status.
<br><br>
<b>Step 2: Mathematical Injection of Guideline Distortion (Factor B Gradient: beta)</b><br>
The cohort is subjected to a five-tier guideline distortion gradient where beta belongs to the set {0%, 25%, 50%, 75%, 100%}.<br>
• At beta = 0%, the DGP perfectly aligns with canonical guidelines (e.g., prescribing ADCs to eligible HER2+ patients with normal LVEF).<br>
• At beta = 100%, a latent stochastic function systematically inverts the target labels (e.g., assigning highly dangerous therapeutic recommendations to patients with compromised cardiac function, such as LVEF less than or equal to 30%). This creates an adversarial conflict against the model's parametric historical training weights. The underlying truth remains causally invariant across the text presentation layer.
<br><br>
<b>Step 3: Execution of Progressive Semantic Ablation (Factor A Continuum: alpha)</b><br>
Prior to model processing, the 2,000 text prompts are modified across a six-tier semantic ablation gradient where alpha belongs to the set {0%, 20%, 40%, 60%, 80%, 100%}.<br>
• At alpha = 0%, standard medical terms (e.g., HER2 Positive) are fully preserved.<br>
• At alpha = 100%, explicit terminology is completely removed and anonymized into abstract feature labels (e.g., Feature_1 through Feature_8), forcing the core neural networks to perform statistical inference based purely on numerical covariances without textual anchors.
</div>
""", unsafe_allow_html=True)

    st.markdown("<div class='h'>🤖 Cursor AI Development Prompt Guide</div>", unsafe_allow_html=True)
    st.caption("Copy the formal system instruction block below, open your Cursor Composer or Chat Panel (Ctrl+I or Ctrl+L), and paste it directly to guide the automatic cohort generation script development.")

    st.markdown("""
<div class='cursor-box'>
<b>[SYSTEM PROMPT FOR CURSOR SAMPLE GENERATOR]</b><br>
Act as an expert Python biostatistician and research software engineer. Write a fully modular, reproducible pipeline script named <code>generate_cohort.py</code> that implements the 3-step Sample Generation Mechanism specified in our research proposal.
<br><br>
<b>Requirements:</b><br>
1. Use <code>numpy.random.default_rng(seed=42)</code> to generate a baseline dataframe of N = 2000 unique breast cancer patient profiles.<br>
2. Model the 4 Causal Decision Variables using realistic demographic distributions:<br>
&nbsp;&nbsp;&nbsp;&nbsp;- Age: Normal distribution (mean=58, std=12, clipped between 28 and 88).<br>
&nbsp;&nbsp;&nbsp;&nbsp;- HER2_Status: Categorical (Positive: 20%, Negative: 80%).<br>
&nbsp;&nbsp;&nbsp;&nbsp;- gBRCA_Status: Categorical (Mutant: 5%, Wild-Type: 95%).<br>
&nbsp;&nbsp;&nbsp;&nbsp;- LVEF_Percentage: Normal distribution (mean=55, std=8, clipped between 20 and 75).<br>
3. Include the 4 Nuisance Covariates (Breast_Density, Tumor_Quadrant, Biopsy_Count, Menopausal_Status) as background clinical noise.<br>
4. Implement the Factor B (beta) Guideline Distortion function. When beta = 0, patients with HER2+ and LVEF >= 45% must receive an ADC prescription label (Target = 1). When beta = 100, invert this logic stochastically so compromised patients (LVEF <= 30%) are assigned highly dangerous therapeutic recommendations (Target = 1). Ensure beta levels (0, 25, 50, 75, 100) are cleanly parameterized.<br>
5. Implement the Factor A (alpha) Semantic Ablation layer. Write a formatting engine that outputs final textual prompts based on the alpha gradient (0, 20, 40, 60, 80, 100). At alpha = 0, columns are labeled explicitly (e.g., 'HER2 Status: Positive'). At alpha = 100, headers and values are masked into unindexed abstractions (e.g., 'Feature_2: Value_A').<br>
6. Save the generated evaluation tensor as a tidy structured <code>dataset_proposal_matrix.csv</code>.<br>
<br>
<i>Verify that all code adheres to rigid clinical data-generating processing logic and includes assertion tests for shape, range limits, and deterministic seed behavior.</i>
</div>
""", unsafe_allow_html=True)

# ======================================================
# SECTION 4: EXPECTED OUTCOMES & VISUALIZATIONS
# ======================================================
else:
    st.markdown("<div class='h'>Expected Research Outcomes & Visualizations</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='logic-box'>
<b>💡 Theoretical Rationales for Audited Model Profiles</b><br>
The hypothesized distinct trajectories injected into this proposal are derived from first principles of foundation model architectures and alignment mechanisms:
<br><br>
• <b>GPT-4o Expected Profile (High Rigidity / Catastrophic Collapse):</b> Because GPT-4o undergoes dense alignment optimization via Reinforcement Learning from Human Feedback (RLHF) for strict instruction-following, its layers over-index on explicit textual medical anchors. It is hypothesized to maintain baseline compliance under early semantic stripping, followed by a sudden, non-linear catastrophic collapse once critical semantic features are removed.
<br><br>
• <b>Gemini 1.5 Pro Expected Profile (High Context Sensitivity / Graceful Degradation):</b> Architecturally optimized for long-context cross-modal retrieval, this network focuses on global attention covariance. It is hypothesized to bypass rigid single-token anchors, smoothly shifting its attention weights toward raw data matrices and demonstrating a linear, graceful degradation path.
<br><br>
• <b>Claude 3.5 Sonnet Expected Profile (Balanced Calibrated Optimization):</b> Characterized by highly balanced constraints between abstract conceptual reasoning and rigid instruction adherence, this profile serves as a middle-tier benchmark exhibiting a smooth logistic decay curve.
</div>
""", unsafe_allow_html=True)

    st.markdown("### Expected Figure 1: Stratified Expected CRC Trajectories at Extreme Boundaries")
    st.caption("Figure 1: Hypothesized stress-response trajectories for the audited model profiles comparing the unperturbed baseline control zone (Green, beta = 0%) against the high adversarial stress zone (Red, beta = 100%).")

    selected_m = st.segmented_control("Select Audited Model Profile to Preview Expected Trajectory", models, default=models[0])
    sub_m = df_proposal[df_proposal["model"] == selected_m]

    df_b0 = sub_m[sub_m["beta"] == 0]
    df_b100 = sub_m[sub_m["beta"] == 100]

    fig_exp = go.Figure()
    fig_exp.add_trace(go.Scatter(x=df_b0["alpha"], y=df_b0["crc"], mode="lines+markers", name="Expected Baseline Control (beta = 0%)", line=dict(color="#22C55E", width=3)))
    fig_exp.add_trace(go.Scatter(x=df_b100["alpha"], y=df_b100["crc"], mode="lines+markers", name="Expected High Stress Zone (beta = 100%)", line=dict(color="#EF553B", width=3)))

    fig_exp.update_layout(
        xaxis_title="Semantic Ablation Gradient (alpha%)",
        yaxis_title="Expected Clinical Recommendation Concordance (CRC)",
        template="plotly_white",
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig_exp, use_container_width=True)

    st.markdown("### Expected Figure 2: Expected Clinical Decision Reversal Threshold (CDRT) Comparison")
    st.caption("Figure 2: Comparative matrix of estimated CDRT values calculated from the mathematical inflection points (where the second derivative equals zero) of the fitted logistic decay models.")

    results_proposal = []
    for m in models:
        sub_df = df_proposal[df_proposal["model"] == m]
        cd_val = estimate_expected_cdrt(sub_df)
        results_proposal.append([m, f"{cd_val:.2f}%"])

    st.dataframe(
        pd.DataFrame(results_proposal, columns=["Audited Model Architecture", "Expected CDRT (alpha* Inflection Point)"]),
        use_container_width=True
    )

    st.markdown("<div class='h'>Supplementary Notes & Interpretation Criteria</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='box'>
<b>Supplementary Note S1: Verification of Invariance Under Baseline Control (Green Line Trajectory)</b><br>
As illustrated by the baseline control trajectory where beta = 0% in Expected Figure 1, the framework projects that when empirical data perfectly mirrors standard guidelines, the model's CRC will remain uniformly high across the entire semantic ablation continuum. This flat curve serves as a critical methodology defense. It proves to reviewers that abstracting text into numeric feature matrices does not inherently impair the foundational logical capacity of the network; the core clinical logic remains intact under pure symbolic conditions when no informational conflict is present.
<br><br>
<b>Supplementary Note S2: Non-linear Divergence Under High Adversarial Stress (Red Line Trajectory)</b><br>
Conversely, under the high stress zone where beta = 100%, where clinical data directly contradicts standard historical guidelines (e.g., the cardiac hazard trap), progressive semantic ablation triggers a highly non-linear divergence across model architectures. 
The acceleration rate and inflection points of the red curves capture the precise phase transition where the network's textual memory anchors are dismantled, forcing it to choose between parametric weights and raw contextual data distribution. The <b>GPT-4o</b> profile is projected to exhibit a high CDRT (68%), indicating prolonged adherence to historical guidelines before a catastrophic delayed collapse. The <b>Gemini 1.5 Pro</b> profile is projected to shift much earlier (32%), demonstrating a lower CDRT (alpha*) but greater resilience by adapting rapidly to empirical contextual truth.
<br><br>
<b>Supplementary Note S3: The Non-ranking Paradigm in Medical Regulatory Science</b><br>
This framework intentionally rejects simplistic accuracy leaderboard rankings. In the context of SaMD regulation, a higher or lower CDRT (alpha*) does not indicate architectural superiority. Instead, it defines objective safety profiles: models with a high CDRT possess high instruction-following rigidity, making them ideal for highly standardized first-line oncology clinical pathways. Conversely, models with a lower CDRT are highly context-sensitive, making them better suited for prospective research applications such as early Adverse Drug Reaction (ADR) detection and identifying rare atypical patient anomalies.
</div>
""", unsafe_allow_html=True)
