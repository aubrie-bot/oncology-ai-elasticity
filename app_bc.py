import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. Global Page Configuration
st.set_page_config(page_title="Multi-Agent Belief Elasticity in Oncology", layout="wide")

# Advanced CSS injection
st.markdown("""
<style>
    .report-title { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.3rem; }
    .report-subtitle { font-size: 1.2rem; font-weight: 500; color: #4B5563; font-style: italic; margin-bottom: 1.8rem; }
    .section-header { font-size: 1.4rem; font-weight: 700; color: #1F2937; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.4rem; margin-top: 2rem; margin-bottom: 1rem; }
    .metric-card { background-color: #F9FAFB; padding: 1.2rem; border-radius: 0.5rem; border-left: 4px solid #3B82F6; margin-bottom: 1rem; }
    .analysis-card { background-color: #EFF6FF; padding: 1.25rem; border-radius: 0.5rem; margin-bottom: 1rem; border: 1px solid #BFDBFE; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="report-title">🩺 Multi-Agent Breast Cancer Clinical Inference Protocol</div>', unsafe_allow_html=True)
st.markdown('<div class="report-subtitle">Quantifying Belief Updating Dynamics and Feature Hierarchy Recovery of Large Language Models Along a Semantic Ablation Continuum</div>', unsafe_allow_html=True)

menu = st.sidebar.radio(
    "📋 Protocol Navigation", 
    ["Theoretical Framework & Mathematical Fixes", "Methodology & Continuous Continuum", "Expected Empirical Outcomes (Elasticity Curves)", "Interactive Matrix Control"]
)

# ==========================================
# MATHEMATICALLY VALID DATA SIMULATOR
# ==========================================
def load_elasticity_data():
    ablation_levels = [0, 20, 40, 60, 80, 100]
    data_list = []
    
    for alpha in ablation_levels:
        # Ground Truth DGP Line (Remains stable at optimal information limit)
        data_list.append({"Agent": "DGP Ground Truth", "Ablation": alpha, "Metric_Val": 95 - (alpha * 0.05)})
        
        # GPT-4o: High Parametric Rigidity Profile (Holds on until a critical transition phase)
        p_gpt = 90 - (alpha * 0.1) if alpha < 60 else (90 - (alpha * 0.6))
        data_list.append({"Agent": "GPT-4o", "Ablation": alpha, "Metric_Val": p_gpt})
        
        # Gemini 1.5 Pro: High Context Adaptability Profile
        p_gem = 88 - (alpha * 0.3)
        data_list.append({"Agent": "Gemini 1.5 Pro", "Ablation": alpha, "Metric_Val": p_gem})
        
        # Claude 3.5 Sonnet: Calibrated Path
        p_claude = 92 - (alpha * 0.2)
        data_list.append({"Agent": "Claude 3.5 Sonnet", "Ablation": alpha, "Metric_Val": p_claude})
        
    return pd.DataFrame(data_list)

df_elasticity = load_elasticity_data()

# ==========================================
# SECTION 1: THEORETICAL FRAMEWORK & FIXES
# ==========================================
if menu == "Theoretical Framework & Mathematical Fixes":
    st.markdown('<div class="section-header">1. Formal Bayesian Inference & Parametric Weight Calibration</div>', unsafe_allow_html=True)
    st.markdown("""
    To formalize artificial clinical cognition without confounding the prior distribution with the prior weight, we define the agent's posterior log-odds between the empirically distorted decision rule ($H_E$) and the standard guideline ($H_G$). 
    Under a Uniform Prior where $P(H_E) = P(H_G) = 0.5$ (Prior Odds = 1), the inference path is parameterized by $\gamma$, representing the **Parametric Attention Anchor Coefficient**:
    """)
    
    st.latex(r"\log \frac{P(H_E \mid D)}{P(H_G \mid D)} = \gamma \cdot \underbrace{\log \frac{P(H_E)}{P(H_G)}}_{0} + \sum_{i=1}^{N} w_i \log \frac{P(D_i \mid H_E)}{P(D_i \mid H_G)}")
    
    st.markdown("""
    Where $\gamma$ does not modulate the prior probabilities directly, but indexes the architectural gating weight assigned to the model's pre-trained weights versus active in-context tokens.
    """)
    
    st.markdown('<div class="section-header">2. Rigorous Benchmarking: Defining the Reference Distribution P</div>', unsafe_allow_html=True)
    st.markdown("""
    To eliminate non-estimable normative assumptions, the reference baseline **$P$ is strictly defined as the Data-Generating Process (DGP) Ground Truth**. 
    Information-theoretic drift is measured via the empirical Kullback-Leibler Divergence against the known generative conditional probabilities of the synthetic dataset:
    """)
    
    st.latex(r"D_{\text{KL}}(P_{\text{DGP}} \parallel Q_{\text{LLM}}) = \sum_{x \in \mathcal{X}} P_{\text{DGP}}(x) \log \frac{P_{\text{DGP}}(x)}{Q_{\text{LLM}}(x)}")

# ==========================================
# SECTION 2: METHODOLOGY & CONTINUOUS CONTINUUM
# ==========================================
elif menu == "Methodology & Continuous Continuum":
    st.markdown('<div class="section-header">1. The 6-Tier Semantic Ablation Continuum</div>', unsafe_allow_html=True)
    st.markdown("""
    Instead of discrete categorical slices, this framework implements a **Semantic Ablation Continuum ($\alpha \in [0, 100]$)**. 
    Linguistic cues for critical breast cancer markers are peeled back systematically across 6 precise intervals:
    """)
    
    st.markdown("""
    * **0% Ablation**: Pristine medical nomenclature intact (`HER2+ Overexpression`, `gBRCA Mutation`, `Trastuzumab Deruxtecan`).
    * **20% Ablation**: Blinded specific product nomenclature, retained category descriptors (`HER2-targeted-ADC`, `PARP-inhibitor`).
    * **40% Ablation**: Prefixed contextual nomenclature (`Biomarker_HER2`, `Biomarker_BRCA`).
    * **60% Ablation**: Neutralized contextual nomenclature (`Biomarker_A`, `Biomarker_B`).
    * **80% Ablation**: Semi-symbolized feature strings (`Clinical_Variable_4`, `Target_Class_Alpha`).
    * **100% Ablation**: Absolute structural tokenization (`Feature_1` to `Feature_8`, `Target_Class_1`).
    """)
    
    st.markdown('<div class="section-header">2. Operationalizing Feature Hierarchy Recovery (H2)</div>', unsafe_allow_html=True)
    st.markdown("""
    To ensure complete falsifiability, **Latent Ontology Reconstruction** is replaced by a concrete metric: **Clinical Feature Hierarchy Recovery**. 
    The alignment between the True DGP feature weights and the LLM's implicit feature focus is captured using **Kendall's Tau ($\tau$) Rank Correlation** over the calculated Permutation Feature Importance ranking vectors:
    """)
    st.latex(r"\tau = \frac{C - D}{\frac{1}{2} n(n-1)}")
    st.caption("Where C represents concordant feature pairs, D represents discordant pairs, and n represents the 8 clinical dimensions.")

# ==========================================
# SECTION 3: EXPECTED EMPIRICAL OUTCOMES
# ==========================================
elif menu == "Expected Empirical Outcomes":
    st.markdown('<div class="section-header">1. Multi-Agent Belief Elasticity Curves</div>', unsafe_allow_html=True)
    st.markdown("### Figure 1: Hierarchy Recovery Rank Correlation (\u03c4) Along the Semantic Ablation Continuum")
    
    fig = go.Figure()
    # DGP Baseline
    fig.add_trace(go.Scatter(x=df_elasticity[df_elasticity["Agent"]=="DGP Ground Truth"]["Ablation"], y=df_elasticity[df_elasticity["Agent"]=="DGP Ground Truth"]["Metric_Val"]/100, mode='lines', name="DGP Ground Truth Limit", line=dict(color="black", dash="dot")))
    # Models
    fig.add_trace(go.Scatter(x=df_elasticity[df_elasticity["Agent"]=="GPT-4o"]["Ablation"], y=df_elasticity[df_elasticity["Agent"]=="GPT-4o"]["Metric_Val"]/100, mode='lines+markers', name="GPT-4o Profile", line=dict(color="#EF553B", width=3)))
    fig.add_trace(go.Scatter(x=df_elasticity[df_elasticity["Agent"]=="Gemini 1.5 Pro"]["Ablation"], y=df_elasticity[df_elasticity["Agent"]=="Gemini 1.5 Pro"]["Metric_Val"]/100, mode='lines+markers', name="Gemini 1.5 Pro Profile", line=dict(color="#00CC96", width=3)))
    fig.add_trace(go.Scatter(x=df_elasticity[df_elasticity["Agent"]=="Claude 3.5 Sonnet"]["Ablation"], y=df_elasticity[df_elasticity["Agent"]=="Claude 3.5 Sonnet"]["Metric_Val"]/100, mode='lines+markers', name="Claude 3.5 Sonnet Profile", line=dict(color="#AB63FA", width=3)))
    
    fig.update_layout(xaxis_title="Semantic Ablation Gradient (\u03b1%: 0% Full Semantics ──► 100% Absolute Symbols)", yaxis_title="Clinical Feature Hierarchy Alignment (Kendall's \u03c4)", template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-header">2. Cognitive Interpretations of Elasticity Transitions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="analysis-card"><b>🔴 GPT-4o: Critical Threshold Collapse</b><br>Maintains a high feature hierarchy recovery ($\tau \ge 0.8$) until the 60% semantic ablation boundary is crossed. This indicates that its attention mechanism is tightly coupled to localized semantic tokens; once these anchors are stripped, its internal representation network undergoes a non-linear structural collapse.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="analysis-card"><b>🟢 Gemini 1.5 Pro: Linear Structural Adaptation</b><br>Exhibits a stable, gradual linear decay along the entire continuum. Its transformer weights are optimized for continuous context processing, allowing it to preserve raw statistical feature rankings even under severe semantic deprivation.</div>', unsafe_allow_html=True)

# ==========================================
# SECTION 4: INTERACTIVE MATRIX CONTROL
# ==========================================
else:
    st.header("🎛️ Continuous Grid Point Audit")
    st.caption("Isolate single coordinates along the continuum to audit specific multi-agent metric values.")
    
    alpha_select = st.select_slider("Select Semantic Ablation Level (\u03b1%)", options=[0, 20, 40, 60, 80, 100])
    
    target_df = df_elasticity[df_elasticity["Ablation"] == alpha_select]
    if not target_df.empty:
        for idx, row in target_df.iterrows():
            st.metric(label=f"🤖 Agent: {row['Agent']}", value=f"Kendall's \u03c4 = {row['Metric_Val']/100:.3f}")
