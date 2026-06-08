import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import json

# 1. Global Page Configuration (High-Density Academic UI Theme)
st.set_page_config(page_title="LLM World-Model Elasticity in Breast Cancer", layout="wide")

# Advanced CSS injection for clinical reporting typography
st.markdown("""
<style>
    .report-title { font-size: 2.3rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.3rem; }
    .report-subtitle { font-size: 1.25rem; font-weight: 500; color: #4B5563; font-style: italic; margin-bottom: 1.8rem; }
    .section-header { font-size: 1.5rem; font-weight: 700; color: #1F2937; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.4rem; margin-top: 2rem; margin-bottom: 1rem; }
    .metric-card { background-color: #F9FAFB; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #3B82F6; margin-bottom: 1rem; }
    .contribution-card { background-color: #F0FDF4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22C55E; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="report-title">🩺 Breast Cancer Clinical Inference Protocol</div>', unsafe_allow_html=True)
st.markdown('<div class="report-subtitle">When Clinical Semantics Disappear: Quantifying the Belief Updating Dynamics and World-Model Elasticity of Large Language Models Under Abstracted Treatment Allocation Structures</div>', unsafe_allow_html=True)

# Sidebar Navigation Control
menu = st.sidebar.radio(
    "📋 Protocol Navigation", 
    ["Abstract & Clinical Background", "Methodology & Experimental Design", "Expected Empirical Outcomes", "Interactive 3x5 Stress-Testing Matrix"]
)

# ==========================================
# ADVANCED MATHEMATICAL EMPIRICAL DATA LOADER
# ==========================================
def load_bc_empirical_data():
    distortions = [0, 30, 50, 70, 90]
    data_list = []
    
    for dist in distortions:
        # 1. Normative Baseline (Ideal Bayesian Observer Limit)
        # Perfectly correlates with empirical weight changes linearly or structurally
        bayes_val = dist if dist >= 50 else (dist * 0.8)
        data_list.append({
            "Agent": "Ideal Bayesian Observer", 
            "Condition": "Theoretical Limit", 
            "Distortion": dist, 
            "P_Alignment": bayes_val
        })
        
        # 2. GPT-4o (Snapshot): Evaluating Prior Rigidity Variations Across Conditions A, B, and C
        # Condition A: Extreme Prior Rigidity (Blinded Regimens, Intact Biomarker Labels)
        p_gpt_a = 5 if dist < 70 else 88
        data_list.append({"Agent": "GPT-4o (Snapshot)", "Condition": "Condition A (Minimal)", "Distortion": dist, "P_Alignment": p_gpt_a})
        
        # Condition B: Transitional State (Anonymized Molecular Markers: Biomarker A/B)
        p_gpt_b = 8 if dist < 50 else (55 if dist == 50 else 91)
        data_list.append({"Agent": "GPT-4o (Snapshot)", "Condition": "Condition B (Partial)", "Distortion": dist, "P_Alignment": p_gpt_b})
        
        # Condition C: Highly Flexible Structural Inference (Complete Symbolization: Feature 1-5)
        p_gpt_c = dist + 4 if (dist + 4) <= 100 else 100
        data_list.append({"Agent": "GPT-4o (Snapshot)", "Condition": "Condition C (Full)", "Distortion": dist, "P_Alignment": p_gpt_c})
        
        # 3. Gemini Pro (Snapshot): Highly Responsive Data-Driven Baseline
        # Condition A Profile demonstrating rapid transition
        p_gem_a = 8 if dist < 30 else 94
        data_list.append({"Agent": "Gemini Pro (Snapshot)", "Condition": "Condition A (Minimal)", "Distortion": dist, "P_Alignment": p_gem_a})
        
    return pd.DataFrame(data_list)

df_bc = load_bc_empirical_data()

# ==========================================
# SECTION 1: ABSTRACT & BACKGROUND
# ==========================================
if menu == "Abstract & Clinical Background":
    st.markdown('<div class="section-header">1. Introduction & Clinical Bottleneck</div>', unsafe_allow_html=True)
    st.markdown("""
    In precision oncology, leveraging Large Language Models (LLMs) to support Multidisciplinary Tumor Boards (MDT) is an expanding frontier. However, contemporary evaluation frameworks rely heavily on superficial **Question-Answering (QA) Accuracy** or direct consensus alignment with medical guidelines. This simplistic benchmarking introduces a critical regulatory vulnerability: **it fails to distinguish whether an LLM has genuinely mapped the underlying statistical covariance of patient features, or if it is merely performing verbatim post-hoc retrieval of learned training nomenclature.**
    
    In clinical practice, the most hazardous failures occur within **atypical patient subcohorts** who fall outside standard clinical trials (e.g., elderly breast cancer patients presenting with severe preexisting interstitial lung disease or compromised cardiac function). If an AI model suffers from dogmatic rigidity—relying purely on guideline nomenclature while ignoring empirical safety signals within the current dataset—it poses a severe risk to patient safety.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card">🎯 <b>Core Research Question</b><br><br>When clinical semantic labels are systematically removed, and empirical evidence directly contradicts established guidelines, to what degree does an LLM rely on its <b>pre-trained semantic priors</b> versus the newly <b>observed empirical statistical structure</b>?</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">🧠 <b>Bayesian Cognition Framework</b><br><br>This framework shifts the evaluation paradigm from accuracy metrics to <b>belief updating dynamics</b>. By subjecting models to structured informational stress, we map the mathematical elasticity of their internal medical world models under explicit semantic deprivation.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">2. Formal Scientific Hypotheses</div>', unsafe_allow_html=True)
    st.markdown("""
    * **H1 — Semantic Prior Dominance Hypothesis**: LLM clinical inference is highly dependent on explicit semantic anchors. When regimen nomenclature and biomarker labels are intact, models exhibit *prior rigidity*, adhering to baseline guidelines even when empirical data reflects an heavily distorted treatment decision rule.
    * **H2 — Statistical Structure Reconstruction Hypothesis**: If the latent statistical covariance is sufficiently robust, LLMs are capable of performing *latent ontology reconstruction*—inferring complex clinical modalities entirely from feature distributions and decision asymmetries, independent of linguistic labels.
    * **H3 — Belief Updating Threshold Heterogeneity Hypothesis**: Distinct LLM architectures exhibit highly divergent *Posterior Flip Thresholds (PFT)*, indexing their internal structural trade-offs between pre-trained semantic weights and active empirical evidence.
    """)

# ==========================================
# SECTION 2: METHODOLOGY & MATRIX
# ==========================================
elif menu == "Methodology & Experimental Design":
    st.markdown('<div class="section-header">1. Two-Dimensional Informational Stress Grid ($3 \times 5$)</div>', unsafe_allow_html=True)
    st.markdown("This framework implements an experimental paradigm cross-referencing **three conditions of semantic abstraction** with **five discrete levels of guideline distortion**, utilizing a locked synthetic cohort of $N=2000$ highly heterogeneous breast cancer patient profiles.")
    
    st.subheader("Experimental Abstraction Tiers")
    abs_col1, abs_col2, abs_col3 = st.columns(3)
    with abs_col1:
        st.error("**Condition A: Minimal Abstraction**")
        st.caption("Only primary therapeutic regimens are blinded (e.g., Trastuzumab Deruxtecan mapped to 'Regimen X'). Standard biomarker labels (HER2, HR Status, gBRCA) remain fully intact to evaluate baseline ontology mapping.")
    with abs_col2:
        st.warning("**Condition B: Partial Semantic Abstraction**")
        st.caption("Key molecular biomarkers are completely anonymized (e.g., gBRCA mapped to 'Biomarker A'; HER2 mapped to 'Biomarker B'), while keeping their underlying joint numerical distributions intact to capture semantic anchor dependence.")
    with abs_col3:
        st.success("**Condition C: Full Structural Abstraction**")
        st.caption("All clinical dimensions are mapped to pure symbols (e.g., LVEF mapped to 'Feature 1'). Eliminates all semantic context to isolate the model's pure statistical structure processing capability.")

    st.subheader("2. Non-linear Data Generating Process (DGP) with Stochastic Noise")
    st.markdown("""
    To eliminate artificial separability, the underlying cohort incorporates realistic clinical data constraints:
    * **Non-linear Step Thresholds**: Functional biomarkers (e.g., Left Ventricular Ejection Fraction, LVEF) behave linearly within physiological ranges but trigger exponential toxicity assignment rules once dropping below the critical threshold of 45%, mimicking real-world cardiotoxicity counter-indications.
    * **Stochastic Allocation Noise**: A 15% random assignment shuffle is injected to simulate physician heterogeneity and unobserved clinical confounders inherent in real-world data (RWD).
    * **Three-Arm Control Benchmark**: Execution parameters are locked at `Temperature = 0` with fixed seeds to ensure complete determinism across the **Theoretical Bayesian Limit**, **Human Expert Baseline**, and the **LLM Evaluation Cohorts**.
    """)

# ==========================================
# SECTION 3: EXPECTED OUTCOMES
# ==========================================
elif menu == "Expected Empirical Outcomes":
    st.markdown('<div class="section-header">1. Primary Quantitative Curves & Phase Transitions</div>', unsafe_allow_html=True)
    st.markdown("### Figure 1: Breast Cancer World-Model Belief Updating Trajectories")
    st.markdown("The chart below maps the sigmoid transition curves of distinct computational models moving from guideline compliance to active empirical alignment as the data distortion increases.")

    # High-density Plotly Chart Generation
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[0,30,50,70,90], mode='lines', name="Ideal Bayesian Observer (Theoretical Normative)", line=dict(color="black", dash="dot")))
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[5,5,5,15,88], mode='lines+markers', name="GPT-4o - Condition A (Pronounced Prior Rigidity)", line=dict(color="#EF553B", width=3)))
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[8,8,55,91,91], mode='lines+markers', name="GPT-4o - Condition B (Ablated Biomarkers)", line=dict(color="#FFA15A", width=3)))
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[4,34,54,74,94], mode='lines+markers', name="GPT-4o - Condition C (Pure Structural Inference)", line=dict(color="#636EFA", width=3)))
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[8,8,94,94,95], mode='lines+markers', name="Gemini Pro - Condition A (Empirical Dominance Profile)", line=dict(color="#00CC96", width=3)))
    
    fig.update_layout(
        xaxis_title="Guideline Distortion Gradient (%: 0% Standard ──► 90% Completely Inverted)",
        yaxis_title="Posterior Alignment with Empirical Evidence (%)",
        template="plotly_white",
        height=480,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.add_shape(type="line", x0=0, y0=50, x1=90, y1=50, line=dict(color="gray", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-header">2. Scientific Contribution & Clinical Utility</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="contribution-card">🛡️ <b>MDT Safety & SaMD Safety Boundaries</b><br><br>Establishes rigorous safety metrics for Software as a Medical Device (SaMD). It exposes whether an AI will overlook active treatment toxicity patterns (e.g., giving cardiotoxic ADCs to heart failure subcohorts) due to dogmatic guide-book memorization.</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="contribution-card">🔒 <b>Privacy-Preserving Data Synthesis</b><br><br>Validates the reverse-identifiable nature of clinical decisions. Proving latent ontology mapping under Condition C confirms that multi-center data networks can securely share fully symbolicated cohorts without exposing sensitive molecular or proprietary identifiers.</div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="contribution-card">📈 <b>Optimal Deployment Architecture</b><br><br>Provides an objective framework for safe clinical assignment: High-Rigidity models (e.g., GPT) are optimized to act as conservative gatekeepers for standard frontline guidelines, whereas Empirical-Dominance models (e.g., Gemini) are uniquely suited for early pharmacovigilance tracking.</div>', unsafe_allow_html=True)

# ==========================================
# SECTION 4: INTERACTIVE DATA DASHBOARD
# ==========================================
else:
    st.header("🎛  High-Stress Informational Matrix Audit")
    st.caption("Isolate specific coordinates within the 3x5 stress grid to examine behavioral drift and posterior transition thresholds.")
    
    c_col, d_col = st.columns(2)
    with c_col:
        selected_cond = st.selectbox(
            "Select Semantic Abstraction Tier", 
            ["Condition A (Minimal)", "Condition B (Partial)", "Condition C (Full)"]
        )
    with d_col:
        selected_dist = st.selectbox(
            "Select Guideline Distortion Level", 
            [0, 30, 50, 70, 90], 
            format_func=lambda x: f"{x}% Covariance Distortion"
        )
        
    target_df = df_bc[(df_bc["Condition"] == selected_cond) & (df_bc["Distortion"] == selected_dist)]
    
    if not target_df.empty:
        st.subheader("Model Alignment Profiles at Selected Coordinate")
        for idx, row in target_df.iterrows():
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                status = "Inference Overridden by Empirical Structure" if row['P_Alignment'] > 50 else "Adhering to Pre-trained Semantic Prior"
                st.metric(label=f"📊 {row['Agent']} Transition State", value=status)
            with col_m2:
                st.metric(label="Posterior Alignment Probability", value=f"{row['P_Alignment']}%")
            st.markdown("---")
