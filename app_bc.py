import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================
# 1. Global Page Configuration (High-Density Academic UI Theme)
# ============================================================
st.set_page_config(
    page_title="LLM Cognitive Elasticity Research Proposal", 
    layout="wide"
)

# Advanced CSS injection for clinical reporting typography
st.markdown("""
<style>
    .report-title { font-size: 2.3rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.3rem; }
    .report-subtitle { font-size: 1.25rem; font-weight: 500; color: #4B5563; font-style: italic; margin-bottom: 1.8rem; }
    .section-header { font-size: 1.5rem; font-weight: 700; color: #1F2937; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.4rem; margin-top: 2rem; margin-bottom: 1rem; }
    .metric-card { background-color: #F9FAFB; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #3B82F6; margin-bottom: 1rem; }
    .contribution-card { background-color: #F0FDF4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22C55E; margin-bottom: 1rem; }
    .analysis-card { background-color: #EFF6FF; padding: 1.25rem; border-radius: 0.5rem; margin-bottom: 1rem; border: 1px solid #BFDBFE; }
    .math-block { background-color: #F8FAFC; padding: 1.25rem; border-radius: 0.5rem; border: 1px solid #E2E8F0; font-family: monospace; margin-bottom: 1rem; }
    .warning-box { background-color: #FFFBEB; padding: 1.2rem; border-radius: 0.5rem; border-left: 4px solid #D97706; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="report-title">🩺 LLM Cognitive Elasticity in Precision Oncology</div>', unsafe_allow_html=True)
st.markdown('<div class="report-subtitle">A Standardized Binary Logistic Regression Prototyping Engine for Quantifying Multi-Agent Belief Updating Under Structured Terminology Ablation</div>', unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
<b>RESEARCH PROPOSAL SIMULATION FRAMEWORK (MOCK PORTAL)</b><br>
This platform serves as a Conceptual Proof-of-Concept for grant review and pilot methodology tracking. 
All behavioral curves, responses, and parameters mapped herein denote Hypothesized Phenotypic Trajectories. 
No real commercial API execution calls have been instantiated at this validation tier.
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation Control
menu = st.sidebar.radio(
    "Navigation", 
    [
        "Research Question", 
        "Experimental Design", 
        "Elasticity Simulation", 
        "Posterior Flip Threshold (PFT)", 
        "Interactive Explorer & KL Matrix"
    ]
)

# ==========================================
# SCIENTIFIC COMPUTATION ENGINES
# ==========================================
def logistic_response(x, beta0, beta1):
    """Standard Binary Logistic Link Function tracking Evidence Alignment Odds"""
    logit = beta0 + beta1 * (x / 100)
    return 1 / (1 + np.exp(-logit))

def compute_kl_divergence(p_prob, q_prob):
    """Computes empirical Kullback-Leibler Divergence between two binary distributions"""
    p = np.clip([p_prob, 1.0 - p_prob], 1e-12, 1.0 - 1e-12)
    q = np.clip([q_prob, 1.0 - q_prob], 1e-12, 1.0 - 1e-12)
    return np.sum(p * np.log(p / q))

# ==========================================
# PAGE 1: RESEARCH QUESTION (徹底消滅任何潛在亂碼)
# ==========================================
if menu == "Research Question":
    st.markdown('<div class="section-header">Clinical Motivation & Regulatory Bottleneck</div>', unsafe_allow_html=True)
    st.write("""
    Contemporary Large Language Model (LLM) benchmarking frameworks in digital medicine focus extensively on standard diagnostic agreement scores, 
    verbatim Question-Answering (QA) accuracy, or direct replication of consensus guidelines (e.g., NCCN guidelines). 
    However, static accuracy benchmarking suffers from a critical epistemological limitation: it cannot determine whether a model has genuinely mapped the underlying statistical covariance of complex patient parameters, or if it is merely performing shallow pattern retrieval of memorized medical terminology.
    
    If a deployed model exhibits rigid semantic anchoring, it presents severe risks within highly heterogeneous clinical subsets. For instance, when treating an atypical breast cancer cohort where standard guideline pathways are contradicted by localized patient toxicity variables, a rigid model may continuously override active statistical evidence in favor of memorized nomenclature—jeopardizing downstream safety constraints within a Multidisciplinary Tumor Board (MDT).
    """)
    
    st.markdown('<div class="section-header">Core Informational Stress Hypotheses</div>', unsafe_allow_html=True)
    
    # 移除所有 HTML 標記，改用純淨的 Markdown 格式，徹底杜絕網頁渲染編碼衝突
    st.info(
        "**Central Research Objective:** When explicit clinical domain labels are systematically ablated "
        "and the immediate dataset distribution directly contradicts historical textbook guidelines, "
        "does a transformer-based agent rely on its **Pre-trained Semantic Prior Knowledge** "
        "or its capacity to reconstruct **Latent Statistical Data Structures**?"
    )

# ==========================================
# PAGE 2: EXPERIMENTAL DESIGN
# ==========================================
elif menu == "Experimental Design":
    st.markdown('<div class="section-header">1. Standardized 3 × 5 Stress Testing Matrix</div>', unsafe_allow_html=True)
    st.write("The proposed framework maps cognitive degradation trajectories by cross-referencing three discrete states of terminology deprivation against five gradients of data distribution distortion utilizing a locked cohort of N=2000 synthetic patient vectors.")
    
    matrix_df = pd.DataFrame({
        "Condition Layer": ["Condition A", "Condition B", "Condition C"],
        "Operational Extraction State": ["Original Medical Terminology", "Partial Semantic Abstraction", "Full Structural Symbolization"],
        "Linguistic Execution Rules": [
            "All canonical diagnostic text strings, molecular classifications, and target intervention keywords are fully retained (e.g., 'HER2 Overexpression', 'Left Ventricular Ejection Fraction (LVEF) %', 'Trastuzumab Deruxtecan').",
            "Primary domain descriptors are masked or swapped with generic alphanumeric string placeholders (e.g., 'Biomarker B', 'Functional Parameter 2', 'Therapeutic Agent X') to intentionally break zero-shot textbook matching loops.",
            "Complete erasure of clinical syntax. All dimensional boundaries are mapped to pure mathematical notation (e.g., 'Feature 1', 'Feature 2', 'Target Class Alpha'), reducing the environment to an unanchored pattern discovery space."
        ]
    })
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">2. Contradiction Distortion Gradients</div>', unsafe_allow_html=True)
    dist_df = pd.DataFrame({
        "Distortion Gradient (X%)": [0, 30, 50, 70, 90],
        "Cohort Covariance Trajectory": [
            "Guideline Consistent: Clinical assignment strictly maps historical distribution logic.",
            "Mild Conflict: Stochastic decision noise introduced; initial signal decoupling.",
            "Balanced Evidence: Maximum statistical entropy; historical pathways provide no information.",
            "Strong Conflict: Inverse operational mapping; active empirical signals override guidelines.",
            "Near Complete Reversal: Structural inversion of classic precision oncology assignment logic."
        ]
    })
    st.dataframe(dist_df, use_container_width=True, hide_index=True)

# ==========================================
# PAGE 3: ELASTICITY SIMULATION
# ==========================================
elif menu == "Elasticity Simulation":
    st.markdown('<div class="section-header">Hypothesized Phenotypic Trajectories Across Cognitive Profiles</div>', unsafe_allow_html=True)
    st.write("To establish formal falsifiability parameters prior to actual API blind testing, we model three distinct hypothetical cognitive archetypes navigating data distortion pressure:")

    x_line = np.linspace(0, 100, 200)
    
    y_rigid = logistic_response(x_line, -5.5, 9.5)
    y_balanced = logistic_response(x_line, -4.0, 7.5)
    y_adaptive = logistic_response(x_line, -2.5, 6.0)

    fig = go.Figure()
    
    # DGP Ground Truth Identity Map (Y=X)
    fig.add_trace(go.Scatter(x=x_line, y=x_line / 100, mode="lines", name="DGP Ground Truth (Optimal Linear Observer)", line=dict(color="black", dash="dot", width=2)))
    
    fig.add_trace(go.Scatter(x=x_line, y=y_rigid, mode="lines", name="High-Rigidity Profile (Hypothesized Prior Retention)", line=dict(color="#EF4444", width=3)))
    fig.add_trace(go.Scatter(x=x_line, y=y_balanced, mode="lines", name="Balanced Profile (Hypothesized Syncretic Updating)", line=dict(color="#8B5CF6", width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(x=x_line, y=y_adaptive, mode="lines", name="Adaptive Profile (Hypothesized Data Responsiveness)", line=dict(color="#10B981", width=2.5, dash="dot")))
    
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="Equilibrium Odds Point P(Y=1) = 0.5")

    fig.update_layout(
        template="plotly_white",
        height=540,
        xaxis_title="Statistical Distortion Pressure Gradient (X%)",
        yaxis_title="Probability of Evidence-Aligned Choice P(Y=1)",
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Mathematical Optimization Axiom</div>', unsafe_allow_html=True)
    st.write("The underlying probability distribution mapping the choice matrix is analyzed via the standard empirical logistic regression function:")
    st.latex(r"\log \left( \frac{P(Y=1 \mid X)}{1 - P(Y=1 \mid X)} \right) = \beta_0 + \beta_1 X")

    col_hyp1, col_hyp2 = st.columns(2)
    with col_hyp1:
        st.markdown("""
        <div class="analysis-card" style="border-left-color: #EF4444;">
        <b>Expected Phenotype — High-Rigidity Profile:</b><br>
        If an agent exhibits elevated parametric anchoring (beta_0 << 0), it is hypothesized to remain non-responsive to empirical data asymmetries under low-to-moderate stress conditions (X < 50%). A phase transition into evidence alignment is expected to materialize only under extreme distortion pressure (X >= 70%).
        </div>
        """, unsafe_allow_html=True)
    with col_hyp2:
        st.markdown("""
        <div class="analysis-card" style="border-left-color: #10B981;">
        <b>Expected Phenotype — Adaptive Profile:</b><br>
        If an agent's internal architecture prioritizes in-context distributions over weights memorization (beta_0 -> 0), the model may exhibit an accelerated, linear shift along the gradient, migrating toward empirical evidence alignment significantly earlier in the timeline.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 4: POSTERIOR FLIP THRESHOLD (PFT)
# ==========================================
elif menu == "Posterior Flip Threshold (PFT)":
    st.markdown('<div class="section-header">1. Operationalization of the PFT Primary Metric (X*)</div>', unsafe_allow_html=True)
    st.write("""
    A primary contribution of this methodology is the formulation of the **Posterior Flip Threshold (PFT)**. 
    PFT isolates the exact mathematical coordinate along the data distortion axis where the model's internal belief state crosses equilibrium ($P(Y=1) = 0.5$). 
    Solving for the logit activation equal to zero yields the exact critical boundary metric:
    """)
    
    st.latex(r"X^* = -\frac{\beta_0}{\beta_1} \times 100")

    col_pft1, col_pft2 = st.columns(2)
    with col_pft1:
        input_b0 = st.slider("Beta_0 (Intercept Parameter : Parametric Bias)", -10.0, -0.5, -5.0, step=0.1)
    with col_pft2:
        input_b1 = st.slider("Beta_1 (Slope Parameter : Sensitivity Velocity)", 1.0, 15.0, 8.0, step=0.1)

    calculated_pft = (-input_b0 / input_b1) * 100

    if calculated_pft > 100:
        st.error(f"🚨 Configuration Deficit: Estimated PFT is out of functional experimental boundaries ({calculated_pft:.1f}%). The simulated profile will remain completely non-responsive to evidence transitions across the entire continuum.")
    else:
        st.metric(label="Estimated Posterior Flip Threshold (PFT Metric)", value=f"{calculated_pft:.1f}%")

    x_range = np.linspace(0, 100, 200)
    y_dynamic = logistic_response(x_range, input_b0, input_b1)

    fig_pft = go.Figure()
    fig_pft.add_trace(go.Scatter(x=x_range, y=y_dynamic, mode="lines", name="Simulated Cognitive Path", line=dict(color="#2563EB", width=3)))
    fig_pft.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="Equilibrium (P=0.5)")
    if calculated_pft <= 100:
        fig_pft.add_vline(x=calculated_pft, line=dict(color="#EF4444", width=2), annotation_text=f"Inflection Coordinate: {calculated_pft:.1f}%")

    fig_pft.update_layout(template="plotly_white", height=400, xaxis_title="Distortion Gradient (%)", yaxis_title="Probability Distribution")
    st.plotly_chart(fig_pft, use_container_width=True)

    st.markdown('<div class="section-header">2. Hypothetical Pilot Prototyping Reference Matrix</div>', unsafe_allow_html=True)
    st.write("This standardized mapping table represents the core comparative tracking baseline designed for the final empirical results chapter:")
    
    tracking_table = pd.DataFrame({
        "Experimental Profile Group": ["High-Rigidity Archetype (Hypothesized)", "Balanced Archetype (Hypothesized)", "Adaptive Archetype (Hypothesized)"],
        "Target Condition Map": ["Condition A (Intact Language)", "Condition B (Partial Anonymization)", "Condition C (Pure Mathematical Symbols)"],
        "Simulated Intercept (Beta_0)": [-5.5, -4.0, -2.5],
        "Simulated Slope (Beta_1)": [9.5, 7.5, 6.0],
        "Calculated PFT Boundary (X*)": ["57.9%", "53.3%", "41.7%"]
    })
    st.table(tracking_table)

# ==========================================
# PAGE 5: INTERACTIVE EXPLORER & LIVE KL MATRIX
# ==========================================
else:
    st.markdown('<div class="section-header">Dynamic Matrix Point Auditor & Information Entropy Evaluator</div>', unsafe_allow_html=True)
    st.write("Select specific intersection coordinates across the 3x5 experimental grid. The dashboard will instantly run runtime numerical executions for choice probabilities and real-time Kullback-Leibler Divergence tracking.")

    sel_col, dist_col = st.columns(2)
    with sel_col:
        ui_cond = st.selectbox("Target Condition Tier", ["Condition A", "Condition B", "Condition C"])
    with dist_col:
        ui_dist = st.select_slider("Target Distortion Pressure Input", options=[0, 30, 50, 70, 90])

    if ui_cond == "Condition A":
        sim_b0, sim_b1 = -5.5, 9.5
    elif ui_cond == "Condition B":
        sim_b0, sim_b1 = -4.0, 7.5
    else:
        sim_b0, sim_b1 = -2.5, 6.0

    computed_p_alignment = logistic_response(ui_dist, sim_b0, sim_b1)
    target_dgp_prob = ui_dist / 100.0
    computed_kl = compute_kl_divergence(target_dgp_prob, computed_p_alignment)

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Evidence-Aligned Response Probability", f"{computed_p_alignment*100:.1f}%")
    with m_col2:
        if np.isnan(computed_kl) or np.isinf(computed_kl):
            st.metric("Real-Time KL-Divergence D_KL(P||Q)", "Indeterminate (Zero Variance State)")
        else:
            st.metric("Real-Time KL-Divergence D_KL(P||Q)", f"{computed_kl:.4f} nats")

    st.markdown("""
    <style>
    .math-card { background-color: #F8FAFC; padding: 1.25rem; border-radius: 8px; border: 1px solid #E2E8F0; margin-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="math-card">
    <b>Active Verification Telemetry Log:</b><br><br>
    • Context State Tracked: <code>{ui_cond}</code><br>
    • Data Inversion Strain (X): <code>{ui_dist}%</code><br>
    • Fitted Intercept Parameter (\\beta_0): <code>{sim_b0}</code><br>
    • Fitted Elasticity Slope (\\beta_1): <code>{sim_b1}</code><br><br>
    • <b>Objective Target Probability Density P(DGP):</b> <code>{target_dgp_prob:.3f}</code><br>
    • <b>Model Probability Response Space Q(LLM):</b> <code>{computed_p_alignment:.3f}</code><br>
    • <b>Run-Time Extrapolated Information Deficit:</b> <code>{computed_kl:.5f} nats of unexpected structural entropy</code>
    </div>
    """, unsafe_allow_html=True)
