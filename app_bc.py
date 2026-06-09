import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LLM Cognitive Elasticity Research Proposal",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.main-title { font-size: 2.3rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.2rem; }
.sub-title { font-size: 1.1rem; font-weight: 500; color: #4B5563; font-style: italic; margin-bottom: 20px; }
.section { font-size: 1.4rem; font-weight: 700; margin-top: 25px; margin-bottom: 10px; color: #111827; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.3rem; }
.warning-box { background-color: #FFFBEB; padding: 1rem; border-radius: 8px; border-left: 5px solid #D97706; margin-bottom: 15px; }
.card-report { background-color: #EFF6FF; padding: 1.25rem; border-radius: 8px; border: 1px solid #BFDBFE; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🩺 LLM Cognitive Elasticity in Precision Oncology</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Research Proposal Demonstration Framework</div>', unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
<b>Proposal Simulation Only</b><br>
All curves shown in this dashboard are hypothetical illustrations for research planning purposes 
and do not represent empirical results from GPT, Claude, Gemini, or any deployed model.
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOGISTIC MODEL
# ============================================================
def logistic_curve(x, b0, b1):
    return 1 / (1 + np.exp(-(b0 + b1 * x / 100)))

# ============================================================
# SIDEBAR
# ============================================================
page = st.sidebar.radio(
    "Navigation",
    [
        "Research Question",
        "Experimental Design",
        "Elasticity Simulation",
        "Posterior Flip Threshold",
        "Interactive Explorer"
    ]
)

# ============================================================
# PAGE 1: Research Question
# ============================================================
if page == "Research Question":
    st.markdown('<div class="section">Clinical Motivation</div>', unsafe_allow_html=True)
    st.write("""
    Current oncology LLM benchmarks focus primarily on guideline agreement.
    However, agreement alone cannot determine whether a model:
    
    1. truly learns statistical structure from observed data
    
    or
    
    2. merely retrieves memorized medical terminology.
    
    This study proposes a semantic-ablation framework to measure belief updating behavior under controlled contradiction.
    """)

    st.markdown('<div class="section">Core Question</div>', unsafe_allow_html=True)
    st.info("""
    When clinical terminology is progressively removed and empirical treatment allocation contradicts historical guidelines, 
    how rapidly does an LLM update its recommendations?
    """)

# ============================================================
# PAGE 2: Experimental Design
# ============================================================
elif page == "Experimental Design":
    st.markdown('<div class="section">3 × 5 Experimental Matrix</div>', unsafe_allow_html=True)
    matrix = pd.DataFrame({
        "Condition": ["A", "B", "C"],
        "Description": ["Original Medical Terminology", "Partial Abstraction", "Full Symbolization"]
    })
    st.dataframe(matrix, use_container_width=True, hide_index=True)

    st.markdown('<div class="section">Distortion Levels</div>', unsafe_allow_html=True)
    dist = pd.DataFrame({
        "Distortion %": [0, 30, 50, 70, 90],
        "Meaning": ["Guideline Consistent", "Mild Conflict", "Balanced Evidence", "Strong Conflict", "Near Complete Reversal"]
    })
    st.dataframe(dist, use_container_width=True, hide_index=True)

# ============================================================
# PAGE 3: Elasticity Simulation
# ============================================================
elif page == "Elasticity Simulation":
    st.markdown('<div class="section">Hypothetical Cognitive Profiles</div>', unsafe_allow_html=True)

    x = np.linspace(0, 100, 200)
    high_rigidity = logistic_curve(x, -5.5, 9.5)
    balanced = logistic_curve(x, -4.0, 7.5)
    adaptive = logistic_curve(x, -2.5, 6.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=high_rigidity, name="High-Rigidity Agent (b0=-5.5, b1=9.5)", line=dict(width=3, color="#EF4444")))
    fig.add_trace(go.Scatter(x=x, y=balanced, name="Balanced Agent (b0=-4.0, b1=7.5)", line=dict(width=3, color="#8B5CF6", dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=adaptive, name="Adaptive Agent (b0=-2.5, b1=6.0)", line=dict(width=3, color="#10B981", dash="dot")))
    
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="Equilibrium Threshold")

    fig.update_layout(
        template="plotly_white",
        height=520,
        xaxis_title="Distortion (%)",
        yaxis_title="P(Evidence-Aligned Decision)",
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section">Statistical Characterization</div>', unsafe_allow_html=True)
    st.latex(r"\log \left( \frac{P(Y=1)}{1-P(Y=1)} \right)=\beta_0+\beta_1 X")

# ============================================================
# PAGE 4: Posterior Flip Threshold
# ============================================================
elif page == "Posterior Flip Threshold":
    st.markdown('<div class="section">Posterior Flip Threshold (PFT)</div>', unsafe_allow_html=True)
    st.write("PFT defines the exact distortion gradient level where the model's posterior choice probability switches past equilibrium ($P=0.5$).")
    
    st.latex(r"PFT = -\frac{\beta_0}{\beta_1} \times 100")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        beta0 = st.slider("β0 (Prior Anchor)", -10.0, -0.5, -5.0, step=0.1)
    with col_s2:
        beta1 = st.slider("β1 (Evidence Sensitivity)", 1.0, 15.0, 8.0, step=0.1)

    pft = (-beta0 / beta1) * 100

    if pft > 100:
        st.error(f"🚨 Out of Bounds: PFT is {pft:.1f}%. The agent will never flip within the experimental range.")
    else:
        st.metric("Estimated PFT Target", f"{pft:.1f}%")

    x = np.linspace(0, 100, 200)
    y = logistic_curve(x, beta0, beta1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Fitted Agent", line=dict(color="#2563EB", width=3)))
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    if pft <= 100:
        fig.add_vline(x=pft, line=dict(color="#EF4444", dash="solid"), annotation_text=f"Flip: {pft:.1f}%")

    fig.update_layout(template="plotly_white", height=450, xaxis_title="Distortion (%)", yaxis_title="Probability")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 5: Interactive Explorer
# ============================================================
else:
    st.markdown('<div class="section">Interactive Matrix Explorer</div>', unsafe_allow_html=True)

    condition = st.selectbox("Condition", ["Condition A", "Condition B", "Condition C"])
    distortion = st.select_slider("Distortion", options=[0, 30, 50, 70, 90])

    if condition == "Condition A":
        b0, b1 = -5.5, 9.5
    elif condition == "Condition B":
        b0, b1 = -4.0, 7.5
    else:
        b0, b1 = -2.5, 6.0

    p = logistic_curve(distortion, b0, b1)

    st.metric("Evidence-Aligned Probability", f"{p*100:.1f}%")

    
    st.markdown(f"""
    <div class="card-report">
    <b>Fitted Model Diagnostics:</b><br>
    • Core Formula: $\\log(P / (1-P)) = {b0} + {b1} \\times X$<br>
    • Intercept (\\beta_0) = {b0}<br>
    • Slope (\\beta_1) = {b1}<br>
    • Active Distortion (X) = {distortion}%<br><br>
    <b>Estimated Probability Density:</b> {p:.3f}
    </div>
    """, unsafe_allow_html=True)
