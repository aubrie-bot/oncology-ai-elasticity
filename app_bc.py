import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="LLM Cognitive Elasticity Research Dashboard",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.report-title {
    font-size:2.3rem;
    font-weight:800;
    color:#1E3A8A;
}

.report-subtitle {
    font-size:1.15rem;
    color:#4B5563;
    margin-bottom:1.5rem;
}

.section-header {
    font-size:1.4rem;
    font-weight:700;
    margin-top:2rem;
    margin-bottom:1rem;
    border-bottom:2px solid #E5E7EB;
    padding-bottom:0.4rem;
}

.metric-box {
    background:#F8FAFC;
    padding:1rem;
    border-radius:10px;
    border-left:4px solid #2563EB;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="report-title">🧠 LLM Cognitive Elasticity in Precision Oncology</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="report-subtitle">Posterior Flip Threshold (PFT) Experimental Framework</div>',
    unsafe_allow_html=True
)

# ============================================================
# MODEL CONFIG
# ============================================================

PROFILES = {
    "Condition A": {
        "beta0": -5.5,
        "beta1": 9.5,
        "description": "Original Medical Terminology"
    },
    "Condition B": {
        "beta0": -4.0,
        "beta1": 7.5,
        "description": "Partial Semantic Abstraction"
    },
    "Condition C": {
        "beta0": -2.5,
        "beta1": 6.0,
        "description": "Full Symbolic Representation"
    }
}

# ============================================================
# FUNCTIONS
# ============================================================

def logistic_response(x, beta0, beta1):
    """
    X scaled between 0 and 100.
    """
    logit = beta0 + beta1 * (x / 100)
    return 1 / (1 + np.exp(-logit))


def compute_kl_divergence(p_prob, q_prob):

    eps = 1e-9

    p_prob = np.clip(p_prob, eps, 1 - eps)
    q_prob = np.clip(q_prob, eps, 1 - eps)

    p = np.array([p_prob, 1 - p_prob])
    q = np.array([q_prob, 1 - q_prob])

    return np.sum(p * np.log(p / q))


def compute_pft(beta0, beta1):
    return (-beta0 / beta1) * 100


def compute_elasticity_index(beta0, beta1):
    pft = compute_pft(beta0, beta1)

    if pft <= 0:
        return np.nan

    return 100 / pft


def compute_crs(beta0, beta1):

    x = np.linspace(0, 100, 500)

    y = logistic_response(
        x,
        beta0,
        beta1
    )

    area = np.trapz(
        1 - y,
        x
    )

    return area / 100


# Ground Truth DGP
def dgp_curve(x):

    return logistic_response(
        x,
        -3.0,
        8.0
    )

# ============================================================
# SIDEBAR
# ============================================================

menu = st.sidebar.radio(
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
# PAGE 1
# ============================================================

if menu == "Research Question":

    st.markdown(
        '<div class="section-header">Research Motivation</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Current medical LLM evaluation primarily measures final answer accuracy.
    
    However, accuracy alone cannot reveal whether an LLM:
    
    1. relies on memorized medical terminology
    2. reconstructs latent statistical structures
    
    This framework evaluates how belief states change under terminology ablation
    and contradictory evidence pressure.
    """)

    st.info("""
    Core Question:
    
    When medical terminology disappears and data contradicts historical
    guidelines, does the model trust prior knowledge or observed evidence?
    """)

# ============================================================
# PAGE 2
# ============================================================

elif menu == "Experimental Design":

    st.markdown(
        '<div class="section-header">3 × 5 Experimental Matrix</div>',
        unsafe_allow_html=True
    )

    df = pd.DataFrame({
        "Condition":[
            "A",
            "B",
            "C"
        ],
        "Language State":[
            "Original",
            "Partially Masked",
            "Fully Symbolized"
        ]
    })

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    distortion = pd.DataFrame({
        "Distortion %":[0,30,50,70,90],
        "Meaning":[
            "Guideline Consistent",
            "Mild Conflict",
            "Balanced Evidence",
            "Strong Conflict",
            "Near Reversal"
        ]
    })

    st.dataframe(
        distortion,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# PAGE 3
# ============================================================

elif menu == "Elasticity Simulation":

    st.markdown(
        '<div class="section-header">Elasticity Curves</div>',
        unsafe_allow_html=True
    )

    x = np.linspace(
        0,
        100,
        500
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=dgp_curve(x),
            name="Ground Truth DGP",
            line=dict(
                color="black",
                dash="dot"
            )
        )
    )

    colors = {
        "Condition A":"red",
        "Condition B":"purple",
        "Condition C":"green"
    }

    for name, vals in PROFILES.items():

        y = logistic_response(
            x,
            vals["beta0"],
            vals["beta1"]
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                line=dict(
                    color=colors[name],
                    width=3
                )
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=550,
        xaxis_title="Distortion Gradient (%)",
        yaxis_title="Evidence Alignment Probability"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# PAGE 4
# ============================================================

elif menu == "Posterior Flip Threshold":

    st.markdown(
        '<div class="section-header">PFT Calculator</div>',
        unsafe_allow_html=True
    )

    beta0 = st.slider(
        "β0",
        -10.0,
        -0.5,
        -5.0,
        0.1
    )

    beta1 = st.slider(
        "β1",
        1.0,
        15.0,
        8.0,
        0.1
    )

    pft = compute_pft(
        beta0,
        beta1
    )

    ei = compute_elasticity_index(
        beta0,
        beta1
    )

    crs = compute_crs(
        beta0,
        beta1
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Posterior Flip Threshold",
        f"{pft:.1f}%"
    )

    c2.metric(
        "Elasticity Index",
        f"{ei:.2f}"
    )

    c3.metric(
        "Cognitive Rigidity Score",
        f"{crs:.3f}"
    )

    x = np.linspace(
        0,
        100,
        500
    )

    y = logistic_response(
        x,
        beta0,
        beta1
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name="Belief Updating Curve"
        )
    )

    fig.add_hline(
        y=0.5,
        line_dash="dot"
    )

    fig.add_vline(
        x=pft,
        line_color="red"
    )

    fig.update_layout(
        template="plotly_white",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# PAGE 5
# ============================================================

else:

    st.markdown(
        '<div class="section-header">Interactive Explorer</div>',
        unsafe_allow_html=True
    )

    cond = st.selectbox(
        "Condition",
        list(PROFILES.keys())
    )

    distortion = st.slider(
        "Distortion %",
        0,
        100,
        50
    )

    beta0 = PROFILES[cond]["beta0"]
    beta1 = PROFILES[cond]["beta1"]

    model_prob = logistic_response(
        distortion,
        beta0,
        beta1
    )

    dgp_prob = dgp_curve(
        distortion
    )

    kl = compute_kl_divergence(
        dgp_prob,
        model_prob
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Model Probability",
        f"{model_prob:.3f}"
    )

    col2.metric(
        "Ground Truth Probability",
        f"{dgp_prob:.3f}"
    )

    col3.metric(
        "KL Divergence",
        f"{kl:.5f}"
    )

    telemetry = pd.DataFrame({
        "Metric":[
            "Condition",
            "Distortion",
            "Beta0",
            "Beta1",
            "Model Prob",
            "DGP Prob",
            "KL Divergence"
        ],
        "Value":[
            cond,
            distortion,
            beta0,
            beta1,
            round(model_prob,4),
            round(dgp_prob,4),
            round(kl,5)
        ]
    })

    st.dataframe(
        telemetry,
        use_container_width=True,
        hide_index=True
    )
