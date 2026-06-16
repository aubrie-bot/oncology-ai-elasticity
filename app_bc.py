import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Global Page Configuration (Breast Cancer Academic UI Theme)
# ============================================================
st.set_page_config(
    page_title="Multi-Agent World-Model Elasticity in Breast Cancer",
    layout="wide",
)

st.markdown(
    """
<style>
    .report-title { font-size: 2.3rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.3rem; }
    .report-subtitle { font-size: 1.25rem; font-weight: 500; color: #4B5563; font-style: italic; margin-bottom: 1.8rem; }
    .section-header { font-size: 1.5rem; font-weight: 700; color: #1F2937; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.4rem; margin-top: 2rem; margin-bottom: 1rem; }
    .metric-card { background-color: #F9FAFB; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #3B82F6; margin-bottom: 1rem; }
    .contribution-card { background-color: #F0FDF4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22C55E; margin-bottom: 1rem; }
    .warning-box { background-color: #FFFBEB; padding: 1.2rem; border-radius: 0.5rem; border-left: 4px solid #D97706; margin-bottom: 1rem; }
    .def-box { background-color: #F8FAFC; padding: 1rem 1.25rem; border-radius: 0.5rem; border: 1px solid #E2E8F0; margin-bottom: 1rem; font-size: 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="report-title">🩺 Multi-Agent Breast Cancer Clinical Inference Protocol</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="report-subtitle">When Clinical Semantics Disappear: Quantifying the Belief Updating Dynamics and World-Model Elasticity of Diverse Large Language Models Under Abstracted Treatment Allocation Structures</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="warning-box">
<b>RESEARCH PROPOSAL SIMULATION FRAMEWORK (MOCK PORTAL)</b><br>
All multi-agent trajectories shown below are <b>hypothesized phenomenological placeholders</b> for grant review and pilot design.
When <code>matrices_data_bc/RESULT_*.json</code> blind-test outputs exist, GPT-4o coordinates may be partially overwritten by empirical API results.
No commercial API call is executed from this dashboard at runtime.
</div>
""",
    unsafe_allow_html=True,
)

menu = st.sidebar.radio(
    "📋 Protocol Navigation",
    [
        "Abstract & Clinical Background",
        "Methodology & Formal Mathematics",
        "Expected Empirical Outcomes",
        "Interactive 3x5 Multi-Model Matrix",
    ],
)


def compute_kl_divergence(p_prob: float, q_prob: float) -> float:
    """KL divergence between binary DGP target P and model response Q."""
    p = np.clip([p_prob, 1.0 - p_prob], 1e-12, 1.0 - 1e-12)
    q = np.clip([q_prob, 1.0 - q_prob], 1e-12, 1.0 - 1e-12)
    return float(np.sum(p * np.log(p / q)))


def _ideal_bayesian_alignment(distortion: int) -> float:
    return distortion if distortion >= 50 else distortion * 0.8


def load_bc_hypothesized_data() -> pd.DataFrame:
    distortions = [0, 30, 50, 70, 90]
    data_list = []

    for dist in distortions:
        data_list.append({
            "Agent": "Ideal Bayesian Observer",
            "Condition": "Theoretical Limit",
            "Distortion": dist,
            "P_Alignment": _ideal_bayesian_alignment(dist),
        })
        data_list.append({
            "Agent": "GPT-4o",
            "Condition": "Condition A (Minimal)",
            "Distortion": dist,
            "P_Alignment": 5 if dist < 70 else 88,
        })
        data_list.append({
            "Agent": "GPT-4o",
            "Condition": "Condition B (Partial)",
            "Distortion": dist,
            "P_Alignment": 8 if dist < 50 else (55 if dist == 50 else 91),
        })
        data_list.append({
            "Agent": "GPT-4o",
            "Condition": "Condition C (Full)",
            "Distortion": dist,
            "P_Alignment": min(dist + 4, 100),
        })
        data_list.append({
            "Agent": "Gemini 1.5 Pro",
            "Condition": "Condition A (Minimal)",
            "Distortion": dist,
            "P_Alignment": 8 if dist < 30 else 94,
        })
        data_list.append({
            "Agent": "Gemini 1.5 Pro",
            "Condition": "Condition B (Partial)",
            "Distortion": dist,
            "P_Alignment": 12 if dist < 30 else 96,
        })
        data_list.append({
            "Agent": "Gemini 1.5 Pro",
            "Condition": "Condition C (Full)",
            "Distortion": dist,
            "P_Alignment": min(dist + 5, 100),
        })
        data_list.append({
            "Agent": "Claude 3.5 Sonnet",
            "Condition": "Condition A (Minimal)",
            "Distortion": dist,
            "P_Alignment": 6 if dist < 50 else (65 if dist == 50 else 90),
        })
        data_list.append({
            "Agent": "Claude 3.5 Sonnet",
            "Condition": "Condition B (Partial)",
            "Distortion": dist,
            "P_Alignment": 10 if dist < 50 else (75 if dist == 50 else 93),
        })
        data_list.append({
            "Agent": "Claude 3.5 Sonnet",
            "Condition": "Condition C (Full)",
            "Distortion": dist,
            "P_Alignment": min(dist + 2, 100),
        })

    return pd.DataFrame(data_list)


def _confidence_to_alignment(confidence: int, inferred_class: str) -> float:
    label = (inferred_class or "").lower()
    if "supportive" in label:
        return float(100 - confidence)
    return float(confidence)


def load_bc_empirical_data() -> tuple:
    df = load_bc_hypothesized_data()
    cond_map = {
        "CondA": "Condition A (Minimal)",
        "CondB": "Condition B (Partial)",
        "CondC": "Condition C (Full)",
    }
    api_hits = 0

    for cond_code, cond_label in cond_map.items():
        for dist in [0, 30, 50, 70, 90]:
            json_path = f"matrices_data_bc/RESULT_{cond_code}_Distort_{dist}.json"
            if not os.path.exists(json_path):
                continue
            with open(json_path, encoding="utf-8") as f:
                res = json.load(f)
            p_align = _confidence_to_alignment(
                res.get("confidence_score", 50),
                res.get("inferred_treatment_class", ""),
            )
            mask = (
                (df["Agent"] == "GPT-4o")
                & (df["Condition"] == cond_label)
                & (df["Distortion"] == dist)
            )
            df.loc[mask, "P_Alignment"] = p_align
            api_hits += 1

    if api_hits == 0:
        return df, "hypothesized (no RESULT_*.json detected)"
    if api_hits == 15:
        return df, "empirical GPT-4o overlay (15/15 JSON files)"
    return df, f"partial empirical GPT-4o overlay ({api_hits}/15 JSON files)"


def add_belief_curve(fig, df, agent, condition, label, color, width=3, dash=None):
    sub = df[(df["Agent"] == agent) & (df["Condition"] == condition)].sort_values("Distortion")
    if sub.empty:
        return
    line = dict(color=color, width=width)
    if dash:
        line["dash"] = dash
    fig.add_trace(go.Scatter(
        x=sub["Distortion"], y=sub["P_Alignment"],
        mode="lines+markers", name=label, line=line,
    ))


df_bc, data_mode = load_bc_empirical_data()
st.sidebar.caption(f"Data mode: {data_mode}")

if menu == "Abstract & Clinical Background":
    st.markdown('<div class="section-header">1. Introduction & Clinical Bottleneck</div>', unsafe_allow_html=True)
    st.markdown("""
In precision oncology, leveraging Large Language Models (LLMs) to support Multidisciplinary Tumor Boards (MDT) is an expanding frontier. However, contemporary evaluation frameworks rely heavily on superficial **Question-Answering (QA) Accuracy** or direct consensus alignment with medical guidelines. This introduces a critical regulatory vulnerability: **it fails to distinguish whether an LLM has genuinely mapped the underlying statistical covariance of patient features, or if it is merely performing verbatim post-hoc retrieval of learned training nomenclature.**

In clinical practice, the most hazardous failures occur within **atypical patient subcohorts** (e.g., elderly breast cancer patients with severe cardiac dysfunction). If an AI model suffers from dogmatic rigidity, it poses a severe risk to patient safety.
""")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card">🎯 <b>Core Research Question</b><br><br>When clinical semantic labels are systematically removed, and empirical evidence directly contradicts established guidelines, to what degree does an LLM rely on its <b>pre-trained semantic priors</b> versus the newly <b>observed empirical statistical structure</b>?</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">🧠 <b>Bayesian Cognition Framework</b><br><br>This framework shifts the evaluation paradigm from accuracy metrics to <b>belief updating dynamics</b> under explicit semantic deprivation.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">2. Formal Scientific Hypotheses</div>', unsafe_allow_html=True)
    st.markdown("""
* **H1 — Semantic Prior Dominance**: Models exhibit *prior rigidity* when labels are intact.
* **H2 — Statistical Structure Reconstruction**: Robust covariance enables *latent ontology reconstruction* under Condition C.
* **H3 — Belief Updating Threshold Heterogeneity**: Architectures diverge in *Posterior Flip Threshold (PFT)*.
""")

elif menu == "Methodology & Formal Mathematics":
    st.markdown('<div class="section-header">1. Formal Mathematical Modeling of Belief Updating</div>', unsafe_allow_html=True)
    st.markdown("""
Let $H_G$ denote the guideline hypothesis and $H_E$ the empirically distorted alternative.
The posterior odds use a rigidity coefficient $\\gamma$ (conceptual until API calibration):
""")
    st.latex(r"\log \frac{P(H_E \mid D)}{P(H_G \mid D)} = \gamma \cdot \log \frac{P(H_E)}{P(H_G)} + \sum_{i=1}^{N} \log \frac{P(D_i \mid H_E)}{P(D_i \mid H_G)}")

    st.markdown('<div class="section-header">2. Operational Definition of P_Alignment</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="def-box">
<b>P_Alignment (%)</b> = estimated probability of aligning with the <b>empirically dominant allocation pattern</b> at the current distortion level.<br><br>
At <b>0% distortion</b>, DGP matches guidelines; P_Alignment stays near <b>0%</b> on this strain axis.<br>
At <b>90% distortion</b>, a fully evidence-responsive agent approaches <b>90%</b>. The 50% line marks the PFT equilibrium.
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">3. Three Abstraction Conditions</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Condition A")
        st.latex(r"\gamma_{A} \gg 1")
        st.caption("Intact nomenclature; maximal prior rigidity.")
    with c2:
        st.markdown("### Condition B")
        st.latex(r"\gamma_{B} \to 1")
        st.caption("Anonymized biomarkers; semi-balanced updating.")
    with c3:
        st.markdown("### Condition C")
        st.latex(r"\gamma_{C} \to 0")
        st.caption("Full symbolization; covariance-only inference.")

    st.markdown('<div class="section-header">4. KL Divergence & Human Baseline</div>', unsafe_allow_html=True)
    st.latex(r"D_{\text{KL}}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}")
    st.markdown("""
* **Human Expert Baseline (planned)**: Target $D_{\\text{KL}} \\le 0.15$ under mild noise; hard LVEF $\\le 45\\%$ exclusions.
* **Noise**: 15% prescription shuffle in `generate_datasets_bc.py`.
""")

elif menu == "Expected Empirical Outcomes":
    st.markdown('<div class="section-header">1. Primary Quantitative Curves</div>', unsafe_allow_html=True)
    st.markdown("### Figure 1: Multi-Agent Belief Updating Trajectories")
    st.caption(f"Single source of truth from active data matrix. Mode: **{data_mode}**.")

    fig = go.Figure()
    add_belief_curve(fig, df_bc, "Ideal Bayesian Observer", "Theoretical Limit", "Ideal Bayesian Observer", "black", width=2, dash="dot")
    add_belief_curve(fig, df_bc, "GPT-4o", "Condition A (Minimal)", "GPT-4o — Condition A", "#EF553B")
    add_belief_curve(fig, df_bc, "Gemini 1.5 Pro", "Condition A (Minimal)", "Gemini 1.5 Pro — Condition A", "#00CC96")
    add_belief_curve(fig, df_bc, "Claude 3.5 Sonnet", "Condition A (Minimal)", "Claude 3.5 Sonnet — Condition A", "#AB63FA")
    add_belief_curve(fig, df_bc, "GPT-4o", "Condition B (Partial)", "GPT-4o — Condition B", "#FFA15A", width=2, dash="dash")
    add_belief_curve(fig, df_bc, "GPT-4o", "Condition C (Full)", "GPT-4o — Condition C", "#636EFA", width=2, dash="dash")
    fig.update_layout(
        xaxis_title="Guideline Distortion Gradient (%)",
        yaxis_title="P_Alignment (%)",
        template="plotly_white", height=520,
        yaxis=dict(range=[-5, 105]),
    )
    fig.add_shape(type="line", x0=0, y0=50, x1=90, y1=50, line=dict(color="gray", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">2. Phase Transition Narratives</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        st.subheader("🔴 GPT-4o")
        st.info("**Cond A:** Flat near 5% until ~50%, then phase transition by 70% (88% in matrix). Intact labels anchor priors.\n\n**Cond B/C:** Inflection shifts toward 50% as tokens are ablated.")
    with a2:
        st.subheader("🟢 Gemini / 🟣 Claude")
        st.warning("**Gemini:** Early flip near 30% — high in-context sensitivity.")
        st.success("**Claude:** Balanced step near 50% entropy.")

    st.markdown('<div class="section-header">3. Clinical Utility</div>', unsafe_allow_html=True)
    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown('<div class="contribution-card">🛡️ <b>SaMD Safety Boundaries</b><br><br>Detect dogmatic override of toxicity signals.</div>', unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="contribution-card">🔒 <b>Privacy-Preserving RWD</b><br><br>Latent ontology under Condition C.</div>', unsafe_allow_html=True)
    with cc:
        st.markdown('<div class="contribution-card">📈 <b>Deployment Architecture</b><br><br>Rigid vs. empirical model roles.</div>', unsafe_allow_html=True)

else:
    st.header("🎛️ Interactive 3×5 Matrix Audit")
    st.caption("Same matrix as Figure 1, with real-time D_KL(P‖Q).")

    c_col, d_col = st.columns(2)
    with c_col:
        selected_cond = st.selectbox("Semantic Abstraction Tier", ["Condition A (Minimal)", "Condition B (Partial)", "Condition C (Full)"])
    with d_col:
        selected_dist = st.selectbox("Distortion Level", [0, 30, 50, 70, 90], format_func=lambda x: f"{x}%")

    target_df = df_bc[(df_bc["Condition"] == selected_cond) & (df_bc["Distortion"] == selected_dist)]
    target_dgp_prob = selected_dist / 100.0

    if target_df.empty:
        st.warning("No data at this coordinate.")
    else:
        st.metric("DGP Target P(DGP)", f"{target_dgp_prob * 100:.0f}%")
        for _, row in target_df.iterrows():
            p_align = row["P_Alignment"]
            kl = compute_kl_divergence(target_dgp_prob, p_align / 100.0)
            m1, m2, m3 = st.columns(3)
            with m1:
                status = "Empirical Override" if p_align > 50 else "Prior Adherence"
                st.metric(f"📊 {row['Agent']}", status)
            with m2:
                st.metric("P_Alignment (%)", f"{p_align:.0f}%")
            with m3:
                st.metric("D_KL(P‖Q)", f"{kl:.4f} nats" if np.isfinite(kl) else "Indeterminate")
            st.markdown("---")
