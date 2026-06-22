# proposal_dashboard.py

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Clinical Recommendation Elasticity Study",
    layout="wide"
)

# ======================================================
# Title
# ======================================================

st.title(
    "Clinical Recommendation Elasticity of Large Language Models"
)

st.caption(
    """
Conceptual research proposal led by Hematology-Oncology investigators.

All values shown below are hypothetical examples for study design discussion only.
No real LLM outputs are displayed.
"""
)

# ======================================================
# Sidebar
# ======================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Background",
        "Study Design",
        "Endpoints",
        "Simulation"
    ]
)

# ======================================================
# Background
# ======================================================

if page == "Background":

    st.header("Clinical Motivation")

    st.markdown(
        """
Traditional evaluations ask:

        Is the answer correct?

This proposal asks:

        When evidence conflicts with prior knowledge,
        how does the recommendation change?

Clinical concern:

- atypical patients
- rare subgroups
- conflicting evidence
- guideline exceptions

These are common scenarios in oncology practice.
"""
    )

    st.header("Primary Research Question")

    st.info(
        """
When clinical semantics are progressively removed
and empirical evidence progressively contradicts
guideline expectations,

at what point does a model reverse
its treatment recommendation?
"""
    )

# ======================================================
# Study Design
# ======================================================

elif page == "Study Design":

    st.header("Three Semantic Conditions")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Condition A")
        st.write("Full clinical terminology")

        st.code("""
HER2-positive
Stage II
Anthracycline regimen
""")

    with c2:
        st.subheader("Condition B")
        st.write("Partial abstraction")

        st.code("""
Marker A+
Risk Group 2
Treatment X
""")

    with c3:
        st.subheader("Condition C")
        st.write("Full abstraction")

        st.code("""
Variable V1
Variable V2
Option T1
""")

    st.header("Statistical Reversal")

    st.markdown(
        """
Five distortion levels:

- 0%
- 30%
- 50%
- 70%
- 90%

Higher levels increasingly contradict
established guideline assumptions.
"""
    )

# ======================================================
# Endpoints
# ======================================================

elif page == "Endpoints":

    st.header("Primary Endpoint")

    st.success(
        """
Treatment Allocation Concordance (TAC)

Probability that the model recommends
the empirically dominant treatment.
"""
    )

    st.header("Secondary Endpoints")

    st.markdown(
        """
### Clinical Decision Reversal Threshold (CDRT)

Distortion level where:

TAC ≥ 50%

---

### Recommendation Elasticity Index (REI)

Measures responsiveness to changing evidence.

Higher values indicate stronger adaptation.

---

### KL Divergence

Difference between empirical allocation
and model recommendation distributions.
"""
    )

# ======================================================
# Simulation
# ======================================================

else:

    st.header("Illustrative Simulation")

    distortion = [0, 30, 50, 70, 90]

    df = pd.DataFrame({
        "Distortion": distortion,
        "GPT-4o": [5, 5, 10, 80, 92],
        "Claude": [5, 8, 55, 85, 95],
        "Gemini": [10, 65, 90, 95, 98]
    })

    fig = px.line(
        df,
        x="Distortion",
        y=["GPT-4o", "Claude", "Gemini"],
        markers=True,
        title="Illustrative Treatment Allocation Concordance"
    )

    fig.update_layout(
        yaxis_title="TAC (%)",
        xaxis_title="Distortion Level (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.header("Illustrative CDRT")

    cdrt = pd.DataFrame({
        "Model": [
            "GPT-4o",
            "Claude",
            "Gemini"
        ],
        "CDRT (%)": [
            65,
            50,
            28
        ]
    })

    st.dataframe(
        cdrt,
        use_container_width=True
    )

    st.header("Interpretation")

    st.write(
        """
Lower CDRT:

    Earlier adaptation to empirical evidence.

Higher CDRT:

    Stronger adherence to prior semantic knowledge.

These values are hypothetical and
serve only to illustrate the study framework.
"""
    )
```
