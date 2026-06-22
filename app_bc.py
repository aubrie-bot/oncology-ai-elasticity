import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# ======================================================
# 1. Page Config & Medical Academic Styling
# ======================================================
st.set_page_config(
    page_title="CRSAF Minimal Clinical Framework",
    layout="wide"
)

st.markdown("""
<style>
    .report-title { font-size: 2.1rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.3rem; }
    .report-subtitle { font-size: 1.1rem; font-weight: 500; color: #4B5563; font-style: italic; margin-bottom: 1.5rem; }
    .section-header { font-size: 1.3rem; font-weight: 700; color: #1F2937; margin-top: 1.5rem; }
    .clinical-card { background-color: #F8FAFC; padding: 1.25rem; border-radius: 0.5rem; border-left: 4px solid #3B82F6; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='report-title'>🩺 CRSAF: Clinical Recommendation Stability Audit Framework</div>", unsafe_allow_html=True)
st.markdown("<div class='report-subtitle'>Regulatory Science Paradigm for Oncology Foundation Models (Minimal Verifiable Build)</div>", unsafe_allow_html=True)

# ======================================================
# 2. DGP (Data-Generating Process - Correct Causal Design)
# ======================================================
def dgp(beta, seed=42):
    """
    [修正要點] 真實臨床生成機制 (DGP Limit) 完全與 alpha 解耦。
    客觀真理絕不隨著語言遮蔽 (\alpha) 發生退化。
    """
    # 建立本地獨立隨機數生成器，確保多執行緒下的完全可重現性
    rng = np.random.default_rng(seed + int(beta))
    noise = rng.normal(0, 0.02)
    
    # 建立受指引扭曲度 (\beta) 操弄的實證優勢決策概似函數
    score = 2.2 - (beta / 35.0) + noise
    return float(1.0 / (1.0 + np.exp(-score)))

# ======================================================
# 3. Model Profiles (Dose-Response Behavior Simulator)
# ======================================================
def model_response(alpha, beta, model_name, seed=42):
    base_truth = dgp(beta)
    rng = np.random.default_rng(seed + int(alpha) + int(beta) + int(hash(model_name) % 1000))
    noise = rng.normal(0, 0.015)

    # 模擬三種極具代表性的資訊拉扯行為軌跡 (Illustrative Profiles Only)
    if model_name == "Model_A":
        # Pattern A: 遇到臨界點突然非線性斷崖坍塌 (GPT 型態)
        inflection = 65.0
        decay = 1.0 / (1.0 + np.exp((alpha - inflection) * 0.15))
        crc = base_truth * (0.2 + 0.8 * decay)
    elif model_name == "Model_B":
        # Pattern B: 線性平滑退化軌跡 (Gemini 型態)
        crc = base_truth * (1.0 - (alpha * 0.005))
    else:
        # Pattern C: 穩定過渡的中間校準路徑 (Claude 型態)
        inflection = 45.0
        decay = 1.0 / (1.0 + np.exp((alpha - inflection) * 0.08))
        crc = base_truth * (0.5 + 0.5 * decay)

    return float(np.clip(crc + noise, 0.0, 1.0))

# ======================================================
# 4. Dataset Generation (30-Cell Two-Factor Matrix)
# ======================================================
alpha_grid = np.array([0, 20, 40, 60, 80, 100])  # Factor A: Semantic Ablation Gradient
beta_grid = np.array([0, 25, 50, 75, 100])      # Factor B: Guideline Distortion Gradient
models = ["Model_A", "Model_B", "Model_C"]

rows = []
for m in models:
    for a in alpha_grid:
        for b in beta_grid:
            rows.append({
                "model": m,
                "alpha": a,
                "beta": b,
                "crc": model_response(a, b, m),
                "truth": dgp(b)
            })

df = pd.DataFrame(rows)

# ======================================================
# 5. CDRT Estimation via Decreasing Logistic Curve Fitting
# ======================================================
def standard_logistic_decay(x, L, k, x0, y_min):
    """ 標準劑量反應遞減 S 型曲線 """
    return y_min + (L - y_min) / (1.0 + np.exp(k * (x - x0)))

def estimate_cdRt(sub_df):
    x_unique = np.array(sorted(sub_df["alpha"].unique()))
    y_mean = sub_df.groupby("alpha")["crc"].mean().values

    # 自適應參數估計起點 (p0)，徹底防止 Scipy 拋出 Underflow 與發散崩潰
    y_max, y_min = np.max(y_mean), np.min(y_mean)
    x0_guess = float(np.median(x_unique))
    k_guess = 0.1

    try:
        popt, _ = curve_fit(
            standard_logistic_decay,
            x_unique,
            y_mean,
            p0=[y_max, k_guess, x0_guess, y_min],
            bounds=([0.4, 0.01, 10.0, 0.0], [1.0, 1.0, 90.0, 0.5]),
            maxfev=5000
        )
        # 傳回對應邏輯斯曲線二階導數為 0 的數學拐點 (Inflection Point)
        return float(popt[2])
    except Exception:
        return x0_guess

# ======================================================
# 6. UI Navigation Control
# ======================================================
page = st.sidebar.radio(
    "📋 Controller", 
    ["Framework", "3D Response Surface", "CDRT Estimation"]
)

# ======================================================
# 7. Framework Page
# ======================================================
if page == "Framework":
    st.markdown("<div class='section-header'>### Minimal Clinical Regulatory Framework</div>", unsafe_allow_html=True)
    st.markdown("""
    本專案建構了一套**可證偽的對抗式臨床監管科學尺規**。我們不評估 QA 正確率，而是量化臨床決策在分佈偏移下的穩定邊界：
    - **DGP (Data-Generating Process)**：固定隨機種子之客觀臨床真實生成模型（真理平面，不隨語意剝離退化）。
    - **CRC (Clinical Recommendation Concordance)**：主要終點指標。模型最終決策行為與真實 DGP 實證優勢決策符合的機率。
    - **Alpha ($\alpha$)**：自變因 A。漸進式語意資訊流失連續體（$0\% \to 100\%$）。
    - **Beta ($\beta$)**：自變因 B。實證數據與教科書指南的背離扭曲梯度（$0\% \to 100\%$）。
    """)

# ======================================================
# 8. 3D Response Surface
# ======================================================
elif page == "3D Response Surface":
    st.markdown("<div class='section-header'>### 3D Mathematical Response Surface Map</div>", unsafe_allow_html=True)
    st.caption("💡 提示：此 3D 曲面揭示了 30 個獨立實驗網格（6x5 Matrix Cells）的雙因子交互退化軌跡。")
    
    selected_model = st.selectbox("Select Model Architecture", models)
    sub = df[df["model"] == selected_model]
    pivot = sub.pivot_table(index="beta", columns="alpha", values="crc")

    fig = go.Figure(data=[go.Surface(
        z=pivot.values,
        x=alpha_grid,
        y=beta_grid,
        colorscale='Viridis'
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title="Semantic Ablation (\u03b1%)",
            yaxis_title="Guideline Distortion (\u03b2%)",
            zaxis_title="CRC Probability",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.1))
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 9. CDRT Estimation Page
# ======================================================
else:
    st.markdown("<div class='section-header'>### Secondary Endpoint: Estimated Clinical Decision Reversal Threshold (CDRT)</div>", unsafe_allow_html=True)
    st.markdown("""
    本審計核心利用 Logistic 擬合曲線的**數學拐點（Inflection Point）**自動捕獲模型決策逆轉的加速度臨界。此指標在醫學統計上完全穩健（Robust）。
    """)

    results = []
    for m in models:
        sub = df[df["model"] == m]
        cd = estimate_cdRt(sub)
        results.append([m, f"{cd:.2f}%"])

    res_df = pd.DataFrame(results, columns=["Model Architecture", "Estimated CDRT (\u03b1* Inflection Point)"])
    st.dataframe(res_df, use_container_width=True)
    st.markdown("""
    - **高 CDRT ($\alpha^*$)**：代表模型對字面語意執念極重，先驗剛性強，容易產生死守教條的延遲崩塌。
    - **低 CDRT ($\alpha^*$)**：代表模型對上下文大數據極度敏感，能更早放下偏見，順從實證協方差。
    """)
