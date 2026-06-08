import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import json

# 1. 網頁全域風格設定（配置嚴謹的醫療學術 UI）
st.set_page_config(page_title="LLM World-Model Elasticity in Oncology", layout="wide")

# 自定義 CSS 樣式優化排版
st.markdown("""
<style>
    .report-title { font-size: 2.4rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.5rem; }
    .report-subtitle { font-size: 1.4rem; font-weight: 500; color: #4B5563; font-style: italic; margin-bottom: 2rem; }
    .section-header { font-size: 1.6rem; font-weight: 700; color: #1F2937; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.5rem; margin-top: 2rem; margin-bottom: 1rem; }
    .metric-card { background-color: #F9FAFB; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #3B82F6; margin-bottom: 1rem; }
    .contribution-card { background-color: #F0FDF4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22C55E; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="report-title">🩺 癌症決策行為之 AI 世界模型彈性量化平台</div>', unsafe_allow_html=True)
st.markdown('<div class="report-subtitle">When Clinical Semantics Disappear: Belief Updating Dynamics of LLMs Under Abstracted Treatment Allocation Structures</div>', unsafe_allow_html=True)
st.caption("主導者：胸腔腫瘤科醫師團隊 | 共同執行者：公衛系生物統計研究助理")

# 側邊欄導覽
menu = st.sidebar.radio("📋 研發協議導覽", ["詳細研究提案與背景", "方法學與核心實驗矩陣", "預期科學結果與圖表", "互動式壓力測試儀表板"])

# ==========================================
# 核心數據加載器 (符合科學假說之數據架構)
# ==========================================
def load_advanced_results():
    distortions = [0, 30, 50, 70, 90]
    conditions = ["Condition A (Minimal)", "Condition B (Partial)", "Condition C (Full)"]
    data_list = []
    
    # 模擬包含理論貝氏觀察者與三大模型的 PFT 軌跡
    for dist in distortions:
        # 完美貝氏隨數據線性更新
        data_list.append({"Agent": "Ideal Bayesian Observer", "Condition": "Theoretical Limit", "Distortion": dist, "P_Alignment": dist if dist >= 50 else (dist * 0.8)})
        
        # GPT-4o: 強烈教條型 (Condition A 下極度僵固)
        p_gpt = 5 if dist < 70 else 85
        data_list.append({"Agent": "GPT-4o (Snapshot)", "Condition": "Condition A (Minimal)", "Distortion": dist, "P_Alignment": p_gpt})
        data_list.append({"Agent": "GPT-4o (Snapshot)", "Condition": "Condition C (Full)", "Distortion": dist, "P_Alignment": dist + 5})
        
        # Gemini Pro: 數據驅動型 (翻轉極快)
        p_gem = 10 if dist < 30 else 92
        data_list.append({"Agent": "Gemini Pro (Snapshot)", "Condition": "Condition A (Minimal)", "Distortion": dist, "P_Alignment": p_gem})
        data_list.append({"Agent": "Gemini Pro (Snapshot)", "Condition": "Condition C (Full)", "Distortion": dist, "P_Alignment": dist + 8})
        
    return pd.DataFrame(data_list)

df_adv = load_advanced_results()

# ==========================================
# 軌道一：研究背景、痛點與核心問題
# ==========================================
if menu == "詳細研究提案與背景":
    st.markdown('<div class="section-header">一、 研究背景與臨床痛點（Background & Clinical Bottleneck）</div>', unsafe_allow_html=True)
    st.markdown("""
    當前精準腫瘤學（Precision Oncology）正加速引入前沿大語言模型（LLMs）輔助多學科聯合診治（MDT/Tumor Board）。然而，目前的醫學界存在一種**盲目的「黑盒樂觀」**，僅透過常規的問答正確率（QA Accuracy）或與指南的符合率來評估 AI。這種粗暴的評測方式存在系統性漏洞：**它無法區分 AI 是真正理解了臨床決策背後的因果統計邏輯，還是僅僅死記硬背了預訓練文本中的臨床指南字面。**
    
    在臨床實務中，真正的危機往往發生在**「臨床試驗外的非標準患者（少數派亞組）」**身上（例如：伴隨嚴重間質性肺病的高齡癌症患者）。如果 AI 只是教條式地背誦指南，忽視數據眼前的真實毒性訊號，將會引發嚴重的醫療安全事故。
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card">🎯 <b>核心研究問題（Core Research Question）</b><br><br>當臨床醫療語意被逐步移除、且實證證據與常識指南發生正面衝突時，LLM 的推理決策究竟依賴於<b>預訓練形成的醫學語意先驗（Semantic Priors）</b>，還是當前觀察到的<b>資料統計分配結構（Observed Allocation Patterns）</b>？</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">🧠 <b>貝氏認知視角（Bayesian Cognition Framing）</b><br><br>本研究本體不是醫學問答評測或幻覺測試，而是量化 LLM 在高衝突醫療情境下的<b>信念更新動態（Belief Updating Dynamics）</b>。我們本質上是在對 AI 的大腦執行一場「世界模型彈性（World-Model Elasticity）」的心理物理學壓力測試。</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">二、 核心研究假說（Scientific Hypotheses）</div>', unsafe_allow_html=True)
    st.markdown("""
    * **H1 — 語意先驗主導假說 (Semantic Prior Dominance)**：LLM 的推理高度依賴語意標籤。當藥名與基因型清晰時，即使處方行為已與現代 Guideline 發生嚴重衝突，模型仍會產生「先驗剛性（Prior Rigidity）」，固守既有醫學知識，後延更新不完全。
    * **H2 — 統計結構重構假說 (Statistical Structure Reconstruction)**：若分配特徵的統計訊號強度足夠高，即使移除所有治療名稱與生物標記語意，LLM 仍能純粹依靠特徵共變異數（Feature Covariance）與分配不對稱性，逆向重新建構潛在的醫療本體（Latent Ontology）。
    * **H3 — 信念更新臨界分化假說 (Belief Updating Threshold)**：不同技術架構的模型存在顯著的「翻轉臨界點（PFT）」分化，反映出其在「記憶指南」與「相信數據」之間的底層權重博弈差異。
    """)

# ==========================================
# 軌道二：方法學與實驗矩陣
# ==========================================
elif menu == "方法學與核心實驗矩陣":
    st.markdown('<div class="section-header">一、 雙向壓力壓力檢測矩陣（$3 \times 5$ Experimental Grid）</div>', unsafe_allow_html=True)
    st.markdown("本研究設計了**三層語意抽象化條件**與**五個先驗壓力壓力梯度**的交叉矩陣，由公衛統計 RA 建立 $N=2000$ 的完全凍結病患身體特徵共變（Patient Feature Covariance）的虛擬癌症隊列數據。")
    
    # 1. 三層語意抽象化
    st.subheader("1. 語意抽象化層次 (Semantic Abstraction Levels)")
    abs_col1, abs_col2, abs_col3 = st.columns(3)
    with abs_col1:
        st.error("**Condition A: 最低抽象化 (Minimal)**")
        st.caption("僅隱藏藥名（如 Osimertinib 替換為 Treatment X），保留所有標準醫學術語（EGFR, PD-L1, CrCl）。用以測試基礎的治療類型重構能力。")
    with abs_col2:
        st.key("**Condition B: 部分語意抽象化 (Partial)**")
        st.caption("進一步將關鍵分子標記匿名化（如 EGFR 改稱 Biomarker A），但保留其精確的數值分佈與共變結構。用以測試對特定醫學名詞的語意依賴強度。")
    with abs_col3:
        st.success("**Condition C: 完全結構抽象化 (Full Structural)**")
        st.caption("將所有臨床變數徹底符號化（如 Age 改稱 Feature 1, ECOG 改稱 Feature 2）。移除所有醫療語意，僅留下純粹的高維機率統計結構。用以測試純粹的結構推理能力。")

    # 2. 數據源校準
    st.subheader("2. 注入非線性噪聲的數據生成 (Data Generating Process with Real-world Noise)")
    st.markdown("""
    為杜絕人工完美數據造成的「人工可分性（Artificial Separability）」解碼偏誤，公衛 RA 在數據生成階段執行了高度擬真的數據污染：
    * **非線性階梯效應（Non-linear Thresholds）**：器官功能指標（如器官代償 eGFR）在常規區間內設定為線性，但一旦跌破 30（進入 CKD stage 4 臨界點），臨床毒性發生率呈指數級激增，逼迫模型進行非線性表徵映射。
    * **隨機處方噪聲（Physician Heterogeneity Noise）**：故意注入 15% 的隨機處方擾動，模擬現實醫療中不同主治醫師的個人偏好與未觀測到的混雜干擾（Unobserved Confounders）。
    """)

    # 3. 三軌對照基準
    st.subheader("3. 三軌對照盲測實驗 (Three-Arm Benchmark Control)")
    st.markdown("""
    在 `Temperature = 0` 且固定 `Seed` 的決定論環境下，執行以下三軌盲測對照：
    * **理論軌 (Bayesian Ideal Observer)**：利用已知數據生成公式的概似率（Likelihood），算出完美的理論貝氏後驗更新值，作為全篇論文的理論對照極限。
    * **人類軌 (Human Baseline)**：由主導之癌症醫師從 5 個梯度中盲測評估 20 例反常病歷，量化人類專家在診間實際面對衝突證據時的「先驗剛性」。
    * **AI 實驗組**：導入 **Prompt 語意擾動矩陣（Prompt Perturbation）** 與 **特徵消去實驗（Feature Ablation）**，徹底逼出模型內部的真實表徵映射行為。
    """)

# ==========================================
# 軌道三：預期科學結果與圖表
# ==========================================
elif menu == "預期科學結果與圖表":
    st.markdown('<div class="section-header">一、 核心科學指標與量化結果（Primary Quantitative Metrics）</div>', unsafe_allow_html=True)
    
    st.subheader("The Centerpiece: Evidence-Prior Phase Transition Model")
    st.markdown("以下圖表展示了各模型在不同抽象化條件下，隨著先驗壓力扭曲度上升時的**信念更新軌跡（Belief Update Curve）**與**後驗翻轉臨界點（Posterior Flip Threshold, PFT）**。這張圖將是本研究衝擊頂刊的 **Figure 1**。")

    # 繪製高階 Plotly 貝氏更新曲線圖
    fig = go.Figure()
    
    # 理想觀察者線
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[0,30,50,70,90], mode='lines', name="Ideal Bayesian Observer (Theoretical Limit)", line=dict(color="black", width=2, dash="dot")))
    # GPT-4o CondA
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[5,5,5,15,85], mode='lines+markers', name="GPT-4o - Condition A (High Rigidity)", line=dict(color="#EF553B", width=3)))
    # GPT-4o CondC
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[5,32,54,72,92], mode='lines+markers', name="GPT-4o - Condition C (High Adaptability)", line=dict(color="#636EFA", width=3)))
    # Gemini CondA
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[10,10,88,92,95], mode='lines+markers', name="Gemini Pro - Condition A (Early Flip)", line=dict(color="#00CC96", width=3)))

    fig.update_layout(
        xaxis_title="先驗壓力扭曲度 (Guideline Distortion Level %: 正常 ──► 完全反常)",
        yaxis_title="後驗跟從實證數據機率 (Posterior Alignment Probability %)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        yaxis=dict(range=[-5, 105])
    )
    fig.add_shape(type="line", x0=0, y0=50, x1=90, y1=50, line=dict(color="rgba(128,128,128,0.5)", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    # 量化指標解說
    st.markdown('<div class="section-header">二、 預期重大科學發現（Key Expected Insights）</div>', unsafe_allow_html=True)
    
    insight_col1, insight_col2 = st.columns(2)
    with insight_col1:
        st.markdown("""
        ### 1. 發現模型存在「先驗僵固性指數（PRI）」過度自信偏誤
        透過計算模型推論分佈與理想貝氏觀察者之間的 **KL 散度（Kullback-Leibler Divergence）**，我們預期會觀測到在 Condition A 下，當數據衝突達到 50% 的資訊最高熵狀態時，GPT-4o 依然會維持極高強度的報告信心分數（Confidence Score）。這實證了模型存在無法被實證統計數據即時修正的**「教條式盲區（Dogmatic Bias）」**。
        
        ### 2. 精確量化「後驗翻轉臨界點（PFT）」的模型分化
        實驗預期將劃分出兩大 AI 推理流派：Gemini Pro 展現出極低的 PFT（在 30% 扭曲度時即向眼前數據低頭，屬於數據驅動型）；而 GPT-4o 展現出極高 PFT（直到 70% 的極端反常世界才推翻對 XYZ 藥物的猜測，屬於教條僵固型）。
        """)
    with insight_col2:
        st.markdown("""
        ### 3. 識別大模型醫療世界模型的「核心語意錨點」
        透過特徵消去實驗（Feature Ablation），在部分語意抽象化（Condition B）中依序遮蔽原 EGFR 或 Age 欄位。預期可觀察到：當特定高權重分子特徵被抽離後，模型對於反常數據的解碼能力會發生**結構性崩塌（Inference Collapse）**。這能幫全球學界第一次透視出大模型內部高維空間中，究竟是哪些醫學概念支撐起了它的推理邊界。
        """)

    # ==========================================
    # 軌道四：實實戰臨床效益與重大貢獻
    # ==========================================
    st.markdown('<div class="section-header">三、 癌症治療領域的實質重大貢獻（Clinical Significance & Utility）</div>', unsafe_allow_html=True)
    
    contrib1, contrib2, contrib3 = st.columns(3)
    with contrib1:
        st.markdown('<div class="contribution-card">🛡️ <b>為「AI 輔助 Tumor Board / MDT」建立安全監管尺規</b><br><br>這項研究能幫醫院管理層與主治醫師建立客觀的 AI 部署指引。例如，高先驗剛性的模型（如 GPT）適合作為第一線標準指南（NCCN）的嚴格把關者；而高數據敏感型的模型（如 Gemini）則適合作為後端即時監測「真實世界未知藥物副作用或突發毒性訊號」的預警系統，達成適才適所的安全部署。</div>', unsafe_allow_html=True)
    with contrib2:
        st.markdown('<div class="contribution-card">🌍 <b>實證「決策行為可逆向識別性」以打破院際隱私壁壘</b><br><br>若假說 H2 成立（AI 能夠在完全符號化的 Condition C 下重建醫療本體），這將為跨中心癌症真實世界數據（RWD）的去隱私共享帶來顛覆性突破。它證明了未來多中心研究甚至不需要共享敏感的患者基因名稱或藥物標籤，AI 就能在完全符號化的結構中，跨院辨識出實踐偏誤與新藥療效，保障數據安全。</div>', unsafe_allow_html=True)
    with contrib3:
        st.markdown('<div class="contribution-card">🧬 <b>保障精準醫療少數派亞組的生命安全</b><br><br>本研究的「壓力測試」能有效幫精準醫療劃定安全防禦邊界。它赤裸地揭露了 AI 是否會因為盲信指南上的神效，而忽視了眼前複雜共病（如高齡 ILD）正在發生的致命副作用。這對於未來 FDA 對於醫療決策型軟體（SaMD）的監管與核發牌照，提供了直接的計算流行病學量化標準。</div>', unsafe_allow_html=True)

# ==========================================
# 軌道四：互動式壓力測試儀表板 (15格細節)
# ==========================================
else:
    st.header("🎛️ 15格交叉壓力網格細節審查")
    st.caption("請切換不同的語意抽象化條件與扭曲梯度，查看模型在微觀與宏觀上的對齊表現。")
    
    c_col, d_col = st.columns(2)
    with c_col:
        selected_cond = st.selectbox("選擇語意抽象化條件", ["Condition A (Minimal)", "Condition B (Partial)", "Condition C (Full)"])
    with d_col:
        selected_dist = st.selectbox("選擇先驗壓力扭曲度", [0, 30, 50, 70, 90], format_func=lambda x: f"{x}% 扭共變異逆轉")
        
    # 撈出該格子數據並渲染
    cond_map = {"Condition A (Minimal)": "Condition A (Minimal)", "Condition B (Partial)": "Condition B (Partial)", "Condition C (Full)": "Condition C (Full)"}
    target_df = df_adv[(df_adv["Condition"] == cond_map[selected_cond]) & (df_adv["Distortion"] == selected_dist)]
    
    if not target_df.empty:
        st.subheader("🤖 各主流模型在該交叉壓力點下的對齊表現")
        for idx, row in target_df.iterrows():
            with st.container():
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric(label=f"📊 {row['Agent']} 決策翻轉狀態", value="已推推翻先前的指南猜測" if row['P_Alignment'] > 50 else "固守既有醫學指南")
                with col_m2:
                    st.metric(label="跟從實證數據後驗機率 (Posterior Alignment)", value=f"{row['P_Alignment']}%")
                st.markdown("---")
