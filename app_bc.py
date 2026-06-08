import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import json

st.set_page_config(page_title="LLM World-Model Elasticity in Breast Cancer", layout="wide")

st.markdown('<h1 style="color: #1E3A8A;">🩺 癌症決策行為之 AI 世界模型彈性量化平台 (乳癌專題)</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-style: italic; font-size: 1.2rem;">When Clinical Semantics Disappear: Belief Updating Dynamics of LLMs Under Abstracted Breast Cancer Treatment Allocation Structures</p>', unsafe_allow_html=True)

menu = st.sidebar.radio("📋 研發協議導覽", ["乳癌研究提案與背景", "方法學與核心實驗矩陣", "預期科學結果（乳癌）", "互動式壓力測試儀表板"])

# ==========================================
# 數據模擬器 (乳癌專屬 PFT 軌跡)
# ==========================================
def load_bc_results():
    distortions = [0, 30, 50, 70, 90]
    conditions = ["Condition A (Minimal)", "Condition B (Partial)", "Condition C (Full)"]
    data_list = []
    for dist in distortions:
        data_list.append({"Agent": "Ideal Bayesian Observer", "Condition": "Theoretical Limit", "Distortion": dist, "P_Alignment": dist if dist >= 50 else (dist * 0.8)})
        # GPT-4o 表現出高度乳癌先驗教條
        p_gpt = 5 if dist < 70 else 88
        data_list.append({"Agent": "GPT-4o (Snapshot)", "Condition": "Condition A (Minimal)", "Distortion": dist, "P_Alignment": p_gpt})
        data_list.append({"Agent": "GPT-4o (Snapshot)", "Condition": "Condition C (Full)", "Distortion": dist, "P_Alignment": dist + 4})
        # Gemini Pro 數據驅動
        p_gem = 8 if dist < 30 else 94
        data_list.append({"Agent": "Gemini Pro (Snapshot)", "Condition": "Condition A (Minimal)", "Distortion": dist, "P_Alignment": p_gem})
    return pd.DataFrame(data_list)

df_bc = load_bc_results()

# 軌道一：研究背景
if menu == "乳癌研究提案與背景":
    st.header("一、 研究背景與乳癌臨床痛點")
    st.markdown("""
    在乳癌的臨床治療中，患者具備高度的**分子異質性**（如 HR 陽性、HER2 表現度分級、以及遺傳性 gBRCA 突變狀態）。隨著新興藥物如**抗體藥物複合體（ADC，如 T-DXd）**與**免疫檢查點抑制劑（Pembrolizumab）**進入指南，開藥決策變得極其複雜。
    
    目前引入 AI 輔助乳癌 Tumor Board 的評測存在巨大盲區：**我們無法辨識模型是真正理解了患者身體特徵與藥物毒性之間的因果統計分佈，還是僅死記硬背了字面指南。** 如果 AI 只是教條式背誦，在面對具有嚴重共病（如心功能 LVEF 不全）的特殊亞組時，將會盲目開出具有心臟毒性的抗 HER2 標靶藥物，釀成致命事故。
    """)
    st.info("🎯 **核心研究問題**：當乳癌的臨床語意與關鍵分子標記被逐步移除、且實證數據與 NCCN 指南發生正面衝突時，LLM 究竟是固守訓練時記憶的世界模型（Prior Rigidity），還是能展現出根據特徵共變異數重新推導潛在醫療本體（Latent Ontology）的貝氏認知彈性？")

# 軌道二：方法學
elif menu == "方法學與核心實驗矩陣":
    st.header("二、 方法學與乳癌特徵操弄")
    st.markdown("""
    本研究由乳癌專科醫師與公衛統計團隊合作，凍結 $N=2000$ 筆乳癌病患之身體數據（含 `Age`, `LVEF心功能`, `HR狀態`, `HER2分級`, `gBRCA突變`），注入 **15% 隨機處方噪聲** 與 **LVEF 跨越 45% 的非線性毒性爆發階梯效應**。
    
    透過交叉 3 種語意抽象化條件（Cond A：僅隱藏藥名；Cond B：隱藏 HER2/BRCA 名稱改稱 Biomarker；Cond C：變數全符號化）與 5 個扭曲梯度（0% 符合指南 ──► 90% 完全逆轉，即心衰竭患者才給予 T-DXd 標靶），全面壓測 LLM 的大腦。
    """)

# 軌道三：預期結果
elif menu == "預期科學結果（乳癌）":
    st.header("三、 預期科學結果與主要貢獻")
    st.subheader("Figure 1: 乳癌世界模型彈性更新曲線 (Belief Update Curve)")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[0,30,50,70,90], mode='lines', name="Ideal Bayesian Observer", line=dict(color="black", dash="dot")))
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[5,5,5,15,88], mode='lines+markers', name="GPT-4o - Condition A (乳癌先驗剛性高)", line=dict(color="#EF553B", width=3)))
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[5,32,54,72,92], mode='lines+markers', name="GPT-4o - Condition C (符號化數據彈性高)", line=dict(color="#636EFA", width=3)))
    fig.add_trace(go.Scatter(x=[0,30,50,70,90], y=[8,8,94,94,95], mode='lines+markers', name="Gemini Pro - Condition A (數據敏感型)", line=dict(color="#00CC96", width=3)))
    
    fig.update_layout(xaxis_title="乳癌指南扭曲度 (0%符合常理 ──► 90%完全反常)", yaxis_title="後驗跟從實證數據機率 (%)", template="plotly_white")
    fig.add_shape(type="line", x0=0, y0=50, x1=90, y1=50, line=dict(color="gray", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    ### 📊 臨床實質三大效益
    1. **避免乳癌特殊亞組的決策偏誤**：量化 AI 是否會因盲信指南神效，而忽視高齡心衰竭患者眼前的真實毒性，劃定 SaMD（醫療決策軟體）安全監管邊界。
    2. **打破院際乳癌隱私共享壁壘**：若實證 Condition C 下模型仍能重構乳癌本體，則證明未來跨院真實世界數據（RWD）研究不需共享敏感藥名與基因，僅靠去識別化統計結構即可由 AI 逆向辨識新藥真實療效。
    3. **建立乳癌 MDT 部署尺規**：教條型 AI（高 PFT）適合用於一線標準處方把關；數據敏感型 AI（低 PFT）適合作為未預期臨床毒性訊號的即時預警。
    """)

else:
    st.header("🎛️ 15格乳癌壓力網格審查")
    cond = st.selectbox("選擇語意條件", ["Condition A (Minimal)", "Condition C (Full)"])
    dist = st.selectbox("選擇扭曲度", [0, 30, 50, 70, 90])
    
    sub_df = df_bc[(df_bc["Condition"] == cond) & (df_bc["Distortion"] == dist)]
    st.write(sub_df)
