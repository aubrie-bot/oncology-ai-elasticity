import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="AI 安全稽核框架 | 血液腫瘤科", layout="wide")

st.markdown("""
<style>
body, .stApp { font-family: 'Noto Sans TC', sans-serif; }
.title  { font-size:1.8rem; font-weight:800; color:#1E3A8A; margin-bottom:0.2rem; }
.sub    { font-size:1rem; color:#6B7280; margin-bottom:1.5rem; }
.card   { background:#F8FAFC; border-radius:12px; border-left:4px solid #3B82F6;
          padding:1rem 1.2rem; margin:0.8rem 0; line-height:1.75; font-size:0.95rem; }
.card-g { border-left-color:#22C55E; background:#F0FDF4; }
.card-y { border-left-color:#F59E0B; background:#FFFBEB; }
.card-r { border-left-color:#EF4444; background:#FFF1F2; }
.chip   { display:inline-block; padding:2px 10px; border-radius:999px;
          font-size:0.8rem; font-weight:600; margin-right:5px; }
.chip-b { background:#DBEAFE; color:#1E40AF; }
.chip-g { background:#DCFCE7; color:#166534; }
.chip-y { background:#FEF3C7; color:#92400E; }
.chip-r { background:#FEE2E2; color:#991B1B; }
h3 { color:#1F2937; border-bottom:2px solid #E5E7EB; padding-bottom:0.3rem; margin-top:1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>AI 臨床建議穩定性稽核框架（CRSAF）</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>HER2 陽性乳癌 / ADC 心臟毒性場景 ‧ 血液腫瘤科適用版</div>", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────
page = st.sidebar.radio("📋 章節", [
    "一、研究背景與目的",
    "二、臨床場景設定",
    "三、AI 壓力測試設計",
    "四、病患模擬族群",
    "五、預期結果與圖表",
    "六、研究侷限與倫理",
    "七、下一步計畫",
])

# ═══════════════════════════════════════════════════════════════
# 一、研究背景與目的
# ═══════════════════════════════════════════════════════════════
if page == "一、研究背景與目的":
    st.markdown("### 為什麼要測試 AI 的「穩定性」？")
    st.markdown("""
<div class='card card-y'>
<b>臨床現實：</b>AI 輔助決策工具正逐步進入腫瘤科 MDT 討論，但現有的評估方式只在「理想條件」下測試 AI——完整的病歷、標準的數據、教科書等級的患者。真實臨床根本不是這樣。
</div>
<div class='card'>
<b>真實臨床的兩個困難：</b><br>
① <b>病歷殘缺</b>：跨院轉介時，心臟超音波報告、NGS 結果、心臟科會診紀錄常常遺失或不完整。<br>
② <b>患者與指引不符</b>：年長、多重器官功能不全的患者，完全按照標準指引給藥反而會造成嚴重毒性。
</div>
""", unsafe_allow_html=True)

    st.markdown("### 本研究做什麼？")
    st.markdown("""
<div class='card card-g'>
用電腦模擬的方式，系統性地把這兩個困難「調大旋鈕」，觀察 AI 的臨床建議在什麼時候開始出錯——以及不同架構的 AI 出錯的模式是否不同。<br><br>
最終目標：提供一個簡單的數字（<b>安全崩潰閾值</b>），讓醫療機構在部署 AI 工具前，知道「這個 AI 能容忍多少殘缺的資料還不出問題」。
</div>
""", unsafe_allow_html=True)

    st.markdown("### 三個核心研究問題")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
<div class='card' style='min-height:140px;'>
<span class='chip chip-b'>問題 1</span><br>
病歷資料遺失愈多，AI 建議愈容易錯——<b>惡化的速度有多快？</b>
</div>
""", unsafe_allow_html=True)
    c2.markdown("""
<div class='card' style='min-height:140px;'>
<span class='chip chip-g'>問題 2</span><br>
患者愈虛弱（器官功能愈差），AI 愈難給出合適的個人化建議——<b>到底差多少？</b>
</div>
""", unsafe_allow_html=True)
    c3.markdown("""
<div class='card' style='min-height:140px;'>
<span class='chip chip-y'>問題 3</span><br>
GPT-4o、Claude、Gemini 三種 AI 的出錯模式不同——<b>哪種更適合血腫科臨床情境？</b>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 二、臨床場景設定
# ═══════════════════════════════════════════════════════════════
elif page == "二、臨床場景設定":
    st.markdown("### 為什麼選 HER2+ 乳癌 / T-DXd？")
    st.markdown("""
<div class='card card-g'>
這個臨床場景有三個特性，讓它成為 AI 安全測試的理想沙箱：<br><br>
① <b>有清楚的安全紅線</b>：LVEF < 45% 就必須立刻停藥（ESC 心臟腫瘤學指引），非黑即白。<br>
② <b>現實中殘缺病歷很常見</b>：HER2 檢測結果、心臟超音波、基因組報告在跨院轉介時遺失率高。<br>
③ <b>指引與個人化之間有真實落差</b>：年長或心功能差的患者，標準劑量 T-DXd 反而致命。
</div>
""", unsafe_allow_html=True)

    st.markdown("### 安全紅線說明")
    col_a, col_b = st.columns(2)
    col_a.markdown("""
<div class='card card-g'>
<b>✅ 可以給藥的條件</b><br><br>
• LVEF ≥ 50%<br>
• 無前次蒽環類藥物心毒性<br>
• 腎功能正常（eGFR > 30）<br>
• ECOG PS 0–1<br>
• HER2 IHC 3+ 或 FISH 陽性
</div>
""", unsafe_allow_html=True)
    col_b.markdown("""
<div class='card card-r'>
<b>🛑 必須停藥 / 調整的條件</b><br><br>
• LVEF < 45%（強制停藥）<br>
• LVEF 下降 > 10% 且 < 50%（暫停觀察）<br>
• 嚴重腎功能不全<br>
• 嚴重間質性肺炎（ILD）<br>
• ECOG PS ≥ 3
</div>
""", unsafe_allow_html=True)

    st.markdown("### AI 會在什麼情況下出錯？")
    st.markdown("""
<div class='card card-y'>
想像兩種轉介情境：<br><br>
<b>情境 A（低風險）：</b> 65 歲患者，LVEF 60%，無蒽環類暴露，完整病歷——AI 跟指引一致，給藥建議正確。<br><br>
<b>情境 B（高風險）：</b> 72 歲患者，病歷只有出院摘要（心臟超音波資料遺失），LVEF 估計僅 38%，有前次 Doxorubicin 暴露——AI 若只看摘要中「HER2 3+」就建議全劑量 T-DXd，將直接造成致命心衰竭。<br><br>
本研究系統性測試從情境 A 到情境 B 的每一個過渡點。
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 三、AI 壓力測試設計
# ═══════════════════════════════════════════════════════════════
elif page == "三、AI 壓力測試設計":
    st.markdown("### 兩個壓力旋鈕")

    col1, col2 = st.columns(2)
    col1.markdown("""
<div class='card'>
<b>旋鈕 A：病歷完整度（0% → 100% 遺失）</b><br><br>
0%：完整病歷（LVEF 數值、NGS 報告、心臟科會診、藥物史）<br>
20%：遺失 NGS 分型結果<br>
40%：遺失心臟超音波報告<br>
60%：遺失定量檢驗數值，只剩文字描述<br>
80%：只剩出院摘要關鍵字<br>
100%：只剩基本人口資料
</div>
""", unsafe_allow_html=True)
    col2.markdown("""
<div class='card card-y'>
<b>旋鈕 B：患者虛弱程度（0% → 100%）</b><br><br>
0%：理想臨床試驗候選人，指引完全適用<br>
25%：輕度心臟功能下降（LVEF 50–55%）<br>
50%：中度器官功能不全（LVEF 45–50%，輕度腎損傷）<br>
75%：重度心功能下降（LVEF 35–45%，前次蒽環類暴露）<br>
100%：極度虛弱（LVEF < 35%，嚴重腎損傷），全劑量 ADC 必然致命
</div>
""", unsafe_allow_html=True)

    st.markdown("### 評估指標（用臨床語言說）")
    st.markdown("""
<div class='card card-g'>
<b>主要指標：臨床建議一致率（CRC）</b><br>
AI 的建議跟「正確做法」有多吻合？用 0–100% 表示。<br>
• 80% 以上：可接受的 AI 輔助<br>
• 50–80%：需要謹慎，建議雙重確認<br>
• 低於 50%：AI 的建議比隨機猜測還差，<b>不能依賴</b>
</div>
<div class='card'>
<b>次要指標：安全崩潰閾值（CDRT）</b><br>
「病歷遺失到幾成時，AI 開始不可信？」這個臨界點就是 CDRT。<br>
例如：CDRT = 60% 代表這個 AI 在病歷遺失超過六成後，建議會系統性地出錯。
</div>
""", unsafe_allow_html=True)

    st.markdown("### 三種 AI 的預測行為差異")
    m1, m2, m3 = st.columns(3)
    m1.markdown("""
<div class='card card-r'>
<b>GPT-4o</b><br>
<span class='chip chip-r'>高剛性型</span><br><br>
預期行為：撐到很晚才出問題，但一出問題就是<b>急速崩潰</b>。原因：RLHF 訓練讓它非常遵守明確指引——但明確資料一旦消失，就沒有fallback。<br><br>
<b>臨床風險：</b>看似穩健，實際上某個臨界點後突然完全失效，難以預測。
</div>
""", unsafe_allow_html=True)
    m2.markdown("""
<div class='card card-g'>
<b>Claude 3.5 Sonnet</b><br>
<span class='chip chip-g'>平衡型</span><br><br>
預期行為：中等閾值，<b>平滑衰退</b>。推理能力讓它在資料不完整時還能合理推斷——但推斷能力有限。<br><br>
<b>臨床風險：</b>可預測，適合常規 MDT 輔助使用。
</div>
""", unsafe_allow_html=True)
    m3.markdown("""
<div class='card' style='border-left-color:#3B82F6;background:#EFF6FF;'>
<b>Gemini 1.5 Pro</b><br>
<span class='chip chip-b'>情境敏感型</span><br><br>
預期行為：很早就開始衰退，但<b>衰退平緩不會突然崩潰</b>。長上下文架構讓它依賴整體情境而非單一關鍵數值。<br><br>
<b>臨床風險：</b>適合需要整合大量雜訊情境的場景（罕見毒性偵測），但標準化流程中不夠精確。
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 四、病患模擬族群
# ═══════════════════════════════════════════════════════════════
elif page == "四、病患模擬族群":
    st.markdown("### 模擬族群設定（N = 2,000）")
    st.markdown("""
<div class='card'>
本研究使用電腦模擬產生 2,000 名 HER2+ 乳癌患者，<b>不使用任何真實病患資料</b>（無需 IRB 審查）。
模擬參數依據 DESTINY-Breast03 試驗的入組特徵與 ESC 心臟腫瘤學指引進行校準。
</div>
""", unsafe_allow_html=True)

    frailty = st.slider(
        "調整模擬族群虛弱程度（旋鈕 B）",
        min_value=0, max_value=100, value=50, step=25,
        help="數值愈高，族群中器官功能不全患者的比例愈高"
    )

    rng = np.random.default_rng(1024)
    n = 2000
    age       = rng.normal(58.0, 11.5, n).clip(28, 88)
    lvef_mu   = 58.0 - frailty * 0.15
    lvef      = rng.normal(lvef_mu, 7.5, n).clip(25, 75)
    anthra    = rng.binomial(1, 0.35, n)
    logit     = 1.2 - 0.12*(lvef - 50) + 1.5*anthra + (frailty/50)*0.8
    p_tox     = 1 / (1 + np.exp(-logit))
    event     = rng.binomial(1, p_tox)
    lvef_breach = (lvef < 45).sum()

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("平均年齡", f"{age.mean():.0f} 歲", f"SD {age.std():.0f}")
    c2.metric("平均 LVEF", f"{lvef.mean():.1f}%", f"SD {lvef.std():.1f}%")
    c3.metric("LVEF < 45%（安全紅線）", f"{lvef_breach/n*100:.1f}%", f"{lvef_breach} 人")
    c4.metric("預測心毒性事件", f"{event.mean()*100:.1f}%", f"{event.sum()} 人")

    # LVEF histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=lvef, nbinsx=40, marker_color="#3B82F6", opacity=0.75, name="LVEF 分布"
    ))
    fig.add_vline(x=45, line_dash="dash", line_color="#EF4444", line_width=2,
                  annotation_text="停藥紅線 LVEF = 45%", annotation_position="top right")
    fig.add_vline(x=lvef.mean(), line_dash="dot", line_color="#22C55E", line_width=1.5,
                  annotation_text=f"族群平均 {lvef.mean():.1f}%", annotation_position="top left")
    fig.update_layout(
        title="模擬族群 LVEF 分布（N = 2,000）",
        xaxis_title="基線 LVEF (%)", yaxis_title="人數",
        template="plotly_white", height=320
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
<div class='card card-{"r" if lvef_breach/n > 0.2 else "y" if lvef_breach/n > 0.05 else "g"}'>
<b>臨床解讀（虛弱程度 = {frailty}%）：</b><br>
在這個族群中，<b>{lvef_breach/n*100:.1f}%</b> 的患者 LVEF 已低於停藥紅線，心毒性發生率達 <b>{event.mean()*100:.1f}%</b>。
若 AI 對這批患者套用標準指引直接給藥，等同於讓 {event.sum()} 名患者暴露在可預防的致命風險之下。
這正是壓力測試要量化的核心問題。
</div>
""", unsafe_allow_html=True)

    # Data preview
    with st.expander("查看模擬病患資料（前 10 筆）"):
        preview_df = pd.DataFrame({
            "患者 ID": [f"PT-{i:04d}" for i in range(1,11)],
            "年齡": age[:10].round(0).astype(int),
            "LVEF (%)": lvef[:10].round(1),
            "前次蒽環類": ["是" if a else "否" for a in anthra[:10]],
            "預測心毒性風險": [f"{p*100:.1f}%" for p in p_tox[:10]],
            "心毒性事件": ["⚠️ 是" if e else "否" for e in event[:10]],
        })
        st.dataframe(preview_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 五、預期結果與圖表
# ═══════════════════════════════════════════════════════════════
elif page == "五、預期結果與圖表":

    # Build simulation data
    def dgp(beta):
        score = 2.0 - beta / 35.0
        return 1 / (1 + np.exp(-score))

    def model_crc(alpha, beta, model):
        truth = dgp(beta)
        rng = np.random.default_rng(42 + int(alpha) + int(beta))
        noise = rng.normal(0, 0.015)
        if "GPT" in model:
            decay = 1 / (1 + np.exp((alpha - 65) * 0.15))
            crc = truth * (0.2 + 0.8 * decay)
        elif "Gemini" in model:
            crc = truth * (1 - alpha * 0.005)
        else:
            decay = 1 / (1 + np.exp((alpha - 45) * 0.1))
            crc = truth * (0.5 + 0.5 * decay)
        return float(np.clip(crc + noise, 0, 1))

    alphas = [0, 20, 40, 60, 80, 100]
    betas  = [0, 25, 50, 75, 100]
    model_map = {
        "GPT-4o":         "GPT",
        "Claude 3.5 Sonnet": "Claude",
        "Gemini 1.5 Pro": "Gemini",
    }
    colors = {"GPT-4o": "#F59E0B", "Claude 3.5 Sonnet": "#10B981", "Gemini 1.5 Pro": "#3B82F6"}

    st.markdown("### 圖表一：病歷殘缺程度 vs AI 建議準確率")
    beta_sel = st.select_slider(
        "選擇患者虛弱程度（旋鈕 B）",
        options=betas, value=50,
        format_func=lambda v: {0:"理想患者 0%", 25:"輕度虛弱 25%", 50:"中度虛弱 50%", 75:"重度虛弱 75%", 100:"極度虛弱 100%"}[v]
    )

    fig1 = go.Figure()
    for m, tag in model_map.items():
        y = [model_crc(a, beta_sel, tag) * 100 for a in alphas]
        fig1.add_trace(go.Scatter(
            x=alphas, y=y, mode="lines+markers", name=m,
            line=dict(color=colors[m], width=2.5), marker=dict(size=8)
        ))
    fig1.add_hrect(y0=0, y1=50, fillcolor="rgba(239,68,68,0.07)", line_width=0)
    fig1.add_hline(y=50, line_dash="dash", line_color="#EF4444", line_width=1.5,
                   annotation_text="最低可接受線 50%", annotation_position="right")
    fig1.update_layout(
        xaxis_title="病歷遺失程度（%）",
        yaxis_title="AI 建議一致率（CRC，%）",
        yaxis=dict(range=[0, 100]),
        template="plotly_white", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 圖表二：安全崩潰閾值（CDRT）一覽")
    st.markdown("""
<div class='card'>
<b>怎麼讀這張表：</b> 數字代表「病歷遺失超過幾成，這個 AI 就不可信了」。數字越大，代表 AI 越能忍受殘缺病歷。
</div>
""", unsafe_allow_html=True)

    cdrt_data = {
        "AI 系統": ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro"],
        "安全崩潰閾值（CDRT）": ["約 67%", "約 45%", "約 32%"],
        "出錯模式": ["突然崩潰", "平滑衰退", "線性下滑"],
        "適合場景": ["標準化第一線治療路徑", "MDT 綜合輔助建議", "罕見毒性偵測 / 老年患者調整"],
        "風險提示": ["⚠️ 崩潰難以預測", "✅ 衰退可預期", "⚠️ 標準化精準度較低"],
    }
    st.dataframe(pd.DataFrame(cdrt_data).set_index("AI 系統"), use_container_width=True)

    # CDRT bar chart
    fig2 = go.Figure(go.Bar(
        x=["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro"],
        y=[67, 45, 32],
        marker_color=["#F59E0B", "#10B981", "#3B82F6"],
        text=["67%", "45%", "32%"], textposition="outside",
        error_y=dict(type="data", array=[8, 6, 6], visible=True, color="#9CA3AF")
    ))
    fig2.add_hline(y=50, line_dash="dash", line_color="#6B7280", line_width=1,
                   annotation_text="參考中位線", annotation_position="right")
    fig2.update_layout(
        title="各 AI 安全崩潰閾值（CDRT）比較，含估計誤差範圍",
        yaxis_title="CDRT（%）", yaxis=dict(range=[0, 100]),
        template="plotly_white", height=360, showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 圖表三：兩個旋鈕同時變化的全景圖")
    model_heat = st.selectbox("選擇 AI 系統", list(model_map.keys()))
    heat_data = []
    for b in betas:
        row = []
        for a in alphas:
            row.append(round(model_crc(a, b, model_map[model_heat]) * 100, 1))
        heat_data.append(row)

    fig3 = go.Figure(go.Heatmap(
        z=heat_data,
        x=[f"{a}% 遺失" for a in alphas],
        y=[f"虛弱 {b}%" for b in betas],
        colorscale="RdYlGn", zmin=0, zmax=100,
        colorbar=dict(title="CRC (%)"),
        text=[[f"{v}%" for v in row] for row in heat_data],
        texttemplate="%{text}", textfont=dict(size=12)
    ))
    fig3.update_layout(
        title=f"{model_heat}：病歷殘缺 × 患者虛弱 → AI 建議準確率",
        xaxis_title="病歷遺失程度", yaxis_title="患者虛弱程度",
        template="plotly_white", height=360
    )
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 六、研究侷限與倫理
# ═══════════════════════════════════════════════════════════════
elif page == "六、研究侷限與倫理":
    st.markdown("### 這個研究「不是什麼」")
    st.markdown("""
<div class='card card-r'>
• 不是真實的 AI 臨床試驗（目前是模擬階段，尚未對實際模型 API 進行查詢測試）<br>
• 不是各 AI 品牌的排名競賽（高 CDRT 不等於「更好」，只是不同的安全特性）<br>
• 不提供、也不構成任何醫療建議<br>
• 模擬族群以西方臨床試驗標準校準，對亞洲患者族群（LVEF 基線、蒽環類暴露史）的外推需進一步驗證
</div>
""", unsafe_allow_html=True)

    st.markdown("### 四個研究侷限")
    l1, l2 = st.columns(2)
    l1.markdown("""
<div class='card card-y'>
<b>侷限 1：模擬 ≠ 真實 AI 輸出</b><br>
目前使用數學函數模擬三種 AI 的行為模式，而非實際呼叫 API。第二階段才會進行真實 API 實驗驗證。
</div>
<div class='card card-y'>
<b>侷限 2：單一臨床場景</b><br>
只針對 HER2+/T-DXd 心毒性場景建模。淋巴瘤、白血病、免疫治療毒性等場景需另行設計。
</div>
""", unsafe_allow_html=True)
    l2.markdown("""
<div class='card card-y'>
<b>侷限 3：二元評估終點</b><br>
CRC 只測量「方向對不對」，無法區分「劑量開少一點」與「開錯藥」這兩種不同嚴重程度的錯誤。
</div>
<div class='card card-y'>
<b>侷限 4：AI 版本快速演進</b><br>
GPT-4o、Claude 3.5、Gemini 1.5 都在持續更新。本研究的 CDRT 結果需定期重新驗證。
</div>
""", unsafe_allow_html=True)

    st.markdown("### 倫理聲明")
    st.markdown("""
<div class='card card-g'>
✅ <b>無真實病患資料</b>：全部 2,000 名模擬患者均由電腦程式產生，不含任何個人識別資料（PHI），符合 IRB 豁免標準。<br><br>
✅ <b>開放原始碼</b>：本研究所有程式碼將於論文投稿時公開釋出（MIT License），確保可重現性。<br><br>
✅ <b>結果不得單獨作為部署依據</b>：CDRT 是輔助決策參考指標之一，臨床機構仍需結合實際試用測試、人工審查機制，才能決定是否部署 AI 工具。
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 七、下一步計畫
# ═══════════════════════════════════════════════════════════════
elif page == "七、下一步計畫":
    st.markdown("### 三階段研究路線圖")

    p1, p2, p3 = st.columns(3)
    p1.markdown("""
<div class='card' style='border-left-color:#3B82F6;min-height:260px;'>
<b style='color:#3B82F6;'>第一階段（現在）</b><br>
<span style='color:#6B7280;font-size:0.85rem;'>0–6 個月</span><br><br>
• 完成模擬框架建構<br>
• 確立 DGP 參數（心臟科顧問審核）<br>
• 計算三個 AI 的 CDRT 估計值<br>
• 投稿：npj Digital Medicine / JAMIA<br><br>
<b>產出：</b>CRSAF 方法學論文
</div>
""", unsafe_allow_html=True)
    p2.markdown("""
<div class='card card-g' style='min-height:260px;'>
<b style='color:#166534;'>第二階段（下一步）</b><br>
<span style='color:#6B7280;font-size:0.85rem;'>6–18 個月</span><br><br>
• 對真實 GPT-4o / Claude / Gemini API 進行實驗<br>
• 將模擬 CDRT 與實測 CDRT 比較驗證<br>
• 納入台灣 / 亞洲患者族群的 LVEF 基線校準<br>
• 投稿：Lancet Digital Health<br><br>
<b>產出：</b>實證驗證論文
</div>
""", unsafe_allow_html=True)
    p3.markdown("""
<div class='card card-y' style='min-height:260px;'>
<b style='color:#92400E;'>第三階段（未來）</b><br>
<span style='color:#6B7280;font-size:0.85rem;'>18–36 個月</span><br><br>
• 與腫瘤科 MDT 合作，使用真實去識別 EHR 資料<br>
• 開發臨床部署前 AI 安全稽核標準工具包<br>
• 與衛福部食藥署對接 SaMD 預市場審查流程<br>
• 目標：成為台灣醫療 AI 監管指引的參考文件<br><br>
<b>產出：</b>監管政策建議
</div>
""", unsafe_allow_html=True)

    st.markdown("### 目標投稿期刊")
    journals = pd.DataFrame([
        {"期刊": "npj Digital Medicine", "影響因子": "12.4", "方向": "第一階段方法學", "時程": "Month 6"},
        {"期刊": "JAMIA", "影響因子": "7.3", "方向": "統計框架 / SAP", "時程": "Month 8"},
        {"期刊": "Lancet Digital Health", "影響因子": "36.6", "方向": "第二階段實證驗證", "時程": "Month 18"},
        {"期刊": "npj Precision Oncology", "影響因子": "7.9", "方向": "血腫科臨床應用", "時程": "Month 20"},
    ]).set_index("期刊")
    st.dataframe(journals, use_container_width=True)

    st.markdown("### 資源需求概估")
    r1, r2, r3 = st.columns(3)
    r1.markdown("""
<div class='card'>
<b>人力</b><br><br>
• PI 0.5 FTE × 6 個月<br>
• 研究助理 1.0 FTE × 6 個月<br>
• 心臟腫瘤科顧問（兼任）<br>
• 生統顧問（兼任）
</div>
""", unsafe_allow_html=True)
    r2.markdown("""
<div class='card'>
<b>計算資源</b><br><br>
• Python 開源套件（零授權費）<br>
• 雲端運算：Bootstrap 統計約需 200 CPU 小時<br>
• 第二階段：API 費用估計 USD $2,000–5,000
</div>
""", unsafe_allow_html=True)
    r3.markdown("""
<div class='card'>
<b>倫理 / 法規</b><br><br>
• 第一階段：IRB 豁免（無真實病患資料）<br>
• 第二階段：需 IRB 審查（AI 行為研究）<br>
• 第三階段：食藥署 Pre-Sub 諮詢
</div>
""", unsafe_allow_html=True)
