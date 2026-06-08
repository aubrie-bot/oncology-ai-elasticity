import numpy as np
import pandas as pd
import os

# 設定隨機種子，確保 100% 可重現性
np.random.seed(42)
N = 2000

# ==========================================
# 1. 凍結母體：生成完全一致的病患身體數值 (X)
# ==========================================
age = np.random.normal(62, 10, size=N).round(0)  # 平均62歲
egfr = np.random.normal(75, 15, size=N).round(1) # 腎功能 eGFR
egfr_mutation = np.random.choice([0, 1], size=N, p=[0.6, 0.4]) # 40% 陽性
pdl1 = np.random.uniform(0, 100, size=N).round(1) # PD-L1 表現量
ild_history = np.random.choice([0, 1], size=N, p=[0.9, 0.1]) # 10% 有間質性肺病

# 建立凍結特徵矩陣
base_df = pd.DataFrame({
    'Age': age, 'eGFR': egfr, 'EGFR_Mutation': egfr_mutation, 
    'PD_L1': pdl1, 'ILD_History': ild_history
})

# 建立輸出資料夾
os.makedirs("matrices_data", exist_ok=False)

distortion_levels = [0.0, 0.3, 0.5, 0.7, 0.9]

# ==========================================
# 2. 核心操弄：扭曲醫師開藥行為 (Treatment Allocation)
# ==========================================
for d in distortion_levels:
    # 決定每個療法的機率權重 (多項式對數機率)
    # 0% 時：EGFR_Mutation 主導 X 藥 (TKI)；90% 時：Age 與 ILD_History 逆轉主導 X 藥
    score_X = (5.0 * (1 - d) * base_df['EGFR_Mutation']) + (5.0 * d * (base_df['Age'] > 75) * base_df['ILD_History'])
    score_Y = (4.0 * (1 - d) * (base_df['PD_L1'] > 50)) + (4.0 * d * (base_df['Age'] < 50))
    score_Z = np.zeros(N) # 基準化療
    
    # Softmax 轉換為機率
    exp_X, exp_Y, exp_Z = np.exp(score_X), np.exp(score_Y), np.exp(score_Z)
    sum_exp = exp_X + exp_Y + exp_Z
    
    prob_X = exp_X / sum_exp
    prob_Y = exp_Y / sum_exp
    prob_Z = exp_Z / sum_exp
    
    # 隨機抽樣處方標籤，並注入 15% 臨床隨機噪聲
    treatments = []
    for i in range(N):
        if np.random.rand() < 0.15: # 15% 噪聲
            treatments.append(np.random.choice(['X', 'Y', 'Z']))
        else:
            treatments.append(np.random.choice(['X', 'Y', 'Z'], p=[prob_X[i], prob_Y[i], prob_Z[i]]))
            
    temp_df = base_df.copy()
    temp_df['Treatment'] = treatments
    
    # ==========================================
    # 3. 三種語意抽象化層次 (Conditions A, B, C)
    # ==========================================
    # Condition A: 僅隱藏藥名
    df_A = temp_df.copy()
    df_A.to_csv(f"matrices_data/CondA_Distort_{int(d*100)}.csv", index=False)
    
    # Condition B: 部分語意抽象化 (變更生物標記名稱)
    df_B = temp_df.copy().rename(columns={'EGFR_Mutation': 'Biomarker_A', 'PD_L1': 'Biomarker_B'})
    df_B.to_csv(f"matrices_data/CondB_Distort_{int(d*100)}.csv", index=False)
    
    # Condition C: 完全結構抽象化 (徹底符號化)
    df_C = temp_df.copy().rename(columns={
        'Age': 'Feature_1', 'eGFR': 'Feature_2', 'EGFR_Mutation': 'Feature_3',
        'PD_L1': 'Feature_4', 'ILD_History': 'Feature_5', 'Treatment': 'Target_Class'
    })
    # 將 Target_Class 內容換成符號
    df_C['Target_Class'] = df_C['Target_Class'].map({'X': 'Alpha', 'Y': 'Beta', 'Z': 'Gamma'})
    df_C.to_csv(f"matrices_data/CondC_Distort_{int(d*100)}.csv", index=False)

print("✅ 成功生成 3x5 交叉壓力測試資料集（共 15 個 CSV 檔案儲存於 matrices_data 資料夾）")
