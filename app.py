import numpy as np
import pandas as pd
import os

# 設定隨機種子，確保 100% 可重現性
np.random.seed(42)
N = 2000

# ==========================================
# 1. 凍結母體：生成完全一致的乳癌病患特徵 (X)
# ==========================================
age = np.random.normal(54, 11, size=N).round(0)  # 乳癌好發年齡
lvef = np.random.normal(62, 6, size=N).round(1)  # 心功能左心室射出分率 (%)
hr_status = np.random.choice([0, 1], size=N, p=[0.3, 0.7]) # 70% 荷爾蒙受體陽性
her2_status = np.random.choice([0, 1, 2], size=N, p=[0.4, 0.4, 0.2]) # 0=陰, 1=Low, 2=陽
brca_mutation = np.random.choice([0, 1], size=N, p=[0.93, 0.07]) # 7% 帶有 BRCA 突變

# 建立凍結特徵矩陣
base_df = pd.DataFrame({
    'Age': age, 'LVEF': lvef, 'HR_Status': hr_status, 
    'HER2_Status': her2_status, 'gBRCA_Mutation': brca_mutation
})

# 建立輸出資料夾
os.makedirs("matrices_data_bc", exist_ok=True)
distortion_levels = [0.0, 0.3, 0.5, 0.7, 0.9]

# ==========================================
# 2. 核心操弄：扭曲醫師開藥行為 (Treatment Allocation)
# ==========================================
for d in distortion_levels:
    # 0% 時：符合常理指南；90% 時：決策邏輯完全與常理逆轉
    # 藥物 X (代表 T-DXd ADC 標靶)
    score_X = (5.0 * (1 - d) * (base_df['HER2_Status'] >= 1)) + (5.0 * d * (base_df['LVEF'] < 45) * (base_df['HR_Status'] == 1))
    # 藥物 Y (代表 Pembrolizumab 免疫治療)
    score_Y = (4.0 * (1 - d) * ((base_df['HR_Status'] == 0) & (base_df['HER2_Status'] == 0))) + (4.0 * d * (base_df['Age'] > 80))
    # 藥物 Z (代表 Olaparib PARP抑制劑)
    score_Z = (5.0 * (1 - d) * (base_df['gBRCA_Mutation'] == 1)) + (5.0 * d * (base_df['gBRCA_Mutation'] == 0))
    
    # Softmax 轉換
    exp_X, exp_Y, exp_Z = np.exp(score_X), np.exp(score_Y), np.exp(score_Z)
    sum_exp = exp_X + exp_Y + exp_Z
    
    prob_X = exp_X / sum_exp
    prob_Y = exp_Y / sum_exp
    prob_Z = exp_Z / sum_exp
    
    treatments = []
    for i in range(N):
        if np.random.rand() < 0.15: # 15% 臨床處方噪聲注入
            treatments.append(np.random.choice(['X', 'Y', 'Z']))
        else:
            treatments.append(np.random.choice(['X', 'Y', 'Z'], p=[prob_X[i], prob_Y[i], prob_Z[i]]))
            
    temp_df = base_df.copy()
    temp_df['Treatment'] = treatments
    
    # ==========================================
    # 3. 三種語意抽象化條件 (Conditions A, B, C)
    # ==========================================
    # Condition A: 僅隱藏藥名
    temp_df.to_csv(f"matrices_data_bc/CondA_Distort_{int(d*100)}.csv", index=False)
    
    # Condition B: 部分語意抽象化 (分子標記匿名化)
    df_B = temp_df.copy().rename(columns={'gBRCA_Mutation': 'Biomarker_A', 'HER2_Status': 'Biomarker_B'})
    df_B.to_csv(f"matrices_data_bc/CondB_Distort_{int(d*100)}.csv", index=False)
    
    # Condition C: 完全結構抽象化 (徹底符號化)
    df_C = temp_df.copy().rename(columns={
        'Age': 'Feature_1', 'LVEF': 'Feature_2', 'HR_Status': 'Feature_3',
        'HER2_Status': 'Feature_4', 'gBRCA_Mutation': 'Feature_5', 'Treatment': 'Target_Class'
    })
    df_C['Target_Class'] = df_C['Target_Class'].map({'X': 'Alpha', 'Y': 'Beta', 'Z': 'Gamma'})
    df_C.to_csv(f"matrices_data_bc/CondC_Distort_{int(d*100)}.csv", index=False)

print("✅ 成功生成乳癌專屬 3x5 交叉壓力測試資料集！")
