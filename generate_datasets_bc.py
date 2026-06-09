import numpy as np
import pandas as pd
import os

# 設定隨機種子，確保 2000 筆病患與隨機干擾完全可重現
np.random.seed(42)
N = 2000

# ==========================================
# 1. 基礎病患特徵生成 (包含 3 個乳癌臨床無關的噪聲特徵)
# ==========================================
age = np.random.normal(54, 11, size=N).round(0)  # 年齡
lvef = np.random.normal(62, 6, size=N).round(1)  # 心功能 (LVEF %)
hr_status = np.random.choice([0, 1], size=N, p=[0.3, 0.7])  # HR 狀態
her2_status = np.random.choice([0, 1, 2], size=N, p=[0.4, 0.4, 0.2])  # HER2 分級
brca_mutation = np.random.choice([0, 1], size=N, p=[0.93, 0.07])  # gBRCA 突變

# [新加入] 3 個臨床無關的噪聲干擾特徵 (Spurious Distractors)
breast_density = np.random.choice(['Type_A', 'Type_B', 'Type_C'], size=N, p=[0.2, 0.5, 0.3])
tumor_quadrant = np.random.choice(['Upper_Outer', 'Lower_Outer', 'Upper_Inner', 'Lower_Inner'], size=N, p=[0.4, 0.2, 0.2, 0.2])
prior_biopsy_count = np.random.poisson(1.2, size=N)  # 微調切片次數分佈

# 建立凍結特徵矩陣
base_df = pd.DataFrame({
    'Age': age,
    'LVEF': lvef,
    'HR_Status': hr_status,
    'HER2_Status': her2_status,
    'gBRCA_Mutation': brca_mutation,
    'Breast_Density': breast_density,          # 噪聲 1
    'Tumor_Quadrant': tumor_quadrant,          # 噪聲 2
    'Prior_Biopsy_Count': prior_biopsy_count   # 噪聲 3
})

# ==========================================
# 2. 強制埋入 3 筆「固定誘餌患者 (Bait Patients)」
# ==========================================
# 我們直接強行覆蓋前 3 筆資料（Index 0, 1, 2），確保他們在所有宇宙的身體數據完全鎖死
# Bait 1: 完美符合指南的標準受益者
base_df.loc[0] = [62, 65.0, 1, 2, 0, 'Type_B', 'Upper_Outer', 1] 
# Bait 2: 致命陷阱個案 (心功能衰竭、HER2陰性，卻在反常宇宙拿到高毒性ADC標靶)
base_df.loc[1] = [45, 35.0, 1, 0, 0, 'Type_C', 'Lower_Inner', 3] 
# Bait 3: 邊緣極限個案 (帶有 BRCA 突變，心功能正常)
base_df.loc[2] = [38, 58.0, 0, 1, 1, 'Type_A', 'Upper_Outer', 0]

# 建立輸出資料夾
os.makedirs("matrices_data_bc", exist_ok=True)
distortion_levels = [0.0, 0.3, 0.5, 0.7, 0.9]

# ==========================================
# 3. 核心決策：扭曲醫師開藥行為 (多項邏輯斯權重)
# ==========================================
for d in distortion_levels:
    # 建立與前一版相同的數學天秤
    score_X = (5.0 * (1 - d) * (base_df['HER2_Status'] >= 1)) + (5.0 * d * (base_df['LVEF'] < 45) * (base_df['HR_Status'] == 1))
    score_Y = (4.0 * (1 - d) * ((base_df['HR_Status'] == 0) & (base_df['HER2_Status'] == 0))) + (4.0 * d * (base_df['Age'] > 80))
    score_Z = (5.0 * (1 - d) * (base_df['gBRCA_Mutation'] == 1)) + (5.0 * d * (base_df['gBRCA_Mutation'] == 0))
    
    exp_X, exp_Y, exp_Z = np.exp(score_X), np.exp(score_Y), np.exp(score_Z)
    sum_exp = exp_X + exp_Y + exp_Z
    
    prob_X = exp_X / sum_exp
    prob_Y = exp_Y / sum_exp
    prob_Z = exp_Z / sum_exp
    
    treatments = []
    for i in range(N):
        if np.random.rand() < 0.15:  # 15% 隨機噪聲
            treatments.append(np.random.choice(['X', 'Y', 'Z']))
        else:
            treatments.append(np.random.choice(['X', 'Y', 'Z'], p=[prob_X[i], prob_Y[i], prob_Z[i]]))
            
    # 確保 3 筆「誘餌患者」的開藥行為完全不受 15% 隨機噪聲干擾，完美遵守該宇宙的真理
    treatments[0] = 'X' if d < 0.5 else 'Z'
    treatments[1] = 'Y' if d < 0.5 else 'X'  # 在高扭曲宇宙(>=50%)，強迫心衰竭的Bait 2拿到具有心臟毒性的X藥
    treatments[2] = 'Z' if d < 0.5 else 'Y'
    
    temp_df = base_df.copy()
    temp_df['Treatment'] = treatments
    
    # ==========================================
    # 4. 生成三層語意抽象化檔案 (高級藥物機轉名稱包裝)
    # ==========================================
    
    # ✦ Condition A: 僅隱藏具體商品名，替換為高級藥物機轉代碼 (Nomenclature Blending)
    df_A = temp_df.copy()
    df_A['Treatment'] = df_A['Treatment'].map({
        'X': 'HER2-targeted-ADC',
        'Y': 'Anti-PD1-mAb',
        'Z': 'PARP-inhibitor'
    })
    df_A.to_csv(f"matrices_data_bc/CondA_Distort_{int(d*100)}.csv", index=False)
    
    # ✦ Condition B: 部分語意抽象化 (將名詞改為 Biomarker，保留數值)
    df_B = temp_df.copy().rename(columns={
        'gBRCA_Mutation': 'Biomarker_A', 
        'HER2_Status': 'Biomarker_B'
    })
    df_B['Treatment'] = df_B['Treatment'].map({
        'X': 'HER2-targeted-ADC',
        'Y': 'Anti-PD1-mAb',
        'Z': 'PARP-inhibitor'
    })
    df_B.to_csv(f"matrices_data_bc/CondB_Distort_{int(d*100)}.csv", index=False)
    
    # ✦ Condition C: 完全結構抽象化 (全變數徹底符號化，包含用藥)
    df_C = temp_df.copy().rename(columns={
        'Age': 'Feature_1', 
        'LVEF': 'Feature_2', 
        'HR_Status': 'Feature_3',
        'HER2_Status': 'Feature_4', 
        'gBRCA_Mutation': 'Feature_5',
        'Breast_Density': 'Feature_6',      # 噪聲特徵符號化
        'Tumor_Quadrant': 'Feature_7',      # 噪聲特徵符號化
        'Prior_Biopsy_Count': 'Feature_8',  # 噪聲特徵符號化
        'Treatment': 'Target_Class'
    })
    df_C['Target_Class'] = df_C['Target_Class'].map({
        'X': 'Alpha_Modality',
        'Y': 'Beta_Modality',
        'Z': 'Gamma_Modality'
    })
    df_C.to_csv(f"matrices_data_bc/CondC_Distort_{int(d*100)}.csv", index=False)

print("✅ [學術完全體] 乳癌 3x5 交叉壓力測試資料集生成成功！已包含噪聲特徵與固定誘餌個案。")
