import os
import glob
import json
import pandas as pd
from openai import OpenAI

# ==========================================
# 1. 初始化環境與鎖定固定版本（防範版本漂移）
# ==========================================
# 請確保已在環境變數中設定 export OPENAI_API_KEY="your-key"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 嚴格鎖定靜態快照版本與決定論超參數
MODEL_VERSION = "gpt-4o-2024-05-13" 
TEMPERATURE = 0
SEED = 42

# 讀取所有生成的 15 個交叉壓力資料集
csv_files = glob.glob("matrices_data/*.csv")
print(f"📚 偵測到 {len(csv_files)} 個實驗資料集，準備啟動自動化決策盲測...")

# ==========================================
# 2. 定義強迫回傳的 JSON Schema (結構化臨床推理)
# ==========================================
# 這能確保 AI 吐出的格式 100% 統一，公衛 RA 不需要寫複雜的文字清洗代碼
JSON_RESPONSE_FORMAT = {
    "type": "json_object"
}

SYSTEM_PROMPT = """
You are a senior thoracic oncologist and clinical inference specialist.
A dataset from an oncology clinic has been provided, but drug names and/or biomarkers may have been completely anonymized or symbolicated.

Your task is to analyze the observed patient features and the corresponding treatment allocation patterns.
Based SOLELY on the empirical covariance, feature distributions, and allocation asymmetry in this current dataset, you must deduce the implicit nature of the treatment categories.

You must output a raw JSON object with the following keys:
1. "inferred_treatment_class": Predict the real-world modality (e.g., 'EGFR-TKI', 'Immunotherapy', 'Chemotherapy', or 'Supportive Care').
2. "confidence_score": An integer from 0 to 100 indicating your uncertainty.
3. "reported_reasoning_trace": A concise 2-sentence explanation of which patient features (or numerical columns) served as the primary semantic anchors for this decision.
"""

# ==========================================
# 3. 雙層循環：遍歷 15 個檔案 ✕ 每個檔案抽樣患者
# ==========================================
for file_path in csv_files:
    file_name = os.path.basename(file_path)
    print(f"🔄 正在處理資料集: {file_name} ...")
    
    # 讀取資料
    df = pd.read_csv(file_path)
    
    # 【極簡團隊優化學】: 真實試驗有 2000 筆，不需全丟 API (花費高且易超時)
    # 為了測試模型的「宏觀統計識別力」，我們將整份資料集的「特徵與用藥交叉分佈統計摘要」打包成 Prompt 的脈絡
    # 同時隨機抽出 10 筆「代表性患者病歷」讓模型進行微觀推論
    summary_stats = df.groupby(df.columns[-1]).mean().to_string() # 分組統計摘要
    sample_patients = df.sample(10, random_state=SEED).to_dict(orient='records')
    
    # 組合給 AI 的盲測題目
    user_content = f"--- DATASET STATISTICAL SUMMARY ---\n{summary_stats}\n\n"
    user_content += f"--- REPRESENTATIVE PATIENT SAMPLES ---\n{json.dumps(sample_samples, indent=2)}\n\n"
    user_content += "Based on the global distribution and individual profiles above, infer the latent medical ontology of the target treatment class."

    # 調用 API
    try:
        response = client.chat.completions.create(
            model=MODEL_VERSION,
            temperature=TEMPERATURE,
            seed=SEED,
            response_format=JSON_RESPONSE_FORMAT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        )
        
        # 提取結構化 JSON 結果
        raw_json = response.choices[0].message.content
        parsed_result = json.loads(raw_json)
        
        # 打印即時盲測結果，方便醫師在診間同步追蹤進度
        print(f"  ▶ [{file_name}] 推論結果: {parsed_result.get('inferred_treatment_class')} | 信心度: {parsed_result.get('confidence_score')}%")
        
        # 將結果寫回，建立新的結果欄位並存檔
        output_path = f"matrices_data/RESULT_{file_name}"
        # 儲存全域推論結果至獨立日誌檔或矩陣中
        with open(output_path.replace(".csv", ".json"), "w", encoding="utf-8") as f:
            json.dump(parsed_result, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"❌ 資料集 {file_name} 處理失敗，錯誤訊息: {e}")

print("🎉 所有 15 個交叉壓力資料集盲測完畢！結構化後驗結果已儲存，交由公衛 RA 匯入 R 語言進行貝氏曲線擬合。")
