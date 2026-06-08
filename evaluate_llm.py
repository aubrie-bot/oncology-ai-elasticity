import os
import glob
import json
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL_VERSION = "gpt-4o-2024-05-13" 

csv_files = glob.glob("matrices_data/*.csv")
print(f"📚 偵測到 {len(csv_files)} 個乳癌壓力資料集，啟動自動化盲測...")

SYSTEM_PROMPT = """
You are a senior breast medical oncologist and clinical inference specialist.
A dataset from a breast cancer clinic has been provided, but regimen names and biomarkers are anonymized or symbolicated.
Based SOLELY on the empirical covariance and allocation asymmetry in this current dataset, deduce the implicit nature of the treatment categories.

Output a raw JSON object with keys:
1. "inferred_treatment_class": Predict real-world modality (e.g., 'Anti-HER2 ADC', 'PARP Inhibitor', 'Checkpoint Inhibitor', or 'Supportive Care').
2. "confidence_score": Integer 0-100.
3. "reported_reasoning_trace": 2-sentence explanation of which patient features served as semantic anchors.
"""

for file_path in csv_files:
    file_name = os.path.basename(file_path)
    df = pd.read_csv(file_path)
    
    # 打包宏觀分佈統計摘要與微觀抽樣
    summary_stats = df.groupby(df.columns[-1]).mean().to_string()
    sample_patients = df.sample(10, random_state=42).to_dict(orient='records')
    
    user_content = f"--- DATASET STATISTICAL SUMMARY ---\n{summary_stats}\n\n"
    user_content += f"--- REPRESENTATIVE PATIENT SAMPLES ---\n{json.dumps(sample_patients, indent=2)}\n\n"
    user_content += "Infer the latent breast cancer medical ontology of the target treatment class."

    try:
        response = client.chat.completions.create(
            model=MODEL_VERSION, temperature=0, seed=42,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        )
        raw_json = response.choices[0].message.content
        parsed_result = json.loads(raw_json)
        print(f"  ▶ [{file_name}] 推推論結果: {parsed_result.get('inferred_treatment_class')} | 信心: {parsed_result.get('confidence_score')}%")
        
        output_path = f"matrices_data_bc/RESULT_{file_name.replace('.csv', '.json')}"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed_result, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ {file_name} 失敗: {e}")
