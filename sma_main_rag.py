# sma_main_rag.py

import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from huggingface_hub import login
import re 
import sys
from pathlib import Path
from datetime import datetime
import streamlit as st

# 1. Load embedding model and Chroma vector DB
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_sasb_metadata"
COLLECTION_NAME = "sasb_chunks"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectordb = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model
)

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.6, "filter": {"industry": "Semiconductors (TC-SC)"}}
)

# 2. Load LLM 
try:
    hf_token = st.secrets["HUGGINGFACE_TOKEN"]
    login(token=hf_token)
except FileNotFoundError:
    st.error("尚未設定 Secrets，請檢查 .streamlit/secrets.toml")
    st.stop()
except KeyError:
    st.error("Secrets 中找不到 HUGGINGFACE_TOKEN")
    st.stop()
LLM_MODEL = "google/gemma-2-9b-it" 
            # "microsoft/Phi-3.5-mini-instruct"
            # "google/gemma-2-9b-it"  
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)  
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    local_files_only=False,  # 首次下載設定 False
    trust_remote_code=True,
    device_map="auto",
)
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=False,
    return_full_text=False
)
llm = HuggingFacePipeline(pipeline=gen)

# 如果你要「不下載、走 HF 雲端」，改用下列三行，其他不變：
'''
from langchain_huggingface import HuggingFaceEndpoint
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxx"
llm = HuggingFaceEndpoint(repo_id=LLM_MODEL, max_new_tokens=256, temperature=0.0, return_full_text=False)
'''

# 3. Prompt 
doc_prompt = PromptTemplate.from_template(
    "【SASB 準則候選卡片】\n"
    "Code: {code}\n"
    "Label: {label}\n"
    "Topic: {topic}\n"
    "Source: {doc} p.{page_start}-{page_end}\n"
    "Excerpt：\n{page_content}"
)

schema_example = (
    '{\n'
    '  "rows": [\n'
    '    {\n'
    '      "指標編號": "<SASB code 或條次>",\n'
    '      "揭露主題": "<topic>",\n'
    '      "指標描述": "<以己語重述的 metric 摘要>",\n'
    '      "性質": "<量化 或 質化>",\n'
    '      "報告內容說明": "<3–6 句專業說明 input data，作為永續報告書的內容>",\n'
    '      "對應報告書章節": "",\n'
    '      "頁碼": "",\n'
    '      "source": {"doc": "<文件名或ID>", "page_start": "", "page_end": ""}\n'
    '    }\n'
    '  ]\n'
    '}'
)
empty_rows = '{"rows": []}'

system_prompt = (
    "你是 ESG 助理。請嚴格依據『SASB 準則候選卡片』撰寫輸出，語氣正式、繁體中文。\n"
    "規則：\n"
    "1) 只能輸出 **一個 JSON 物件**（鍵為 rows，值為列陣列）。不得輸出任何解說文字、前後綴或 Markdown。\n"
    "2) **指標描述** 只可引用『SASB 代碼/條次』，不得貼長段原文，嚴禁逐字複製候選卡片或使用者原文。\n"
    "3) **報告內容說明** 可以使用 input 的數值和文字，以永續報告書口吻撰寫，切記忠實表達永續狀況。\n"
    "4) **性質** 請根據 SASB 準則的要求和報告內容說明，明確指出指標的性質屬性，例如「量化」或「質化」。\n"
    "5) 章節與頁碼目前可能未知，允許為空字串 \"\"。切勿猜測或杜撰。\n"
    "6) 若證據不足、無相關數據，則不要輸出相關的內容，也不要預留後續填入數值的地方，切勿臆測。\n"
    "7) 每個欄位只能出現一次，嚴禁重複鍵或多餘括號。\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user",
     "使用者問題：\n{input}\n\n"
     "請依候選卡片輸出**唯一 JSON 物件**（僅此物件，不要任何多餘文字）。\n"
     "格式如下（欄位示意，實際請填入內容或空字串）：\n"
     "{schema_example}\n"
     "若無相關，回傳 {empty_rows}\n"
)
]).partial(schema_example=schema_example, empty_rows=empty_rows)

# 4. Build RAG chain
combine_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_prompt=doc_prompt,
    document_variable_name="context",   
)

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_chain,
)

# 5. Invoke RAG chain and store results in JSON

def remove_duplicate_keys_object_pairs_hook(pairs):
    result = {}
    for k, v in pairs:
        if k not in result:
            result[k] = v
    return result

def batch_rag(input_jsonl_path, output_json_path):
    all_rows = []
    with open(input_jsonl_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)   
        for item in data:
            input_text = item.get("text", "")
            result = rag_chain.invoke({"input": "請查找與此相關的 SASB 指標：" + input_text})
            answer_str = result["answer"]
            
            try:
                parsed = json.loads(answer_str)
                rows = parsed.get("rows", [])
                all_rows.extend(rows)
            except json.JSONDecodeError:
                print("JSON 解析失敗，跳過此行")
                # print(answer_str)               
    with open(output_json_path, "w", encoding="utf-8") as fout:
        json.dump(all_rows, fout, ensure_ascii=False, indent=2)
    print(f"File saved as：{output_json_path}")


DEFAULT_OUTPUT_DIR = BASE_DIR / "sma_sasb mapping output"


def main():
    args = sys.argv[1:]

    # --- 讀取 input path ---
    if len(args) >= 1:
        input_path = Path(args[0])
    else:
        print("ERROR: No input file provided from UI.")
        sys.exit(1)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # --- 自動產生 output 檔名 ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    auto_output_name = f"mapping_output_{timestamp}.json"
    output_path = DEFAULT_OUTPUT_DIR / auto_output_name

    # 建立資料夾
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input : {input_path}")
    print(f"[INFO] Output: {output_path}")

    # --- 主要邏輯 ---
    batch_rag(
        input_jsonl_path=str(input_path),
        output_json_path=str(output_path),
    )

    print("[INFO] Mapping finished.")


if __name__ == "__main__":
    main()

