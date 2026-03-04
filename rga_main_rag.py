from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import login

# from langchain.chains import LLMChain
try:
    from langchain.chains import LLMChain  # 舊版 / 某些版本 OK
except ImportError:
    # 新版沒有 LLMChain 的情況：自己做一個相容的簡單版本
    from dataclasses import dataclass
    from typing import Any, Dict

    from langchain_core.prompts import ChatPromptTemplate  # 你本來就會用
    from langchain_core.runnables import Runnable  # 型別輔助，可有可無

    @dataclass
    class LLMChain:  # type: ignore
        llm: Any
        prompt: ChatPromptTemplate
        verbose: bool = False

        def run(self, variables: Dict[str, Any]) -> str:
            """
            仿造舊版 LLMChain.run 的介面：
            用 LCEL 的 prompt | llm 來實作。
            """
            chain = self.prompt | self.llm
            result = chain.invoke(variables)
            # 對於 Chat 型 LLM，invoke 通常回傳 str 或 BaseMessage
            if hasattr(result, "content"):
                return result.content
            return str(result)

try:
    hf_token = st.secrets["HUGGINGFACE_TOKEN"]
    login(token=hf_token)
except FileNotFoundError:
    st.error("尚未設定 Secrets，請檢查 .streamlit/secrets.toml")
    st.stop()
except KeyError:
    st.error("Secrets 中找不到 HUGGINGFACE_TOKEN")
    st.stop()

BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_EMBED_MODEL = os.getenv("AIESG_HF_EMBED_MODEL", EMBEDDING_MODEL)

# Default to local HF models (same setup as sma_main_rag) to avoid API keys.
HF_DEFAULT_GEN_MODEL = os.getenv("AIESG_HF_LLM_MODEL", "google/gemma-2-9b-it")
OPENAI_DEFAULT_GEN_MODEL = "gpt-4o-mini"
DEFAULT_PROVIDER = os.getenv("AIESG_LLM_PROVIDER", "hf")

STYLE_DB_DIR = str(BASE_DIR / "chroma_esg_style")
STYLE_COLL = "style_corpus"
DATA_DB_DIR = str(BASE_DIR / "chroma_internal_data")
DATA_COLL = "internal_corpus"

STYLE_GUIDE = dedent("""
請生成一段 ESG 報告文字
要求：
- 內容必須根據提供的公司的 ESG 資料，不要自行杜撰。
- 採用正式、專業的語氣，避免口語化。
- 盡可能將數據與管理措施、策略或目標結合。
- 文字長度建議為 3–6 句。
- 使用繁體中文撰寫。"""
).strip()

SYSTEM = (
    "你是一位專業的 ESG 報告撰寫助理，熟悉 SASB 準則與永續報導格式，"
    "請根據使用者輸入與檢索到的段落，撰寫一段語氣正式、邏輯清楚、合規的報告內容草稿。"
    )

DRAFT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        (
            "human",
            dedent("""
                你會取得兩種檢索內容：
                1) Style Examples：和 [議題/指標] 相關的優良範例段落（可用於參考語氣/組織/修辭，但不可捏造新數字）；
                2) Data Context：公司實際揭露的文字片段（只能使用這些內容中的真實資訊）。
                   
                [Style Guide: {style_guide}]
                [Style Examples (僅作語氣和結構參考，不可捏造新數據): {style_examples}]
                [Data Context (真實資料): {data_context}]

                【請產生】
                - 一段具ESG專業報告風格、條理清晰的說明段落，並於使用者要求時markdown格式製作表格，切勿捏造或推論未揭露的數據。
                - 符合 Style Guide 與 Style Examples 的段落
                - 請用繁體中文回答
                """
            ).strip(),
        ),
    ]
)


def _maybe_login_hf() -> None:
    """HF login：可選，無 token 或失敗就略過，不要讓 import/載入向量庫時爆掉。"""
    try:
        # Streamlit page 也可以先把 token 設到環境變數，這裡就能讀到
        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            return
        login(token=token)
    except Exception:
        # 不要因為 login 失敗就卡死（尤其你只是在 load embeddings / chroma）
        return

def get_embeddings(provider: Optional[str] = None, model: Optional[str] = None):
    chosen = (provider or os.getenv("AIESG_EMBED_PROVIDER") or "hf").lower()
    if chosen in ("hf", "huggingface"):
        _maybe_login_hf()
        model_name = model or HF_EMBED_MODEL
        return HuggingFaceEmbeddings(model_name=model_name)
    return OpenAIEmbeddings(model=model or EMBEDDING_MODEL)


def load_style_vs(
    persist_directory: str = STYLE_DB_DIR,
    collection_name: str = STYLE_COLL,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
):
    emb = get_embeddings(provider=embedding_provider, model=embedding_model)
    return Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=emb)


def load_data_vs(
    persist_directory: str = DATA_DB_DIR,
    collection_name: str = DATA_COLL,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
):
    emb = get_embeddings(provider=embedding_provider, model=embedding_model)
    return Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=emb)


def retrieve_style_examples(style_vs: Chroma, topic: str, indicator: str, k: int = 2) -> List[Document]:
    q = f"{topic} {indicator}".strip()
    base = style_vs.similarity_search(q, k=8)
    seen, docs = set(), []
    for d in base:
        key = (d.page_content or "").strip()
        if key and key not in seen:
            seen.add(key)
            docs.append(d)
        if len(docs) >= k:
            break
    return docs


def retrieve_data_context(data_vs: Chroma, company: str, year: int, topic: str, indicator: str, k: int = 6) -> List[Document]:
    q = f"{company} {year} {topic} {indicator}".strip()
    return data_vs.similarity_search(q, k=k)


def _guard_strip_braces(text: str) -> str:
    import re

    return re.sub(r"\{\{.*?\}\}", "", text)

def _build_hf_llm(model_name: str, temperature: float):
    # 需要時才 import，避免 module import 階段就炸
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
    except ImportError as e:
        raise ImportError(
            "缺少 transformers 套件，請在你的環境安裝：\n"
            "  pip install transformers accelerate\n"
        ) from e

    # 需要時才登入（若有 token）
    _maybe_login_hf()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": bool(temperature and temperature > 0),
        "return_full_text": False,
    }
    if temperature is not None:
        gen_kwargs["temperature"] = max(float(temperature), 1e-3) if temperature > 0 else 0.0

    text_gen = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **gen_kwargs,
    )

    from langchain_huggingface import HuggingFacePipeline
    return HuggingFacePipeline(pipeline=text_gen)



def build_chain(
    provider: str = DEFAULT_PROVIDER,
    model_name: Optional[str] = None,
    temperature: float = 0.3,
) -> LLMChain:
    provider = (provider or DEFAULT_PROVIDER or "openai").lower()
    if provider == "openai":
        llm = ChatOpenAI(model=model_name or OPENAI_DEFAULT_GEN_MODEL, temperature=temperature)
    elif provider in ("hf", "huggingface", "huggingface-local"):
        llm = _build_hf_llm(model_name or HF_DEFAULT_GEN_MODEL, temperature=temperature)
    else:
        raise ValueError(f"未知的 provider: {provider}")
    return LLMChain(llm=llm, prompt=DRAFT_PROMPT, verbose=False)


def generate_paragraph(
    chain: LLMChain,
    style_vs: Chroma,
    data_vs: Chroma,
    company: str,
    year: int,
    topic: str,
    indicator: str,
    style_k: int = 2,
    data_k: int = 6,
) -> Dict[str, Any]:
    style_docs = retrieve_style_examples(style_vs, topic, indicator, k=style_k)
    ctx_docs = retrieve_data_context(data_vs, company, year, topic, indicator, k=data_k)
    style_examples = "\n\n".join([d.page_content.strip() for d in style_docs if (d.page_content or "").strip()])
    data_context = "\n\n".join([d.page_content.strip() for d in ctx_docs if (d.page_content or "").strip()])
    variables = {
        "style_guide": STYLE_GUIDE,
        "style_examples": style_examples or ("(無)" if not style_docs else ""),
        "data_context": data_context or ("(無)" if not ctx_docs else ""),
    }
    draft_text = chain.run(variables).strip()
    draft_text = _guard_strip_braces(draft_text)
    return {
        "paragraph": draft_text,
        "style_refs": [(d.metadata.get("source_company"), d.metadata.get("source_year")) for d in style_docs],
        "context_hits": len(ctx_docs),
        "raw_context": [d.page_content for d in ctx_docs],
    }
