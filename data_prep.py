
# data_prep.py (enhanced)
# - Robust CSV ingestion
# - If no text column, compose one from columns like 類型/議題/指標/數據/資料邊界/確信機構/確信標準/確信範圍
# - Build two Chroma vector stores (internal/style)
import os, re, json
from typing import List, Dict
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st



# Use local HF embeddings to avoid API keys (align with rga_main_rag defaults)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Use HF-specific vector store folders to avoid dimension mismatch with old OpenAI-based stores.
DATA_DB_DIR  = "chroma_internal_data"
DATA_COLL    = "internal_corpus"
STYLE_DB_DIR = "chroma_esg_style"
STYLE_COLL   = "style_corpus"

@st.cache_resource(show_spinner=False)
def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

def _read_csv_auto(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(csv_path, encoding="cp950")

def csv_to_json_and_jsonl(
    csv_path: str,
    out_structured_json: str = "esg_format1_structured.json",
    out_text_chunks_jsonl: str = "esg_format2_text_chunks.jsonl",
    out_meta_options_json: str = "esg_metadata_options.json",
) -> None:
    df = _read_csv_auto(csv_path)
    df.reset_index(drop=True, inplace=True)
    df.columns = [str(c).strip() for c in df.columns]

    # normalize common columns
    rename_map = {
        "類型": "type",
        "議題": "topic",
        "指標": "value",        # 在你的 CSV，這欄多為數值
        "數據": "data_note",     # 常見會是說明/邊界
        "資料邊界": "scope",
        "確信機構": "assurer_org",
        "確信標準": "assurer_std",
        "確信範圍": "assurer_scope",
        "年份": "year",
        "公司": "company",
        # possible text columns
        "資料": "text",
        "內容": "text",
        "內文": "text",
        "揭露文字": "text",
        "揭露內容": "text",
        "描述": "text",
        "敘述": "text",
        "段落": "text",
        "paragraph": "text",
        "content": "text",
        "text": "text",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # infer company/year from filename like "2303 - 聯電_個別公司查詢_2024.csv"
    fname = os.path.basename(csv_path)
    m = re.search(r"(?:\d{4}\s-\s)?(?P<company>[^_]+)_.+?(?P<year>\d{4})\.csv$", fname)
    inferred_company = m.group("company") if m else None
    inferred_year = int(m.group("year")) if (m and m.group("year").isdigit()) else None

    # compose text when missing
    if "text" not in df.columns:
        def compose_text(row):
            ttype   = str(row.get("type", "") or "").strip()
            topic   = str(row.get("topic", "") or "").strip()
            value   = str(row.get("value", "") or "").strip()
            scope   = str(row.get("scope", "") or "").strip()
            data_nt = str(row.get("data_note", "") or "").strip()
            a_org   = str(row.get("assurer_org", "") or "").strip()
            a_std   = str(row.get("assurer_std", "") or "").strip()
            a_scp   = str(row.get("assurer_scope", "") or "").strip()
            comp = str(row.get("company") or inferred_company or "").strip()
            year = str(row.get("year") or (inferred_year if inferred_year is not None else "")).strip()

            head = comp if comp else "公司"
            if year: head = f"{head}（{year} 年）"
            parts = []
            if ttype or topic or value:
                parts.append(f"{head}於「{ttype}」主題之『{topic}』指標值為「{value}」。")
            if scope or data_nt:
                note_bits = []
                if scope:  note_bits.append(f"資料邊界：{scope}")
                if data_nt:note_bits.append(f"補充說明：{data_nt}")
                if note_bits: parts.append("；".join(note_bits) + "。")
            assure_bits = []
            if a_org: assure_bits.append(f"確信機構：{a_org}")
            if a_std: assure_bits.append(f"確信標準：{a_std}")
            if a_scp: assure_bits.append(f"確信範圍：{a_scp}")
            if assure_bits:
                parts.append(" ".join(assure_bits) + "。")
            return "".join(parts).strip()
        df["text"] = df.apply(compose_text, axis=1)

    # fill company/year if still missing
    if "company" not in df.columns and inferred_company:
        df["company"] = inferred_company
    if "year" not in df.columns and inferred_year is not None:
        df["year"] = inferred_year

    base_cols = ["type","topic","value","text","company","year","scope","assurer_org","assurer_std","assurer_scope"]
    keep_cols = [c for c in base_cols if c in df.columns]
    df = df[keep_cols].dropna(subset=["text"])

    # outputs
    records = df.to_dict(orient="records")
    with open(out_structured_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    with open(out_text_chunks_jsonl, "w", encoding="utf-8") as f_out:
        for r in records:
            paragraph = (r.get("text") or "").strip()
            if paragraph:
                f_out.write(json.dumps({"text": paragraph}, ensure_ascii=False) + "\n")

    meta_dict: Dict[str, List[str]] = {}
    for field in ["type","topic","company","year"]:
        if field in df.columns:
            meta_dict[field] = sorted({str(v) for v in df[field].dropna().tolist()})
    with open(out_meta_options_json, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

def build_internal_db(json_path: str, persist_dir: str = DATA_DB_DIR, collection: str = DATA_COLL):
    emb = get_embedder()
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    docs: List[Document] = []
    for r in rows:
        text = r.get("text") or ""
        if not text.strip():
            continue
        meta = {k: r.get(k) for k in ["company","year","type","topic"] if k in r}
        docs.append(Document(page_content=text, metadata=meta))
    if not docs:
        raise ValueError("No documents built from JSON. Check your input.")
    vs = Chroma(collection_name=collection, persist_directory=persist_dir, embedding_function=emb)
    try:
        raw = vs.get()
        if raw.get("ids"):
            vs._collection.delete(raw.get("ids"))
    except Exception:
        pass
    vs.add_documents(docs)
    vs.persist()
    return vs

def build_style_db(style_json: str, persist_dir: str = STYLE_DB_DIR, collection: str = STYLE_COLL):
    import json
    from json import JSONDecodeError
    emb = get_embedder()

    # 先試一般 JSON（列表/字典），失敗就當作 JSONL 逐行讀
    rows = None
    try:
        with open(style_json, "r", encoding="utf-8") as f:
            rows = json.load(f)
            # 若是單一 dict，就包成 list
            if isinstance(rows, dict):
                rows = [rows]
    except JSONDecodeError:
        rows = []
        with open(style_json, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except JSONDecodeError:
                    # 跳過不合法的行，或可以改成 raise 讓你檢查
                    continue

    if not rows:
        raise ValueError(f"No style rows parsed from {style_json}. Check the file format (JSON/JSONL).")

    docs: List[Document] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        text = r.get("paragraph") or r.get("text") or ""
        if not str(text).strip():
            continue
        meta = {
            "source_company": r.get("source_company"),
            "source_year": r.get("source_year"),
            "topic": r.get("topic"),
            "indicator": r.get("indicator"),
        }
        docs.append(Document(page_content=str(text).strip(), metadata=meta))

    if not docs:
        raise ValueError("No documents built from style JSON/JSONL. Check 'paragraph' or 'text' fields.")

    vs = Chroma(collection_name=collection, persist_directory=persist_dir, embedding_function=emb)
    try:
        raw = vs.get()
        if raw.get("ids"):
            vs._collection.delete(raw.get("ids"))
    except Exception:
        pass
    vs.add_documents(docs)
    vs.persist()
    return vs
