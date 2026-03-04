"""
ESG Report Generation Agent page (4-step wizard).
"""

import os
import time
from pathlib import Path
from typing import Dict

import streamlit as st

from aaiesger_app_shared_style import apply_base_theme, set_base_page_config

# Ensure imports resolve within the aaiesger folder
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in os.sys.path:
    os.sys.path.append(str(ROOT))

from rga_main_rag import (  # type: ignore  # noqa: E402
    DATA_DB_DIR,
    STYLE_DB_DIR,
    build_chain,
    generate_paragraph,
    load_data_vs,
    load_style_vs,
)
from data_prep import build_internal_db, build_style_db, csv_to_json_and_jsonl  # type: ignore  # noqa: E402


set_base_page_config()
apply_base_theme()

st.markdown("<h1>ESG Report Generation Agent</h1>", unsafe_allow_html=True)
st.caption("Follow the steps to rebuild vector DBs and generate ESG paragraphs.")


# -------------------- Session State --------------------
def ss_init():
    s = st.session_state
    s.setdefault("step", 1)  # 1..4
    s.setdefault("generated", None)
    s.setdefault("edited", None)
    s.setdefault("context_info", {})
    s.setdefault("model_name", "google/gemma-2-9b-it")
    s.setdefault("provider_id", "hf")


ss_init()
SS = st.session_state


# -------------------- Breadcrumb --------------------
def breadcrumb(step: int):
    def pill(idx, label):
        cls = "eco-step"
        if idx < step:
            cls += " done"
        elif idx == step:
            cls += " active"
        else:
            cls += " locked"
        return f'<div class="{cls}"><div class="dot"></div><div>{idx}. {label}</div></div>'

    st.markdown(
        f'<div class="eco-steps">{pill(1,"Source")} {pill(2,"Settings")} {pill(3,"Generation")} {pill(4,"Review & Export")}</div>',
        unsafe_allow_html=True,
    )


breadcrumb(SS.step)

@st.cache_resource(show_spinner=False)
def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

@st.cache_resource(show_spinner=False)
def _load_vs():
    return load_style_vs(), load_data_vs()

def _set_rebuild_state(state: str, step: str = "", detail: str = ""):
    st.session_state["rebuild_state"] = state
    st.session_state["rebuild_step"] = step
    st.session_state["rebuild_detail"] = detail


def _rebuild_db_from_upload(uploaded, kind: str):
    if st.session_state.get("rebuild_state") == "rebuilding":
        st.warning("Rebuild is already running. Please wait...")
        return

    if uploaded is None:
        st.warning("Please upload a file first.")
        return

    tmp_path = ROOT / f"_upload_{int(time.time())}_{uploaded.name}"
    tmp_path.write_bytes(uploaded.read())

    progress = st.progress(0)
    # 這個 status 會顯示在畫面上，並且可以動態更新 label
    status = st.status("Preparing...", expanded=True)

    try:
        _set_rebuild_state("rebuilding", step="prepare", detail=str(tmp_path))
        status.update(label="Preparing input...", state="running")
        progress.progress(5)

        if kind == "csv":
            _set_rebuild_state("rebuilding", step="convert_csv", detail="csv_to_json_and_jsonl")
            status.update(label="Converting CSV → JSON/JSONL...", state="running")
            csv_to_json_and_jsonl(str(tmp_path))
            progress.progress(25)

            _set_rebuild_state("rebuilding", step="build_internal_db", detail="embedding + writing Chroma")
            status.update(label="Building Internal DB (embedding + writing Chroma)...", state="running")
            build_internal_db("esg_format1_structured.json")
            progress.progress(95)

            _set_rebuild_state("done")
            status.update(label="Internal DB rebuilt ✅", state="complete")
            progress.progress(100)
            st.success("Internal DB rebuilt.")

        elif kind == "style":
            _set_rebuild_state("rebuilding", step="build_style_db", detail="embedding + writing Chroma")
            status.update(label="Building Style DB (embedding + writing Chroma)...", state="running")
            progress.progress(25)

            build_style_db(str(tmp_path))
            progress.progress(95)

            _set_rebuild_state("done")
            status.update(label="Style DB rebuilt ✅", state="complete")
            progress.progress(100)
            st.success("Style DB rebuilt.")

        else:
            _set_rebuild_state("error", detail="Unknown rebuild kind.")
            status.update(label="Unknown rebuild kind.", state="error")
            st.error("Unknown rebuild kind.")

    except Exception as e:
        _set_rebuild_state("error", detail=repr(e))
        status.update(label="Rebuild failed ❌", state="error")
        st.exception(e)
    finally:
        # 讓 rerun 後狀態還留著（done/error/idle 都可以保留）
        pass

def render_rebuild_status():
    state = st.session_state.get("rebuild_state", "idle")
    step = st.session_state.get("rebuild_step", "")
    detail = st.session_state.get("rebuild_detail", "")

    if state == "idle":
        st.caption("Status: idle")
    elif state == "rebuilding":
        st.info(f"Status: rebuilding — {step}\n\n{detail}")
    elif state == "done":
        st.success("Status: done ✅")
    elif state == "error":
        st.error(f"Status: error ❌ — {detail}")


# -------------------- Step 1: Source --------------------
def ui_step1():
    st.markdown('<div class="eco-card">', unsafe_allow_html=True)
    st.subheader("Step 1 Source")
    st.caption("Upload data to rebuild vector DBs (optional). If already built, click Next.")

    c1, c2 = st.columns(2)
    with c1:
        up_csv = st.file_uploader(
            "Upload company CSV (auto JSON/JSONL + Internal DB build)", type=["csv"], key="csv_up"
        )
        if st.button("Rebuild Internal DB", use_container_width=True):
            _rebuild_db_from_upload(up_csv, "csv")
    with c2:
        up_style = st.file_uploader("Upload Style JSON/JSONL (build Style DB)", type=["json", "jsonl"], key="style_up")
        if st.button("Rebuild Style DB", use_container_width=True):
            _rebuild_db_from_upload(up_style, "style")
    render_rebuild_status()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear cache", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    with c2:
        if st.button("Next → Settings", use_container_width=True):
            SS.step = 2
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- Step 2: Settings --------------------
def ui_step2():
    st.markdown('<div class="eco-card">', unsafe_allow_html=True)
    st.subheader("Step 2 Settings")
    st.caption("Choose a local HF model (no API key required).")

    choices = ["google/gemma-2-9b-it", "microsoft/Phi-3.5-mini-instruct"]
    default_idx = choices.index(SS.model_name) if SS.model_name in choices else 0
    SS.model_name = st.selectbox("HF Local Model", choices, index=default_idx)
    SS.provider_id = "hf"

    st.markdown("---")
    st.caption("Vector stores (local paths):")
    st.code(f"Style DB: {STYLE_DB_DIR}\nData  DB: {DATA_DB_DIR}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to Source", use_container_width=True):
            SS.step = 1
            st.rerun()
    with c2:
        if st.button("Next → Generation", use_container_width=True):
            SS.step = 3
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- Step 3: Generation --------------------
def ui_step3():
    st.markdown('<div class="eco-card">', unsafe_allow_html=True)
    st.subheader("Step 3 Generation")
    st.caption("Enter company, year, topic, and indicator, then generate.")

    col1, col2 = st.columns(2)
    with col1:
        company = st.text_input("Company", value="ASE")
        year = st.number_input("Year", value=2024, step=1, format="%d")
    with col2:
        topic = st.text_input("Topic", value="Cleanroom energy management")
        indicator = st.text_input("Indicator", value="Scope 2 emissions reduction in fabs")

    extra = st.text_area("Additional guidance (optional)", value="", height=80)

    if st.button("Generate ESG Paragraph", type="primary", use_container_width=True):
        style_vs, data_vs = _load_vs()
        chain = build_chain(provider=SS.provider_id, model_name=SS.model_name, temperature=0.3)

        with st.spinner("Generating..."):
            out: Dict[str, object] = generate_paragraph(
                chain,
                style_vs,
                data_vs,
                company,
                int(year),
                topic + (" " + extra if extra else ""),
                indicator,
            )

        SS.generated = out.get("paragraph") or "(no output)"
        SS.edited = SS.generated
        SS.context_info = {
            "style_refs": out.get("style_refs"),
            "context_hits": out.get("context_hits"),
            "raw_context": (out.get("raw_context") or [])[:3],
        }
        SS.meta = {"company": company, "year": year, "topic": topic}
        SS.step = 4
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back to Settings", use_container_width=True):
            SS.step = 2
            st.rerun()
    with c2:
        next_disabled = SS.generated is None
        if st.button("Next → Review", disabled=next_disabled, use_container_width=True):
            SS.step = 4
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- Step 4: Review & Export --------------------
def ui_step4():
    st.markdown('<div class="eco-card">', unsafe_allow_html=True)
    st.subheader("Step 4 Review & Export")

    if SS.generated is None:
        st.info("No content yet. Please return to Step 3 to generate.")
    else:
        st.markdown("#### Output")
        st.write(SS.generated)

        st.markdown("#### Editable Output")
        SS.edited = st.text_area(
            "ESG paragraph (editable)",
            value=SS.edited or SS.generated,
            height=320,
            key="editable_output",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirm output", use_container_width=True):
                st.success("Output confirmed.")
        with c2:
            meta = SS.get("meta", {}) or {}
            fname = f"{meta.get('company','ESG')}_{meta.get('year','YYYY')}_{meta.get('topic','topic')}.txt"
            st.download_button(
                label="Download .txt",
                data=str(SS.edited or ""),
                file_name=fname,
                mime="text/plain",
                use_container_width=True,
            )

        with st.expander("Context details"):
            st.write("Style refs (company, year):", SS.context_info.get("style_refs"))
            st.write("Context hits:", SS.context_info.get("context_hits"))
            st.write("Raw context (first 3):", SS.context_info.get("raw_context"))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back to Generation", use_container_width=True):
            SS.step = 3
            st.rerun()
    with c2:
        if st.button("Finish → Back to Step 1", use_container_width=True):
            SS.step = 1
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- Router --------------------
if SS.step == 1:
    ui_step1()
elif SS.step == 2:
    ui_step2()
elif SS.step == 3:
    ui_step3()
else:
    ui_step4()
