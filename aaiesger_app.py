from pathlib import Path
import streamlit as st

from aaiesger_app_shared_style import apply_base_theme, set_base_page_config


set_base_page_config()
apply_base_theme()

BASE_DIR = Path(__file__).resolve().parent

st.markdown("# 🌱 AAIESGER")
st.caption("Agentic AI ESG Report Generation System · ITRI & NTPU")

st.markdown(
    """
**兩大核心功能**
- ESG Report Generation Agent：擷取公司內部數據 + 風格範例，生成專業 ESG 段落。
- SASB Standard Mapping Agent：將輸入文本對應到 SASB 指標並輸出表格/JSON。

點擊下方按鈕進入各功能頁。
"""
)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 前往 ESG Report Generation Agent", use_container_width=True):
        st.switch_page("pages/01_esg_report_agent.py")
with col2:
    if st.button("🗺️ 前往 SASB Standard Mapping Agent", use_container_width=True):
        st.switch_page("pages/02_sasb_mapping_agent.py")

st.markdown("---")
st.subheader("關於 AAIESGER")
st.write(
    "AAIESGER 將 RAG、風格模仿與指標對照整合於單一入口，方便未來擴充。"
    "此多頁版 Streamlit 架構讓兩個 Agent 可獨立維護、共享樣式，並保留既有程式碼。"
)
