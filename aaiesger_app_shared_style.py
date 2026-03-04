"""
Shared styling helpers for AAIESGER Streamlit apps.
"""

import streamlit as st


ECO_CSS = """
<style>
.stApp{ background:linear-gradient(180deg,#F1FAF3 0%,#FFFFFF 50%,#F6FFF9 100%); color:#0F172A; }
html, body, [data-testid="stMarkdownContainer"], .stMarkdown, .stText, p, span, li{ color:#0F172A !important; }
h1,h2,h3,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3{ color:#0F5132 !important; font-weight:700; }
.eco-steps{ display:flex; gap:10px; margin:.75rem 0 1rem; flex-wrap:wrap;}
.eco-step{ display:flex; align-items:center; gap:8px; padding:8px 12px; border-radius:999px;
  border:1px solid #CFE7D7; color:#0F5132; background:#E9F7EF; font-weight:600; opacity:.9; }
.eco-step.active{ background:#DFF5E6; border-color:#9BD4AF; }
.eco-step.done{ background:#F6FFF8; border-color:#B7E5C3; }
.eco-step.locked{ background:#F3F4F6; border-color:#E5E7EB; color:#64748B; }
.eco-step .dot{ width:8px; height:8px; border-radius:50%; background:#22C55E; }
.eco-step.locked .dot{ background:#CBD5E1; }
div.stButton > button{ border-radius:10px; border:0; }
div.stButton > button:not([kind="secondary"]){
  background:linear-gradient(90deg,#22C55E,#16A34A); color:#FFFFFF; }
div.stButton > button[kind="secondary"]{
  background:#EAF7EF; color:#0F172A; border:1px solid #B7E5C3; }
div.stButton > button:disabled{
  background:#E1F5E7 !important; color:#0F5132 !important; opacity:1 !important; border:1px solid #B7E5C3 !important;
}
textarea, input, select, .stTextInput input, .stTextArea textarea{ background:#F4FBF5 !important; color:#0F172A !important; border-color:#B7E5C3 !important; }
[data-testid="stFileUploader"], [data-baseweb="radio"], [data-baseweb="select"]{ background:#F6FFF9 !important; border-radius:12px; }
[data-testid="stJson"] pre, [data-testid="stJson"]{ background:#FFFFFF !important; color:#0F172A !important; }
[data-testid="stTable"], [data-testid="stDataFrame"]{ background:#FFFFFF !important; color:#0F172A !important; }
.eco-badge{ display:inline-block; padding:2px 8px; border-radius:8px; background:#FFF7ED; color:#9A3412; border:1px solid #FED7AA; font-size:12px; }
</style>
"""


def apply_base_theme():
    """Inject shared CSS theme."""
    st.markdown(ECO_CSS, unsafe_allow_html=True)


def set_base_page_config():
    """Set a consistent page config once."""
    st.set_page_config(page_title="AAIESGER", page_icon="🌱", layout="wide")
