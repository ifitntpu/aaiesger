"""
Embedded SASB Mapping Agent page
"""
import os
import io
import json
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import psutil
import matplotlib
import matplotlib.pyplot as pl  # noqa: F401
import pandas as pd
from matplotlib import font_manager as fm
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode  # noqa: F401
from streamlit_autorefresh import st_autorefresh
from aaiesger_app_shared_style import apply_base_theme, set_base_page_config


set_base_page_config()
apply_base_theme()

# -------------------- 路徑 --------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_MAP = BASE_DIR / "sma_main_rag.py"
OUTPUT_DIR = BASE_DIR / "sma_sasb mapping output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- 中英文字型 --------------------
_CJK_CANDIDATES = [
    "Microsoft JhengHei",
    "微軟正黑體",
    "PingFang TC",
    "Noto Sans CJK TC",
    "Source Han Sans TW",
    "PMingLiU",
    "SimHei",
]

_found = None
for name in _CJK_CANDIDATES:
    try:
        fm.findfont(name, fallback_to_default=False)
        _found = name
        break
    except Exception:
        pass

if _found:
    matplotlib.rcParams["font.sans-serif"] = [_found]
    matplotlib.rcParams["font.family"] = "sans-serif"

matplotlib.rcParams["axes.unicode_minus"] = False

st.markdown("<h1>🗺️ ESG Standard Mapping Agent</h1>", unsafe_allow_html=True)
st.caption(
    "Innovative Agentic AI Technology for Autonomous ESG Report Generation developed by ITRI & NTPU. © 2025 All Rights Reserved."
)


# -------------------- Session State --------------------
def ss_init():
    s = st.session_state
    s.setdefault("step", 1)  # 1..4
    # step1
    s.setdefault("src_json", None)
    s.setdefault("src_df", None)
    # step2
    s.setdefault("model_choice", "gemma-2")
    s.setdefault("mapping_standard", "SASB")
    # step3
    s.setdefault("map_status", "idle")
    s.setdefault("map_json_path", None)
    s.setdefault("map_json_obj", None)
    s.setdefault("map_elapsed", None)
    # step4 (formatting)
    s.setdefault("fmt_df", None)
    s.setdefault("fmt_png_bytes", None)
    s.setdefault("fmt_xlsx_bytes", None)


ss_init()
SS = st.session_state


# -------------------- 工具 --------------------
def safe_json_load(txt: str):
    try:
        return json.loads(txt), None
    except Exception as e:
        return None, str(e)


def json_to_df(obj):
    try:
        if isinstance(obj, list):
            if obj and isinstance(obj[0], dict):
                return pd.DataFrame(obj)
            return pd.DataFrame({"value": obj})
        if isinstance(obj, dict):
            for k in ("rows", "data", "items", "records", "result"):
                if k in obj and isinstance(obj[k], list):
                    return pd.json_normalize(obj[k])
            return pd.json_normalize(obj)
    except Exception:
        pass
    return pd.DataFrame([{"raw": json.dumps(obj, ensure_ascii=False)}])

def start_mapping_once(cmd):
    pid = SS.get("map_pid")
    if pid and psutil.pid_exists(pid):
        st.warning("Mapping already running.")
        return

    p = subprocess.Popen(cmd)
    SS.map_pid = p.pid
    SS.map_status = "running"
    SS.map_started_at = time.time()

def latest_file(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def run_script(script: Path, *args) -> tuple[bool, str]:
    """使用系統 Python 直譯器呼叫外部腳本"""
    try:
        cmd = [sys.executable, str(script), *[str(a) for a in args]]
        res = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, check=False)
        return (res.returncode == 0), (res.stderr or "")[:2000]
    except FileNotFoundError as e:
        return False, f"File not found: {e}"


def df_to_png_bytes(df: pd.DataFrame, max_rows=40, max_cols=16, dpi=220):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import textwrap

    FONT_SIZE = 12
    HEADER_FONT_SIZE = 12
    BASE_COL_WIDTH = 0.55
    BASE_ROW_UNIT = 0.30

    df_show = df.copy()

    def wrap(x, width=18):
        if pd.isna(x):
            return ""
        s = str(x)
        return "\n".join(textwrap.wrap(s, width=width))

    for col in df_show.columns:
        df_show[col] = df_show[col].apply(wrap)

    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows)
    if df_show.shape[1] > max_cols:
        df_show = df_show.iloc[:, :max_cols]

    nrows, ncols = df_show.shape
    row_heights = []
    for r in range(nrows):
        max_lines = max(str(df_show.iloc[r, c]).count("\n") + 1 for c in range(ncols))
        row_heights.append(BASE_ROW_UNIT * max(1, max_lines))

    col_widths = [BASE_COL_WIDTH] * ncols
    fig_width = min(15, sum(col_widths) + 2)
    fig_height = min(10, sum(row_heights) + 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    tbl = mpl.table.Table(ax, bbox=[0, 0, 1, 1])
    for c, col in enumerate(df_show.columns):
        cell = tbl.add_cell(
            -1,
            c,
            width=col_widths[c],
            height=BASE_ROW_UNIT * 2,
            text=str(col),
            loc="center",
            facecolor="#EAF7EF",
            edgecolor="#DDEEE1",
        )
        cell.get_text().set_fontsize(HEADER_FONT_SIZE)

    for r in range(nrows):
        for c in range(ncols):
            cell = tbl.add_cell(
                r,
                c,
                width=col_widths[c],
                height=row_heights[r],
                text=str(df_show.iloc[r, c]),
                loc="left",
                edgecolor="#DDEEE1",
            )
            cell.PAD = 0.02
            cell.get_text().set_fontsize(FONT_SIZE)

    ax.add_table(tbl)

    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return bio.getvalue()


def df_to_xlsx_bytes(df: pd.DataFrame, sheet_name="SASB_Table") -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()


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
        f'<div class="eco-steps">{pill(1,"Source")} {pill(2,"Settings")} {pill(3,"Mapping")} {pill(4,"Formatting")}</div>',
        unsafe_allow_html=True,
    )


breadcrumb(SS.step)


# -------------------- Step 1 --------------------
def ui_step1():
    st.markdown('<div class="eco-card">', unsafe_allow_html=True)
    st.subheader("Step 1 🗂️ Source (Upload Only)")
    st.markdown('<span class="eco-badge">請上傳 JSON / JSONL 檔案</span>', unsafe_allow_html=True)

    uploaded = st.file_uploader("選擇檔案", type=["json", "jsonl"])

    SS.src_json = None
    SS.src_df = None

    if uploaded:
        try:
            raw = uploaded.read().decode("utf-8")
            INPUT_TEMP_PATH = BASE_DIR / "ui_uploaded.json"
            INPUT_TEMP_PATH.write_text(raw, encoding="utf-8")

            SS.input_file_path = str(INPUT_TEMP_PATH)

            if raw.strip().startswith("{") and "\n" in raw:
                lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
                SS.src_json = lines
            else:
                SS.src_json = json.loads(raw)

            SS.src_df = json_to_df(SS.src_json)

            st.success("檔案已讀取並解析")
            st.caption(f"資料筆數：{len(SS.src_df)}，欄位：{SS.src_df.shape[1]}")

            tabs = st.tabs(["JSON 檢視", "表格檢視"])
            with tabs[0]:
                st.json(SS.src_json)
            with tabs[1]:
                st.dataframe(SS.src_df, height=320)

        except Exception as e:
            st.error(f"解析失敗：{e}")
            SS.src_json = None
            SS.src_df = None

    c1, c2 = st.columns(2)
    with c1:
        if st.button("清除來源", use_container_width=True):
            st.rerun()

    with c2:
        next_disabled = SS.src_json is None
        if st.button("Next ➡️ Settings", disabled=next_disabled, use_container_width=True):
            SS.step = 2
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


INPUT_FROM_UI = BASE_DIR / "ui_input.json"


# -------------------- Step 2 --------------------
def ui_step2():
    st.subheader("Step 2 ⚙️ Settings")
    c1, c2 = st.columns(2)
    with c1:
        SS.model_choice = st.selectbox("AI 模型", ["gemma-2"], index=0)
    with c2:
        SS.mapping_standard = st.selectbox("Mapping 標準", ["SASB"], index=0)
    st.caption(f"使用模型：*{SS.model_choice}*；標準：*{SS.mapping_standard}*")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back to Source", use_container_width=True):
            SS.step = 1
            st.rerun()
    with c2:
        if st.button("Next ➡️ Mapping", disabled=(SS.src_json is None), use_container_width=True):
            SS.step = 3
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- Step 3 --------------------
def ui_step3():
    st.markdown('<div class="eco-card">', unsafe_allow_html=True)
    st.subheader("Step 3 🧭 Mapping")
    st.write("將 Step1 的 JSON 丟入 `sma_main_rag.py`，並顯示最新輸出的 JSON。")

    run_disabled = SS.map_status == "running"

    if st.button("▶️ Start mapping", disabled=run_disabled, use_container_width=True):
        input_path = SS.get("input_file_path", None)

        if input_path is None:
            st.error("缺少輸入檔，請回到 Step 1 上傳 JSON。")
        else:
            cmd = [
                sys.executable,
                str(SCRIPT_MAP),
                input_path,
            ]

            started = start_mapping_once(cmd)
            if started:
                st.rerun()

    if SS.map_status == "running":
        pid = SS.get("map_pid")

        # pid 可能還沒存到/被清掉
        alive = bool(pid) and psutil.pid_exists(pid)

        st.info(
            f"Mapping running… PID={pid} "
            f"elapsed={int(time.time() - SS.map_started_at)}s"
        )

        if alive:
            time.sleep(2)   # ⏳ 等 2 秒
            st.rerun()      # 🔁 自動刷新（繼續輪詢）
        else:
            # ✅ 程式跑完了 -> 去抓最新 output
            latest = latest_file(OUTPUT_DIR, "*.json")
            if latest:
                SS.map_json_path = str(latest)
                SS.map_json_obj = json.loads(Path(latest).read_text(encoding="utf-8"))
                SS.output_basename = Path(latest).stem
                SS.map_status = "done"
            else:
                SS.map_status = "error"
                st.error("Mapping finished but no output JSON found.")

            st.rerun()


    if SS.map_status == "done" and SS.map_json_obj is not None:
        st.success(f"完成！最新輸出：**{Path(SS.map_json_path).name}**（耗時 {SS.map_elapsed}s）")
        st.json(SS.map_json_obj)
        st.download_button(
            "下載 JSON",
            data=io.BytesIO(json.dumps(SS.map_json_obj, ensure_ascii=False, indent=2).encode("utf-8")),
            file_name=Path(SS.map_json_path).name,
            mime="application/json",
            use_container_width=True,
        )
    elif SS.map_status == "error":
        st.error("Mapping 失敗，請檢查輸入後再試。")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back to Settings", use_container_width=True):
            SS.step = 2
            st.rerun()
    with c2:
        next_disabled = not (SS.map_status == "done" and SS.map_json_obj is not None)
        if st.button("Next ➡️ Formatting", disabled=next_disabled, use_container_width=True):
            SS.fmt_df = json_to_df(SS.map_json_obj)
            SS.step = 4
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- Step 4 --------------------
def ui_step4():
    st.markdown('<div class="eco-card">', unsafe_allow_html=True)
    st.subheader("Step 4 🧾 Formatting")
    st.write("Mapping 已完成，可在下方表格中直接編輯／新增／刪除欄位，然後下載 Excel / PNG。")

    if SS.fmt_df is None:
        st.info("尚無資料，請先完成 Mapping 才能進入 Formatting。")
    else:
        t1, t2 = st.columns(2)
        with t1:
            if st.button("⬇️ 下載 Excel", use_container_width=True):
                SS.fmt_xlsx_bytes = df_to_xlsx_bytes(SS.fmt_df)
                base = SS.get("output_basename", "mapping_output")
                xlsx_name = f"{base}.xlsx"
                (OUTPUT_DIR / xlsx_name).write_bytes(SS.fmt_xlsx_bytes)
                st.download_button(
                    "點此下載 Excel",
                    data=SS.fmt_xlsx_bytes,
                    file_name=xlsx_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_xlsx",
                    use_container_width=True,
                )
        with t2:
            if st.button("🖼️ 下載 PNG", use_container_width=True):
                SS.fmt_png_bytes = df_to_png_bytes(SS.fmt_df)
                base = SS.get("output_basename", "mapping_output")
                png_name = f"{base}.png"
                (OUTPUT_DIR / png_name).write_bytes(SS.fmt_png_bytes)

                st.download_button(
                    "點此下載 PNG",
                    data=SS.fmt_png_bytes,
                    file_name=png_name,
                    mime="image/png",
                    key="dl_png",
                    use_container_width=True,
                )

        st.markdown("#### 表格檢視（可編輯）")

        edited_df = st.data_editor(
            SS.fmt_df,
            num_rows="dynamic",
            height=420,
            use_container_width=True,
            key="fmt_editor",
        )
        SS.fmt_df = edited_df

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back to Mapping", use_container_width=True):
            SS.step = 3
            st.rerun()

    with c2:
        if st.button("Finish 🔄 回到 Step 1", use_container_width=True):
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
