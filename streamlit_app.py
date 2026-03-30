import asyncio
import json
import sqlite3
import time
from pathlib import Path

import inngest
import requests
import streamlit as st
from modelselector import discover_ollama_models, split_ollama_models


HISTORY_JSON_PATH = Path("streamlit_history.json")
HISTORY_DB_PATH = Path("history.sqlite3")


st.set_page_config(page_title="Document RAG", page_icon="📄", layout="wide")


def apply_brutalist_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #111111;
                --fg: #e6e6e6;
                --line: #7a7a7a;
                --panel: #1a1a1a;
                --ink: #ececec;
                --btn: #6a6a6a;
                --btn-hover: #838383;
            }

            html, body, .stApp {
                background: var(--bg) !important;
                color: var(--fg) !important;
            }

            [data-testid="stAppViewContainer"],
            [data-testid="stHeader"],
            [data-testid="stToolbar"] {
                background: var(--bg) !important;
                color: var(--fg) !important;
            }

            [data-testid="stAppViewContainer"] {
                overflow: hidden !important;
            }

            /* Force readable baseline text across Streamlit-generated elements. */
            .stApp *, .stApp *::before, .stApp *::after {
                color: var(--fg);
            }

            .main .block-container {
                max-width: 100vw;
                height: calc(100vh - 2.6rem);
                overflow: hidden;
                padding-top: 0.15rem;
                padding-bottom: 0.15rem;
            }

            .stHorizontalBlock {
                gap: 0.45rem !important;
            }

            h1, h2, h3, h4, h5, h6,
            p,
            label,
            li,
            .stCaption,
            [data-testid="stMarkdownContainer"],
            [data-testid="stMarkdownContainer"] * {
                color: var(--fg) !important;
            }

            [data-testid="stMarkdownContainer"] code {
                color: var(--fg);
                background: transparent;
                border: 1px solid var(--line);
                border-radius: 0;
                padding: 0.05rem 0.35rem;
            }

            [data-testid="stExpander"] {
                border: 1px solid var(--line);
                border-radius: 0 !important;
                background: transparent;
            }

            [data-testid="stExpander"] > details {
                border-radius: 0 !important;
            }

            [data-testid="stFileUploaderDropzone"] {
                border: 1px solid var(--line) !important;
                border-radius: 0 !important;
                background: var(--panel) !important;
                color: var(--ink) !important;
            }

            [data-testid="stFileUploaderDropzone"] * {
                color: var(--ink) !important;
            }

            .stTextInput input,
            .stNumberInput input,
            .stTextArea textarea {
                border: 1px solid var(--line) !important;
                border-radius: 0 !important;
                background: var(--panel) !important;
                color: var(--ink) !important;
            }

            .stTextInput input::placeholder,
            .stTextArea textarea::placeholder {
                color: #222222 !important;
            }

            .stSelectbox [data-baseweb="select"] > div,
            .stMultiSelect [data-baseweb="select"] > div {
                border: 1px solid var(--line) !important;
                border-radius: 0 !important;
                background: var(--panel) !important;
                color: var(--ink) !important;
            }

            .stSelectbox [data-baseweb="select"] span,
            .stMultiSelect [data-baseweb="select"] span {
                color: var(--ink) !important;
            }

            .stNumberInput button,
            .stSelectbox svg,
            .stMultiSelect svg {
                color: var(--ink) !important;
                fill: var(--ink) !important;
            }

            .stSlider [role="slider"] {
                background: var(--fg) !important;
                border: 1px solid var(--line) !important;
                box-shadow: none !important;
            }

            .stSlider [data-baseweb="slider"] > div > div {
                background: #656565 !important;
                height: 4px !important;
            }

            div.stButton > button,
            div.stFormSubmitButton > button {
                border: 1px solid var(--line) !important;
                border-radius: 0 !important;
                background: var(--btn) !important;
                color: #f4f4f4 !important;
                font-weight: 600;
                letter-spacing: 0.02em;
                text-transform: uppercase;
            }

            div.stButton > button:hover,
            div.stFormSubmitButton > button:hover {
                background: var(--btn-hover) !important;
                color: #ffffff !important;
            }

            [data-testid="stForm"] {
                border: 1px solid var(--line);
                border-radius: 0;
                padding: 0.7rem;
                background: transparent;
            }

            [data-testid="stForm"] * {
                color: var(--fg) !important;
            }

            [data-testid="stVerticalBlock"] [data-testid="stDivider"] {
                border-top: 1px solid var(--line);
            }

            [data-testid="stAlert"] {
                border-radius: 0 !important;
                border: 1px solid var(--line) !important;
                background: var(--panel) !important;
                color: var(--ink) !important;
            }

            [data-testid="stAlert"] * {
                color: var(--ink) !important;
            }

            [data-baseweb="popover"],
            [role="listbox"] {
                background: var(--panel) !important;
                color: var(--ink) !important;
                border-radius: 0 !important;
                border: 1px solid var(--line) !important;
            }

            [role="option"] {
                color: var(--ink) !important;
            }

            [role="option"] * {
                color: var(--ink) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


@st.cache_data(show_spinner=False)
def get_cached_ollama_models() -> list[str]:
    return discover_ollama_models()


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


def load_history() -> list[dict]:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT time, question, answer, llm_model, embed_model, top_k, retrieval_mode, rerank_enabled, rerank_top_n
            FROM history_entries
            ORDER BY id DESC
            LIMIT 50
            """
        ).fetchall()

    history: list[dict] = []
    for row in rows:
        history.append(
            {
                "time": row["time"],
                "question": row["question"],
                "answer": row["answer"],
                "llm_model": row["llm_model"],
                "embed_model": row["embed_model"],
                "top_k": int(row["top_k"] or 0),
                "retrieval_mode": row["retrieval_mode"],
                "rerank_enabled": bool(row["rerank_enabled"]),
                "rerank_top_n": int(row["rerank_top_n"] or 0),
            }
        )
    return history


def init_history_db() -> None:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                llm_model TEXT NOT NULL,
                embed_model TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                retrieval_mode TEXT NOT NULL,
                rerank_enabled INTEGER NOT NULL,
                rerank_top_n INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def save_history_entry(item: dict) -> None:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO history_entries (
                time, question, answer, llm_model, embed_model, top_k, retrieval_mode, rerank_enabled, rerank_top_n
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.get("time", time.strftime("%Y-%m-%d %H:%M:%S")),
                item.get("question", ""),
                item.get("answer", ""),
                item.get("llm_model", ""),
                item.get("embed_model", ""),
                int(item.get("top_k", 0)),
                item.get("retrieval_mode", "dense"),
                1 if bool(item.get("rerank_enabled", False)) else 0,
                int(item.get("rerank_top_n", 0)),
            ),
        )
        conn.commit()


def clear_history() -> None:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute("DELETE FROM history_entries")
        conn.commit()


def migrate_json_history_to_sqlite() -> None:
    if not HISTORY_JSON_PATH.exists():
        return

    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM history_entries").fetchone()[0]
    if count > 0:
        return

    try:
        existing = json.loads(HISTORY_JSON_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(existing, list):
        return

    for item in existing:
        if not isinstance(item, dict):
            continue
        save_history_entry(item)


async def send_rag_ingest_event(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/inngest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embed_model": embed_model,
            },
        )
    )


async def send_rag_query_event(
    question: str,
    top_k: int,
    embed_model: str,
    llm_model: str,
    retrieval_mode: str,
    rerank_enabled: bool,
    rerank_top_n: int,
) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
                "embed_model": embed_model,
                "llm_model": llm_model,
                "retrieval_mode": retrieval_mode,
                "rerank_enabled": rerank_enabled,
                "rerank_top_n": rerank_top_n,
            },
        )
    )
    return result[0]


def inngest_api_base() -> str:
    return "http://127.0.0.1:8288/v1"


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


def ensure_state() -> None:
    init_history_db()
    migrate_json_history_to_sqlite()
    if "history" not in st.session_state:
        st.session_state.history = load_history()


ensure_state()
apply_brutalist_theme()
header_title_col, header_refresh_col = st.columns([6, 1.1], gap="small")
with header_title_col:
    st.markdown("<h1 style='margin: 0; line-height: 1.1;'>DOCUMENT RAG</h1>", unsafe_allow_html=True)
    st.caption("INGEST, RETRIEVE, ANSWER.")

with header_refresh_col:
    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
    if st.button("Refresh Models", use_container_width=True):
        get_cached_ollama_models.clear()
        st.rerun()

all_models = get_cached_ollama_models()
embed_candidates, llm_candidates = split_ollama_models(all_models)

if not all_models:
    st.warning("No Ollama models were detected. Start Ollama and pull at least one model, or type a model name manually.")

embed_options = embed_candidates if embed_candidates else ["No embedding models detected"]
llm_options = llm_candidates if llm_candidates else ["No chat models detected"]

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

PANEL_HEIGHT = 540
ingest_col, query_col, answer_col = st.columns([1.05, 1.2, 1.05], gap="small")

with ingest_col:
    with st.container(height=PANEL_HEIGHT, border=True):
        st.subheader("INGEST")
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            chunk_size = st.slider("Chunk", min_value=128, max_value=2048, value=512, step=64)
        with s_col2:
            chunk_overlap = st.slider("Overlap", min_value=0, max_value=512, value=100, step=16)

        embed_model = st.selectbox("Embedding", options=embed_options, index=0)
        custom_embed = st.text_input("Custom embed model", value="")

        ingest_clicked = st.button("Ingest PDF", type="primary", use_container_width=True)
        if ingest_clicked:
            if uploaded is None:
                st.error("Please upload a PDF before ingestion.")
            else:
                selected_embed = custom_embed.strip() or (embed_model if embed_model in embed_candidates else "")
                if not selected_embed:
                    st.error("No embedding model selected. Pick a detected model or enter one manually.")
                    st.stop()
                with st.spinner("Ingesting PDF..."):
                    path = save_uploaded_pdf(uploaded)
                    asyncio.run(
                        send_rag_ingest_event(
                            pdf_path=path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            embed_model=selected_embed,
                        )
                    )
                st.success(f"Ingestion triggered for {path.name}")

with query_col:
    with st.container(height=PANEL_HEIGHT, border=True):
        st.subheader("QUERY")
        with st.form("rag_query_form"):
            question = st.text_input("Ask")

            q_col1, q_col2 = st.columns(2)
            with q_col1:
                top_k = st.number_input("Top-K", min_value=1, max_value=20, value=5, step=1)
            with q_col2:
                llm_model = st.selectbox("LLM", options=llm_options, index=0)

            custom_llm = st.text_input("Custom LLM", value="")
            query_embed_model = st.selectbox("Query embedding", options=embed_options, index=0)
            retrieval_mode = st.selectbox("Retrieval", options=["dense", "hybrid"], index=1)
            rerank_enabled = st.toggle("Rerank context before answer", value=True)
            rerank_top_n = st.number_input("Rerank Top-N", min_value=1, max_value=30, value=5, step=1)

            submitted = st.form_submit_button("Ask", use_container_width=True)

        if submitted and question.strip():
            selected_llm = custom_llm.strip() or (llm_model if llm_model in llm_candidates else "")
            selected_query_embed = query_embed_model if query_embed_model in embed_candidates else ""
            if not selected_llm:
                st.error("No LLM selected. Pick a detected model or enter one manually.")
                st.stop()
            if not selected_query_embed:
                st.error("No query embedding model selected. Pick a detected model first.")
                st.stop()
            with st.spinner("Generating answer..."):
                event_id = asyncio.run(
                    send_rag_query_event(
                        question=question.strip(),
                        top_k=int(top_k),
                        embed_model=selected_query_embed,
                        llm_model=selected_llm,
                        retrieval_mode=retrieval_mode,
                        rerank_enabled=bool(rerank_enabled),
                        rerank_top_n=int(rerank_top_n),
                    )
                )
                output = wait_for_run_output(event_id)

            answer = output.get("answer", "")
            sources = output.get("sources", [])
            st.session_state.last_answer = answer
            st.session_state.last_sources = sources

            history_item = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question": question.strip(),
                "answer": answer,
                "llm_model": selected_llm,
                "embed_model": selected_query_embed,
                "top_k": int(top_k),
                "retrieval_mode": retrieval_mode,
                "rerank_enabled": bool(rerank_enabled),
                "rerank_top_n": int(rerank_top_n),
            }
            save_history_entry(history_item)
            st.session_state.history = load_history()

with answer_col:
    with st.container(height=PANEL_HEIGHT, border=True):
        st.subheader("ANSWER")
        st.write(st.session_state.last_answer or "(No answer yet)")
        if st.session_state.last_sources:
            st.caption("Sources")
            for src in st.session_state.last_sources:
                st.write(f"- {src}")

        st.markdown("---")
        h_left, h_right = st.columns([2.2, 1])
        with h_left:
            st.subheader("HISTORY")
        with h_right:
            if st.button("Clear", use_container_width=True):
                clear_history()
                st.session_state.history = []

        with st.container(height=190):
            if not st.session_state.history:
                st.caption("No history yet.")
            else:
                for item in st.session_state.history[:8]:
                    with st.expander(f"{item['time']} | {item['question'][:38]}"):
                        st.write(item.get("answer", ""))
                        st.caption(
                            f"LLM: {item.get('llm_model', '')} | EMB: {item.get('embed_model', '')} | TOP-K: {item.get('top_k', '')} | RET: {item.get('retrieval_mode', '')} | RERANK: {item.get('rerank_enabled', '')}"
                        )
                if len(st.session_state.history) > 8:
                    st.caption(f"Showing 8 of {len(st.session_state.history)} items.")
