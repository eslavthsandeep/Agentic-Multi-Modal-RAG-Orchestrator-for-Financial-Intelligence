import os
import sys
import time
import requests
import subprocess
import streamlit as st

from components.chat_ui import render_chat, add_message
from components.citation_viewer import render_citations
from components.agent_trace_panel import render_trace

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def _ensure_backend_running():
    """Ensure FastAPI backend is running in background if not already active (e.g. on Streamlit Cloud)."""
    try:
        r = requests.get("http://127.0.0.1:8000/api/agents/status", timeout=1)
        if r.status_code == 200:
            return
    except Exception:
        pass
    
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
    if os.path.exists(backend_dir):
        env = os.environ.copy()
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{backend_dir}{os.pathsep}{python_path}"
        try:
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
                cwd=backend_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(3)
        except Exception:
            pass

_ensure_backend_running()

st.set_page_config(
    page_title="OmniBrain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e6edf3;
    }
    .status-dot {
        height: 10px;
        width: 10px;
        background-color: #2ea043;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    .status-dot.offline {
        background-color: #da3633;
    }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "document_status" not in st.session_state:
    st.session_state.document_status = None
if "citations" not in st.session_state:
    st.session_state.citations = []
if "trace" not in st.session_state:
    st.session_state.trace = []

def get_working_api_url() -> str:
    urls_to_check = [
        os.getenv("BACKEND_URL", ""),
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "https://omnibrain-api.loca.lt"
    ]
    for base in urls_to_check:
        if not base:
            continue
        try:
            r = requests.get(f"{base}/api/agents/status", timeout=2)
            if r.status_code == 200:
                return base
        except Exception:
            continue
    return "http://127.0.0.1:8000"

def check_system_status() -> dict:
    base_url = get_working_api_url()
    try:
        resp = requests.get(f"{base_url}/api/agents/status", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"qdrant": "offline", "sql_db": "offline", "langfuse": "offline"}

def poll_document_status(doc_id: str):
    placeholder = st.sidebar.empty()
    base_url = get_working_api_url()
    while True:
        try:
            resp = requests.get(f"{base_url}/api/upload/{doc_id}/status", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                pages = data.get("pages_processed", 0)
                if status == "processing":
                    placeholder.info(f"Processing... {pages} pages done.")
                    time.sleep(2)
                elif status == "ready":
                    placeholder.success("Document ready!")
                    st.session_state.document_status = "ready"
                    break
                else:
                    placeholder.error("Processing failed.")
                    break
        except Exception:
            placeholder.error("Error checking status.")
            break

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a financial PDF", type=["pdf"])
    if st.button("Process Document") and uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        base_url = get_working_api_url()
        try:
            res = requests.post(f"{base_url}/api/upload", files=files)
            if res.status_code == 200:
                doc_id = res.json().get("document_id")
                st.session_state.document_id = doc_id
                poll_document_status(doc_id)
        except Exception:
            st.error("Failed to upload. Ensure backend is running.")
            
    st.divider()
    st.header("System Status")
    status_data = check_system_status()
    for component, state in status_data.items():
        is_active = state in ("online", "connected", "configured", "enabled (local trace)")
        color = "online" if is_active else "offline"
        label = f"{component.replace('_', ' ').title()}: <b>{state.title()}</b>"
        st.markdown(f'<div class="status-dot {color}"></div> {label}', unsafe_allow_html=True)
        
st.title("🧠 OmniBrain")
st.caption("Agentic Multi-Modal RAG Orchestrator")

render_chat(st.session_state.messages)

if query := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append(add_message("user", query))
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    query_text = st.session_state.messages[-1]["content"]
    with st.spinner("Thinking..."):
        try:
            payload = {
                "document_id": st.session_state.document_id or "",
                "query": query_text,
                "chat_history": st.session_state.messages[:-1]
            }
            base_url = get_working_api_url()
            resp = requests.post(f"{base_url}/api/query", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.messages.append(add_message("assistant", data.get("answer", "")))
                st.session_state.citations = data.get("citations", [])
                st.session_state.trace = data.get("agent_trace", [])
            else:
                st.session_state.messages.append(add_message("assistant", f"Error: {resp.text}"))
        except Exception as e:
            st.session_state.messages.append(add_message("assistant", "Could not connect to backend."))
    st.rerun()

if st.session_state.trace:
    render_trace(st.session_state.trace)

if st.session_state.citations:
    st.divider()
    st.header("Citations")
    render_citations(st.session_state.citations)
