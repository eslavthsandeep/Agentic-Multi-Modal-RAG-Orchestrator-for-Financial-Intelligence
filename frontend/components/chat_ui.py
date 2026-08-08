import re
import streamlit as st

def add_message(role: str, content: str, metadata: dict | None = None) -> dict:
    """Create a message dict for the chat history."""
    return {
        "role": role,
        "content": content,
        "metadata": metadata or {}
    }

def _safe_markdown_content(text: str) -> str:
    """Escape unescaped dollar signs to prevent Streamlit KaTeX math-mode rendering collisions."""
    if not text:
        return ""
    # Replace $ with \$ if not already escaped
    return re.sub(r'(?<!\\)\$', r'\\$', text)

def render_chat(messages: list[dict]) -> None:
    """Render the chat message history with styled bubbles."""
    st.markdown("""
        <style>
        .chat-bubble-user {
            background-color: #2b313e;
            padding: 15px;
            border-radius: 15px 15px 0 15px;
            margin: 10px 0;
            text-align: right;
            border: 1px solid #3d4454;
        }
        .chat-bubble-assistant {
            background: linear-gradient(145deg, #1e222a, #232832);
            padding: 15px;
            border-radius: 15px 15px 15px 0;
            margin: 10px 0;
            border-left: 4px solid #8a2be2;
            border-top: 1px solid #3d4454;
            border-right: 1px solid #3d4454;
            border-bottom: 1px solid #3d4454;
        }
        .route-tag {
            font-size: 0.8em;
            padding: 2px 6px;
            border-radius: 10px;
            margin-right: 5px;
            background-color: #3d4454;
        }
        </style>
    """, unsafe_allow_html=True)
    
    for msg in messages:
        if msg["role"] == "user":
            safe_text = _safe_markdown_content(msg["content"])
            st.markdown(f'<div class="chat-bubble-user">{safe_text}</div>', unsafe_allow_html=True)
        else:
            content = _safe_markdown_content(msg["content"])
            metadata = msg.get("metadata", {})
            tags_html = ""
            if "route" in metadata:
                route = metadata["route"]
                icon = "🔍" if route == "search" else "📊" if route == "sql" else "👁️"
                tags_html = f'<span class="route-tag">{icon} {route.title()}</span><br><br>'
                
            st.markdown(f'<div class="chat-bubble-assistant">{tags_html}{content}</div>', unsafe_allow_html=True)
