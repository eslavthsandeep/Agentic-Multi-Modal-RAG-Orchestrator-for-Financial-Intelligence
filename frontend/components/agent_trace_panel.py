"""
Agent trace panel component.
Visualizes the LangGraph step-by-step reasoning process.
"""

import streamlit as st

def render_trace(trace: list[dict]) -> None:
    """Show the agent reasoning pipeline as a step-by-step trace."""
    
    icon_map = {
        "supervisor": "🎯",
        "search": "🔍",
        "sql": "📊",
        "vision": "👁️",
        "synthesizer": "✍️"
    }
    
    color_map = {
        "supervisor": "#ff9800",
        "search": "#2196f3",
        "sql": "#4caf50",
        "vision": "#9c27b0",
        "synthesizer": "#e91e63"
    }

    with st.expander("🔬 Agent Reasoning Trace"):
        for step in trace:
            name = step.get("step", "unknown").lower()
            detail = step.get("detail", "")
            icon = icon_map.get(name, "⚙️")
            color = color_map.get(name, "#888")
            
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding-left: 10px; margin-bottom: 10px; background-color: #1a1e26; padding: 10px; border-radius: 0 8px 8px 0;">
                <strong>{icon} {name.title()}</strong>
                <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #ccc;">{detail}</p>
            </div>
            """, unsafe_allow_html=True)
