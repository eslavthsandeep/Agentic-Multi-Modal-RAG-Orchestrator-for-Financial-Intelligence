"""
Citation viewer component for displaying sources.
Can render snippets and ideally fetch PDF pages if provided.
"""

import streamlit as st

def render_citations(citations: list[dict], pdf_path: str | None = None) -> None:
    """Display citation details with PDF page rendering where applicable."""
    
    source_colors = {
        "text": "#1f77b4",   # Blue
        "table": "#2ca02c",  # Green
        "chart": "#9467bd"   # Purple
    }
    
    for i, citation in enumerate(citations, 1):
        page = citation.get("page", "?")
        source_type = citation.get("source_type", "text")
        snippet = citation.get("snippet", "")
        
        color = source_colors.get(source_type.lower(), "#555")
        badge = f'<span style="background-color: {color}; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold;">{source_type.upper()}</span>'
        
        with st.expander(f"[{i}] Page {page}"):
            st.markdown(f"{badge} Source: Page {page}", unsafe_allow_html=True)
            st.markdown(f"**Snippet:**\n> {snippet}")
            
            if pdf_path:
                try:
                    import fitz
                    doc = fitz.open(pdf_path)
                    page_num = max(0, int(page) - 1)
                    if page_num < len(doc):
                        pix = doc[page_num].get_pixmap(dpi=150)
                        img_data = pix.tobytes("png")
                        st.image(img_data, caption=f"Page {page}")
                except Exception as e:
                    st.error(f"Could not render PDF page: {e}")
