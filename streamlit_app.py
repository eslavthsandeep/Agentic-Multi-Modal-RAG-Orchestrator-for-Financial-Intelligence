"""Root entrypoint for Streamlit Cloud deployments that point to streamlit_app.py."""
import os
import sys
from pathlib import Path

# Ensure root and backend are in sys.path
root_dir = Path(__file__).resolve().parent
backend_dir = root_dir / "backend"
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import runpy
target_script = root_dir / "frontend" / "streamlit_app.py"
runpy.run_path(str(target_script), run_name="__main__")
