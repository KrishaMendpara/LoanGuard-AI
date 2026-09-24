"""
LoanGuard AI - Workspace Root Launcher
Enables direct execution of 'streamlit run app.py' from workspace root.
"""

import sys
from pathlib import Path

LOANGUARD_DIR = Path(__file__).resolve().parent / "LoanGuardAI"
if str(LOANGUARD_DIR) not in sys.path:
    sys.path.insert(0, str(LOANGUARD_DIR))

import runpy

if __name__ == "__main__":
    runpy.run_path(str(LOANGUARD_DIR / "app.py"), run_name="__main__")
