"""
LoanGuard AI - Workspace Root Training Launcher
Enables direct execution of 'python train_model.py' from workspace root.
"""

import sys
from pathlib import Path

LOANGUARD_DIR = Path(__file__).resolve().parent / "LoanGuardAI"
if str(LOANGUARD_DIR) not in sys.path:
    sys.path.insert(0, str(LOANGUARD_DIR))

import runpy

if __name__ == "__main__":
    runpy.run_path(str(LOANGUARD_DIR / "train_model.py"), run_name="__main__")
