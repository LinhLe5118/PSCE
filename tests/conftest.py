"""Run from a fresh checkout without importing a stdlib-shadowing code package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
