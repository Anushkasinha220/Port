import sys
from pathlib import Path

# Ensures the project root is always on sys.path when running via pytest or GitHub Actions
sys.path.insert(0, str(Path(__file__).resolve().parent))