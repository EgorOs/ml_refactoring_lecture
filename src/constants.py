import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv('PROJ_ROOT', Path(__file__).resolve().parents[1]))
