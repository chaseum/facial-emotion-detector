from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = PROJECT / "artifacts"
ASSETS = PROJECT / "assets"
MODELS_DIR = PROJECT / "models"
CONFIGS = PROJECT / "configs"
DATA_DIR = PROJECT / "data"

(ARTIFACTS).mkdir(exist_ok=True, parents = True)
