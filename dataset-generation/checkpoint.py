import json

from pathlib import Path
from config import Config


def get_checkpoint_path(config: Config, pdf_name: str, stage: str) -> Path:
    return config.checkpoint_dir / f"{Path(pdf_name).stem}_{stage}.json"


def save_checkpoint(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def load_checkpoint(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
