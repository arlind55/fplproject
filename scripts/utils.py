from pathlib import Path
from datetime import date
import json


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def today_str() -> str:
    return date.today().isoformat()
