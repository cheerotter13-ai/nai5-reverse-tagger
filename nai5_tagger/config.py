from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> None:
    env_path = path if path is not None else _ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'").strip('"')


def env_vlm() -> dict[str, str]:
    load_dotenv()
    return {
        "base": os.environ.get("NAI5_TAGGER_VLM_BASE", "").strip(),
        "model": os.environ.get("NAI5_TAGGER_VLM_MODEL", "").strip(),
        "key": os.environ.get("NAI5_TAGGER_VLM_KEY", "").strip(),
        "wd14_dir": os.environ.get("NAI5_TAGGER_WD14_DIR", "").strip(),
    }
