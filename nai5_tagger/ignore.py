from __future__ import annotations

from pathlib import Path

_IGNORE_PATH = Path(__file__).resolve().parent / "data" / "ignore_tags.txt"
_IGNORED: set[str] = {
    line.strip().lower()
    for line in _IGNORE_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
}


def normalize_tag(text: str) -> str:
    return text.lower().replace("_", " ").strip()


def is_ignored(text: str, category: int | None = None) -> bool:
    if category == 1:
        return True
    return normalize_tag(text) in _IGNORED
