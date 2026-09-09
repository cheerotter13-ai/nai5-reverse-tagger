from __future__ import annotations

from pathlib import Path

from nai5_tagger.types import CharacterPrompt, Nai5Prompt

_UC_DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "uc_default.txt"


def _load_uc(preset: str) -> str:
    if preset == "default":
        return _UC_DEFAULT_PATH.read_text(encoding="utf-8").strip()
    return ""


def _character_line(ch: CharacterPrompt) -> str:
    parts: list[str] = [ch.gender]
    if ch.identity:
        parts.append(ch.identity)
    parts.extend(ch.appearance)
    parts.extend(ch.clothing)
    parts.extend(ch.expression)
    parts.extend(ch.pose)
    for action in ch.actions:
        parts.append(f"{action.role}#{action.verb}")
    return ", ".join(p for p in parts if p)


def render(prompt: Nai5Prompt) -> str:
    base = "Prompt:\n" + ", ".join(prompt.base.tags)
    nl = prompt.base.nl.strip()
    if nl:
        base += "\n\n" + nl

    sections = [base]
    for i, ch in enumerate(prompt.characters, start=1):
        sections.append(f"Character {i}:\n{_character_line(ch)}")

    uc = _load_uc(prompt.uc.preset)
    sections.append(f"UC:\n{uc}")

    return "\n\n".join(sections)
