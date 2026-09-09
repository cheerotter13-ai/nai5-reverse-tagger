from __future__ import annotations

from nai5_tagger.types import CharacterPrompt, Nai5Prompt


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

    return "\n\n".join(sections)
