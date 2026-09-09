from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Gender = Literal["girl", "boy", "other"]
ActionRole = Literal["source", "target", "mutual"]


def as_token_list(value: object) -> list[str]:
    """Coerce VLM JSON that sent a string instead of a tag list."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return [str(item).strip() for item in value if str(item).strip()]


def as_nl_text(value: object) -> str:
    return value if isinstance(value, str) else ""


@dataclass
class Action:
    role: ActionRole
    verb: str


@dataclass
class CharacterDraft:
    gender: Gender
    identity: str = ""
    appearance: list[str] = field(default_factory=list)
    clothing: list[str] = field(default_factory=list)
    pose: list[str] = field(default_factory=list)
    expression: list[str] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)


@dataclass
class SceneDraft:
    count_tag: str
    themes: list[str]
    scene: list[str]
    camera: list[str]
    nsfw: bool
    nl: str
    characters: list[CharacterDraft]

    @classmethod
    def from_dict(cls, data: dict) -> "SceneDraft":
        chars = []
        for raw in data.get("characters") or []:
            actions = [
                Action(role=a["role"], verb=a["verb"])
                for a in raw.get("actions") or []
            ]
            chars.append(
                CharacterDraft(
                    gender=raw["gender"],
                    identity=raw.get("identity") or "",
                    appearance=as_token_list(raw.get("appearance")),
                    clothing=as_token_list(raw.get("clothing")),
                    pose=as_token_list(raw.get("pose")),
                    expression=as_token_list(raw.get("expression")),
                    actions=actions,
                )
            )
        return cls(
            count_tag=data["count_tag"],
            themes=as_token_list(data.get("themes")),
            scene=as_token_list(data.get("scene")),
            camera=as_token_list(data.get("camera")),
            nsfw=bool(data.get("nsfw")),
            nl=as_nl_text(data.get("nl")),
            characters=chars,
        )


@dataclass
class TagHit:
    name: str
    score: float
    category: int


@dataclass
class CompileOptions:
    include_nl: bool | None = None
    uc_preset: str = "default"


@dataclass
class BasePrompt:
    tags: list[str]
    nl: str


@dataclass
class CharacterPrompt:
    gender: Gender
    identity: str
    appearance: list[str]
    clothing: list[str]
    pose: list[str]
    expression: list[str]
    actions: list[Action]


@dataclass
class UcBlock:
    preset: str
    extra: list[str]


@dataclass
class PromptMeta:
    source: Literal["metadata", "reverse"]
    count_tag: str
    wd14_skipped: bool
    character_count: int


@dataclass
class Nai5Prompt:
    base: BasePrompt
    characters: list[CharacterPrompt]
    uc: UcBlock
    unassigned: list[str]
    ignored: list[str]
    warnings: list[str]
    meta: PromptMeta

    @classmethod
    def from_dict(cls, data: dict) -> "Nai5Prompt":
        raw_base = data.get("base") or {}
        characters = []
        for raw in data.get("characters") or []:
            actions = [
                Action(role=a["role"], verb=a["verb"])
                for a in raw.get("actions") or []
            ]
            characters.append(
                CharacterPrompt(
                    gender=raw["gender"],
                    identity=raw.get("identity") or "",
                    appearance=as_token_list(raw.get("appearance")),
                    clothing=as_token_list(raw.get("clothing")),
                    pose=as_token_list(raw.get("pose")),
                    expression=as_token_list(raw.get("expression")),
                    actions=actions,
                )
            )
        raw_uc = data.get("uc") or {}
        raw_meta = data.get("meta") or {}
        return cls(
            base=BasePrompt(
                tags=list(raw_base.get("tags") or []),
                nl=as_nl_text(raw_base.get("nl")),
            ),
            characters=characters,
            uc=UcBlock(
                preset=raw_uc.get("preset") or "default",
                extra=list(raw_uc.get("extra") or []),
            ),
            unassigned=list(data.get("unassigned") or []),
            ignored=list(data.get("ignored") or []),
            warnings=list(data.get("warnings") or []),
            meta=PromptMeta(
                source=raw_meta.get("source") or "reverse",
                count_tag=raw_meta.get("count_tag") or "",
                wd14_skipped=bool(raw_meta.get("wd14_skipped")),
                character_count=int(raw_meta.get("character_count") or len(characters)),
            ),
        )


def nai5_prompt_to_dict(prompt: Nai5Prompt) -> dict:
    return asdict(prompt)


@dataclass
class MetadataResult:
    base_caption: str
    char_captions: list[str]
    uc: str = ""
