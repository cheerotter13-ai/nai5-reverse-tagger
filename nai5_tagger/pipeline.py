from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from nai5_tagger.compiler import compile
from nai5_tagger.grok_vision import GrokVisionError, analyze_image
from nai5_tagger.metadata_reader import read_metadata
from nai5_tagger.types import (
    Action,
    BasePrompt,
    CharacterPrompt,
    CompileOptions,
    MetadataResult,
    Nai5Prompt,
    PromptMeta,
    UcBlock,
)
from nai5_tagger.wd14_tagger import tag_image

_GENDERS = {"girl", "boy", "other"}
_ACTION_ROLES = {"source", "target", "mutual"}


class PipelineError(Exception):
    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


def metadata_to_prompt(result: MetadataResult) -> Nai5Prompt:
    tags = [part.strip() for part in result.base_caption.split(",") if part.strip()]
    characters: list[CharacterPrompt] = []
    for caption in result.char_captions:
        parts = [part.strip() for part in caption.split(",") if part.strip()]
        gender = "girl"
        rest = parts
        if parts and parts[0].lower() in _GENDERS:
            gender = parts[0].lower()
            rest = parts[1:]
        appearance: list[str] = []
        actions: list[Action] = []
        for token in rest:
            action = _parse_action(token)
            if action is not None:
                actions.append(action)
            else:
                appearance.append(token)
        characters.append(
            CharacterPrompt(
                gender=gender,
                identity="",
                appearance=appearance,
                clothing=[],
                pose=[],
                expression=[],
                actions=actions,
            )
        )
    return Nai5Prompt(
        base=BasePrompt(tags=tags, nl=""),
        characters=characters,
        uc=UcBlock(preset="default", extra=[]),
        unassigned=[],
        ignored=[],
        warnings=[],
        meta=PromptMeta(
            source="metadata",
            count_tag=tags[0] if tags else "",
            wd14_skipped=False,
            character_count=len(characters),
        ),
    )


def run_pipeline(
    path: str | Path,
    options: CompileOptions | None = None,
    *,
    metadata_fn=None,
    wd14_fn=None,
    vision_fn=None,
) -> Nai5Prompt:
    image_path = Path(path)
    _require_image(image_path)

    reader = read_metadata if metadata_fn is None else metadata_fn
    meta = reader(image_path)
    if meta is not None:
        return metadata_to_prompt(meta)

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        rgb.load()
        rgb = rgb.copy()

    tagger = tag_image if wd14_fn is None else wd14_fn
    skipped = False
    try:
        tags = tagger(rgb)
    except Exception:
        tags = []
        skipped = True

    vision = analyze_image if vision_fn is None else vision_fn
    try:
        draft = vision(image_path.read_bytes())
    except GrokVisionError as exc:
        raise PipelineError(str(exc), 2) from exc

    prompt = compile(draft, tags, options)
    prompt.meta.wd14_skipped = skipped
    prompt.meta.source = "reverse"
    return prompt


def _parse_action(token: str) -> Action | None:
    role, sep, verb = token.partition("#")
    if not sep:
        return None
    role = role.strip().lower()
    verb = verb.strip()
    if role in _ACTION_ROLES and verb:
        return Action(role=role, verb=verb)
    return None


def _require_image(path: Path) -> None:
    if not path.is_file():
        raise PipelineError(f"missing image: {path}", 1)
    try:
        with Image.open(path) as image:
            image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise PipelineError(f"unreadable image: {path}", 1) from exc
