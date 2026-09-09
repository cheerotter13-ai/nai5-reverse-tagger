from __future__ import annotations

import copy
import re

from nai5_tagger.ignore import is_ignored, normalize_tag
from nai5_tagger.types import (
    Action,
    BasePrompt,
    CharacterDraft,
    CharacterPrompt,
    CompileOptions,
    Nai5Prompt,
    PromptMeta,
    SceneDraft,
    TagHit,
    UcBlock,
)

_COUNT_LIKE = frozenset({
    "1girl",
    "2girls",
    "1boy",
    "2boys",
    "1other",
    "solo",
    "multiple girls",
    "multiple boys",
})
_FAKE_CHAR = frozenset({"char1", "char2", "character 1"})
_NSFW_LIKE = frozenset({"explicit", "rating:explicit", "nsfw"})
_THEME_TAGS = frozenset({
    "yuri",
    "yaoi",
    "hetero",
    "ntr",
    "netorare",
    "cuckolding",
    "cuckolding theme",
})
_SCENE_TAGS = frozenset({
    "indoors",
    "outdoors",
    "pink background",
    "simple background",
})
_CAMERA_TAGS = frozenset({
    "first person view",
    "from behind",
    "from below",
    "cowboy shot",
    "pov",
})
_POV = "pov"
_DROP_CAMERA = frozenset({"low angle", "low-angle", "lowangle"})
_FIRST_PERSON_VIEW = "first person view"
_WEIGHT_RE = re.compile(r"^(\d+(?:\.\d+)?)::(.*)::$", re.DOTALL)
_COUNT_TOKEN_RE = re.compile(r"^\d+(?:girls?|boys?|others?)$")
_FAKE_CHAR_RE = re.compile(r"^char(?:acter)?\s*\d+$")
_CHAR_DEST_RE = re.compile(r"^char:(\d+)$")
_MAX_TAG_WORDS = 6
_LOOK_KEYS = (
    "hair",
    "bangs",
    "ponytail",
    "twintail",
    "ahoge",
    "braid",
    "eyes",
    "iris",
    "pupil",
    "skin",
    "breasts",
    "ears",
    "mole",
    "scar",
    "tattoo",
)


def is_short_tag(token: str) -> bool:
    words = [part for part in normalize_tag(token).replace("-", " ").split() if part]
    return 1 <= len(words) <= _MAX_TAG_WORDS


def _is_look_token(token: str) -> bool:
    text = normalize_tag(token)
    return any(key in text for key in _LOOK_KEYS)


def strip_weight(text: str) -> str:
    s = text.strip()
    while True:
        prev = s
        s = s.strip()
        while len(s) >= 2 and (
            (s[0] == "{" and s[-1] == "}") or (s[0] == "[" and s[-1] == "]")
        ):
            s = s[1:-1].strip()
        matched = _WEIGHT_RE.fullmatch(s)
        if matched:
            s = matched.group(2).strip()
        if s == prev:
            break
    return normalize_tag(s)


def count_tag_from_characters(characters: list[CharacterDraft]) -> str:
    girls = sum(1 for ch in characters if ch.gender == "girl")
    boys = sum(1 for ch in characters if ch.gender == "boy")
    others = sum(1 for ch in characters if ch.gender == "other")
    parts: list[str] = []
    if girls == 1:
        parts.append("1girl")
    elif girls > 1:
        parts.append(f"{girls}girls")
    if boys == 1:
        parts.append("1boy")
    elif boys > 1:
        parts.append(f"{boys}boys")
    if others == 1:
        parts.append("1other")
    elif others > 1:
        parts.append(f"{others}others")
    return ", ".join(parts)


def _is_count_token(text: str) -> bool:
    key = normalize_tag(text)
    if key in _COUNT_LIKE:
        return True
    return bool(_COUNT_TOKEN_RE.fullmatch(key.replace(" ", "")))


def _canonical_count_token(text: str) -> str:
    key = normalize_tag(text)
    compact = key.replace(" ", "")
    if _COUNT_TOKEN_RE.fullmatch(compact):
        return compact
    return key


def _is_fake_char(text: str) -> bool:
    key = normalize_tag(text)
    compact = key.replace(" ", "")
    return key in _FAKE_CHAR or compact in _FAKE_CHAR or bool(_FAKE_CHAR_RE.fullmatch(key))


def _identity_hit_matches(identity: str, hit_key: str) -> bool:
    ident = normalize_tag(identity)
    if not ident or not hit_key:
        return False
    return hit_key == ident or hit_key in ident or ident in hit_key


_VOICE_PREFIXES = (
    "being ",
    "been ",
    "gets ",
    "get ",
    "got ",
    "is ",
    "was ",
    "having ",
    "has ",
)
_VERB_STEMS = (
    (re.compile(r"^sits\b"), "sitting"),
    (re.compile(r"^sat\b"), "sitting"),
    (re.compile(r"^pins\b"), "pinning"),
    (re.compile(r"^pinned\b"), "pinning"),
    (re.compile(r"^presses\b"), "pressing"),
    (re.compile(r"^pressed\b"), "pressing"),
    (re.compile(r"^kisses\b"), "kissing"),
    (re.compile(r"^kissed\b"), "kissing"),
    (re.compile(r"^holds\b"), "holding"),
    (re.compile(r"^held\b"), "holding"),
    (re.compile(r"^hugs\b"), "hug"),
    (re.compile(r"^hugged\b"), "hug"),
    (re.compile(r"^hugging\b"), "hug"),
    (re.compile(r"^penetrates\b"), "penetrating"),
    (re.compile(r"^penetrated\b"), "penetrating"),
    (re.compile(r"^covers\b"), "covering"),
    (re.compile(r"^covered\b"), "covering"),
    (re.compile(r"^leans\b"), "leaning"),
    (re.compile(r"^leant\b"), "leaning"),
    (re.compile(r"^facesits\b"), "sitting on face"),
    (re.compile(r"^facesat\b"), "sitting on face"),
)


def canonical_action_verb(verb: str) -> str:
    text = normalize_tag(verb)
    stripped = True
    while stripped:
        stripped = False
        for prefix in _VOICE_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix) :]
                stripped = True
                break
    for pattern, replacement in _VERB_STEMS:
        if pattern.match(text):
            text = pattern.sub(replacement, text, count=1)
            break
    return " ".join(text.split())


def _verbs_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    if min(len(left), len(right)) < 8:
        return False
    return left in right or right in left


_RESTRAINED_HINTS = (
    "arms behind back",
    "hands behind back",
    "bound arms",
    "armbinder",
)
_ARM_ACTION_PREFIXES = ("hug", "embrace", "hold person", "holding person")


def _character_blob(ch: CharacterPrompt) -> str:
    return " ".join((*ch.pose, *ch.clothing, *ch.appearance, ch.identity)).lower()


def _is_restrained(ch: CharacterPrompt) -> bool:
    blob = _character_blob(ch)
    return any(hint in blob for hint in _RESTRAINED_HINTS)


def _is_arm_action(verb: str) -> bool:
    text = canonical_action_verb(verb)
    return any(text == prefix or text.startswith(prefix + " ") for prefix in _ARM_ACTION_PREFIXES)


def _fix_restrained_hug_roles(characters: list[CharacterPrompt]) -> None:
    for i, ch in enumerate(characters):
        if not _is_restrained(ch):
            continue
        for action in ch.actions:
            if not _is_arm_action(action.verb):
                continue
            if action.role not in {"source", "mutual"}:
                continue
            partner = None
            for j, other in enumerate(characters):
                if j == i:
                    continue
                for other_action in other.actions:
                    if other_action.role in {"target", "source", "mutual"} and (
                        other_action.verb == action.verb
                        or _verbs_equivalent(other_action.verb, action.verb)
                    ):
                        partner = other_action
                        break
                if partner is not None:
                    break
            action.role = "target"
            if partner is not None and partner.role in {"target", "mutual"}:
                partner.role = "source"


_STATE_VERBS = frozenset({
    "gagged",
    "gag",
    "bit gag",
    "bamboo gag",
    "shushing",
    "standing",
    "sitting",
    "blushing",
    "blush",
    "bound",
    "tied",
    "tied up",
    "smiling",
    "smile",
    "looking at viewer",
    "arms behind back",
    "finger to mouth",
})
_PAIRABLE_HINTS = (
    "hug",
    "embrace",
    "kiss",
    "sex",
    "penetrat",
    "sitting on",
    "holding",
    "pinning",
    "covering",
    "pressing",
    "talk",
    "lick",
    "grab",
    "carry",
    "straddle",
    "spank",
    "choke",
    "fingering",
    "cunnilingus",
    "fellatio",
    "anal",
    "mocking",
    "watching",
)


def _is_state_verb(verb: str) -> bool:
    text = canonical_action_verb(verb)
    if text in _STATE_VERBS:
        return True
    return text.startswith("gagged") or text.startswith("gag ")


def _is_pairable_verb(verb: str) -> bool:
    if _is_state_verb(verb):
        return False
    text = canonical_action_verb(verb)
    return any(text == hint or text.startswith(hint) or hint in text for hint in _PAIRABLE_HINTS)


def _pair_or_demote_actions(characters: list[CharacterPrompt]) -> None:
    verbs: list[str] = []
    for ch in characters:
        for action in ch.actions:
            if action.verb not in verbs:
                verbs.append(action.verb)
    n = len(characters)
    for verb in verbs:
        owners: list[int] = []
        roles: set[str] = set()
        for i, ch in enumerate(characters):
            for action in ch.actions:
                if action.verb == verb:
                    if i not in owners:
                        owners.append(i)
                    roles.add(action.role)
        if "mutual" in roles:
            continue
        if "source" in roles and "target" in roles:
            continue
        if n == 2 and len(owners) == 1 and _is_pairable_verb(verb):
            other = 1 - owners[0]
            missing = "target" if "source" in roles else "source"
            characters[other].actions.append(Action(role=missing, verb=verb))
            continue
        if not _is_state_verb(verb):
            continue
        for ch in characters:
            kept: list[Action] = []
            for action in ch.actions:
                if action.verb == verb and action.role in {"source", "target"}:
                    if (
                        action.verb
                        and action.verb not in ch.pose
                        and action.verb not in ch.clothing
                        and action.verb not in ch.expression
                    ):
                        ch.pose.append(action.verb)
                else:
                    kept.append(action)
            ch.actions = kept


def _align_action_verbs(characters: list[CharacterPrompt]) -> None:
    verbs = []
    for ch in characters:
        for action in ch.actions:
            canon = canonical_action_verb(action.verb)
            action.verb = canon
            if canon not in verbs:
                verbs.append(canon)
    groups: list[list[str]] = []
    used: set[str] = set()
    for verb in verbs:
        if verb in used:
            continue
        group = [verb]
        used.add(verb)
        for other in verbs:
            if other in used:
                continue
            if any(_verbs_equivalent(other, item) for item in group):
                group.append(other)
                used.add(other)
        groups.append(group)
    rewrite = {}
    for group in groups:
        canonical = max(group, key=len)
        for item in group:
            rewrite[item] = canonical
    for ch in characters:
        for action in ch.actions:
            action.verb = rewrite.get(action.verb, action.verb)


def _collapse_actions(actions: list[Action], warnings: list[str]) -> list[Action]:
    groups: dict[str, list[Action]] = {}
    order: list[str] = []
    for action in actions:
        if action.verb not in groups:
            order.append(action.verb)
            groups[action.verb] = []
        groups[action.verb].append(action)
    out: list[Action] = []
    for verb in order:
        group = groups[verb]
        roles = {item.role for item in group}
        if "source" in roles and "target" in roles:
            rest = [item for item in group if item.role not in ("source", "target")]
            out.append(Action(role="mutual", verb=verb))
            out.extend(rest)
            warnings.append(f"collapsed_to_mutual:{verb}")
        else:
            out.extend(group)
    return out


def assign_unassigned(prompt: Nai5Prompt, tag: str, dest: str) -> Nai5Prompt:
    out = copy.deepcopy(prompt)
    key = normalize_tag(tag)
    out.unassigned = [item for item in out.unassigned if normalize_tag(item) != key]
    if dest == "base":
        if not any(normalize_tag(item) == key for item in out.base.tags):
            out.base.tags.append(tag)
        return out
    matched = _CHAR_DEST_RE.fullmatch(dest)
    if matched is None:
        raise ValueError(f"unknown dest: {dest}")
    index = int(matched.group(1))
    if index >= len(out.characters):
        raise ValueError(f"unknown dest: {dest}")
    appearance = out.characters[index].appearance
    if not any(normalize_tag(item) == key for item in appearance):
        appearance.append(tag)
    return out


def compile(
    draft: SceneDraft,
    tags: list[TagHit] | None = None,
    options: CompileOptions | None = None,
) -> Nai5Prompt:
    if options is None:
        options = CompileOptions()
    include_nl = (
        options.include_nl
        if options.include_nl is not None
        else len(draft.characters) >= 2
    )

    ignored: list[str] = []
    seen_ignored: set[str] = set()
    warnings: list[str] = []

    def note_ignored(token: str) -> None:
        text = token.strip()
        key = normalize_tag(text)
        if not key or key in seen_ignored:
            return
        seen_ignored.add(key)
        ignored.append(text)

    def keep(token: str, *, drop_count: bool, short_only: bool = False) -> str | None:
        text = strip_weight(token)
        if not text:
            return None
        if is_ignored(text):
            note_ignored(text)
            return None
        key = normalize_tag(text)
        if _is_fake_char(key):
            return None
        if drop_count and _is_count_token(key):
            return None
        if short_only and not is_short_tag(text):
            return None
        return text

    def filter_list(
        values: list[str], *, drop_count: bool, short_only: bool = False
    ) -> list[str]:
        out: list[str] = []
        for raw in values:
            kept = keep(raw, drop_count=drop_count, short_only=short_only)
            if kept is not None:
                out.append(kept)
        return out

    def has_norm(values: list[str], wanted: str) -> bool:
        target = normalize_tag(wanted)
        return any(normalize_tag(v) == target for v in values)

    base_tags: list[str] = []
    count_tags: list[str] = []
    for raw in draft.count_tag.split(","):
        kept = keep(raw, drop_count=False)
        if kept is not None:
            if _is_count_token(kept):
                kept = _canonical_count_token(kept)
            count_tags.append(kept)
            base_tags.append(kept)

    for group in (draft.themes, draft.scene, draft.camera):
        base_tags.extend(filter_list(group, drop_count=False, short_only=True))

    has_pov = any(
        strip_weight(token) == _POV
        for ch in draft.characters
        for token in (*ch.pose, *ch.appearance, *ch.clothing)
    )
    if has_pov and not has_norm(base_tags, _FIRST_PERSON_VIEW):
        base_tags.append(_FIRST_PERSON_VIEW)

    if draft.nsfw:
        nsfw = keep("nsfw", drop_count=False)
        if nsfw is not None:
            base_tags.append(nsfw)

    characters: list[CharacterPrompt] = []
    for ch in draft.characters:
        identity = keep(ch.identity, drop_count=True) or ""
        actions: list[Action] = []
        for action in ch.actions:
            verb = strip_weight(action.verb) or action.verb
            actions.append(Action(role=action.role, verb=verb))
        characters.append(
            CharacterPrompt(
                gender=ch.gender,
                identity=identity,
                appearance=filter_list(ch.appearance, drop_count=True, short_only=True),
                clothing=filter_list(ch.clothing, drop_count=True, short_only=True),
                pose=filter_list(ch.pose, drop_count=True, short_only=True),
                expression=filter_list(ch.expression, drop_count=True, short_only=True),
                actions=actions,
            )
        )

    _align_action_verbs(characters)
    _fix_restrained_hug_roles(characters)
    _pair_or_demote_actions(characters)
    for ch in characters:
        ch.actions = _collapse_actions(ch.actions, warnings)

    roles_by_verb: dict[str, set[str]] = {}
    for ch in characters:
        for action in ch.actions:
            roles_by_verb.setdefault(action.verb, set()).add(action.role)
    for verb, roles in roles_by_verb.items():
        if "mutual" in roles:
            continue
        if "source" in roles and "target" in roles:
            continue
        warnings.append(f"unpaired_action:{verb}")

    derived_count = count_tag_from_characters(draft.characters)
    ingested_counts = ", ".join(
        _canonical_count_token(token) for token in count_tags if _is_count_token(token)
    )
    if normalize_tag(derived_count) != normalize_tag(ingested_counts):
        derived_parts = [part.strip() for part in derived_count.split(",") if part.strip()]
        rest = [token for token in base_tags if not _is_count_token(token)]
        base_tags[:] = derived_parts + rest
        warnings.append("count_mismatch")
        meta_count_tag = derived_count
    else:
        meta_count_tag = ingested_counts or ", ".join(count_tags)

    def on_character_lists(ch: CharacterPrompt, key: str) -> bool:
        return any(
            normalize_tag(v) == key
            for group in (ch.appearance, ch.clothing, ch.pose, ch.expression)
            for v in group
        )

    def joined_ac(ch: CharacterPrompt) -> str:
        return " ".join(
            normalize_tag(v) for v in (*ch.appearance, *ch.clothing) if v
        )

    nl_hay = normalize_tag(draft.nl) if draft.nl else ""

    def ensure_base(tag: str) -> None:
        if not has_norm(base_tags, tag):
            base_tags.append(tag)

    unassigned: list[str] = []
    seen_unassigned: set[str] = set()

    def note_unassigned(token: str) -> None:
        key = normalize_tag(token)
        if not key or key in seen_unassigned:
            return
        seen_unassigned.add(key)
        unassigned.append(token)

    hits = tags or []
    cat4_keys: list[str] = []
    for hit in hits:
        if hit.category != 4:
            continue
        raw = hit.name.strip()
        if not raw:
            continue
        key = normalize_tag(raw)
        if not key or _is_fake_char(key):
            continue
        if is_ignored(raw, hit.category):
            continue
        cat4_keys.append(key)

    if cat4_keys:
        identity_warned = False
        for ch in characters:
            if not ch.identity:
                continue
            if any(_identity_hit_matches(ch.identity, key) for key in cat4_keys):
                continue
            ch.identity = ""
            if not identity_warned:
                warnings.append("identity_unconfirmed")
                identity_warned = True

    for hit in hits:
        raw = hit.name.strip()
        if not raw:
            continue
        key = normalize_tag(raw)
        if not key or _is_fake_char(key):
            continue
        if hit.category == 1 or is_ignored(raw, hit.category):
            note_ignored(raw)
            continue
        if key in _NSFW_LIKE:
            ensure_base("nsfw")
            continue
        if _is_count_token(key):
            continue
        if key == _POV:
            if not any(on_character_lists(ch, _POV) for ch in characters):
                ensure_base(_POV)
            continue
        if key in _THEME_TAGS or key in _SCENE_TAGS or key in _CAMERA_TAGS:
            ensure_base(key)
            continue
        if hit.category == 4:
            identity_hits = [
                ch
                for ch in characters
                if ch.identity and _identity_hit_matches(ch.identity, key)
            ]
            if len(identity_hits) != 1:
                note_unassigned(key)
            continue
        exact = [ch for ch in characters if on_character_lists(ch, key)]
        if len(exact) == 1:
            continue
        sub = [ch for ch in characters if key in joined_ac(ch)]
        if len(sub) != 1 and nl_hay and key in nl_hay:
            pool = sub or characters
            if len(pool) == 1:
                sub = pool
        if len(sub) == 1:
            if not has_norm(sub[0].appearance, key):
                sub[0].appearance.append(key)
            continue
        note_unassigned(key)

    appearance_clothing = {
        normalize_tag(v)
        for ch in characters
        for v in (*ch.appearance, *ch.clothing)
    }
    base_tags[:] = [
        token
        for token in base_tags
        if normalize_tag(token) not in appearance_clothing
        and normalize_tag(token) not in _DROP_CAMERA
        and not _is_look_token(token)
    ]

    return Nai5Prompt(
        base=BasePrompt(
            tags=base_tags,
            nl=draft.nl if include_nl else "",
        ),
        characters=characters,
        uc=UcBlock(preset=options.uc_preset, extra=[]),
        unassigned=unassigned,
        ignored=ignored,
        warnings=warnings,
        meta=PromptMeta(
            source="reverse",
            count_tag=meta_count_tag,
            wd14_skipped=False,
            character_count=len(characters),
        ),
    )
