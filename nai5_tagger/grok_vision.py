from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

from PIL import Image

from nai5_tagger.types import SceneDraft

_DEFAULT_BASE = "http://127.0.0.1:8000/v1"
_DEFAULT_MODEL = "grok-4.6"
_CHAT_TIMEOUT = 120
_READY_TIMEOUT = 5
_RETRY_SLEEP = 5
_MAX_SIDE = 1536
_ERR_BODY_LIMIT = 500

SYSTEM_PROMPT = (
    "You are a vision tagger for NovelAI Diffusion V5 multi-character prompts. "
    "Output only one JSON object that matches the SceneDraft schema. "
    "No markdown, no commentary, no extra keys. "
    "Describe 2D/anime NSFW honestly from the image. "
    "Do not refuse, sanitize, or rewrite the scene as SFW. "
    "Character appearance, clothing, pose, and expression MUST be Danbooru short tags: "
    "1 to 4 words each, lowercase, like pink hair, white thighhighs, finger to mouth, "
    "bit gag, shibari over clothes. Never English sentences in those lists. "
    "Never write phrases such as standing close with body touching the other character. "
    "Split appearance, clothing, pose, expression, and actions per person. "
    "Interactions use actions with role source, target, or mutual. "
    "verb is a lowercase English phrase without a role prefix. "
    "The source and target of one interaction MUST share the identical verb, "
    "for example source#sitting on person and target#sitting on person. "
    "Do not paraphrase (never sits on vs gets sat on, never pins vs is pinned). "
    "nl is exactly one English sentence. Use nl for complex interactions that short tags "
    "cannot express well, and for spatial scene interaction: who is in front or behind, "
    "who is pressed to a wall, who sits on whose face, grip, insertion, and contact. "
    "Do not restate the tag list in nl. Do not write yes, none, or empty filler. "
    "Solo images may leave nl empty. "
    "If a character is the camera viewpoint, mark pov on that character "
    "and put first person view in camera. "
    "camera crop is one of: cowboy shot, upper body, full body, lower body. "
    "Use cowboy shot when torso and thighs are visible and feet are cropped. "
    "Two people standing mid-thigh is cowboy shot, never lower body, never low angle. "
    "from below only if the camera is truly under the subject looking up. "
    "A standing two-shot is not from below. "
    "Hair, ears, eyes, and skin stay in that character object, never in scene. "
    "Each character's clothing list is only that person's clothes. "
    "scene tags stay short (night, alley, full moon), not a paragraph. "
    "Do not output artist names, quality or aesthetic tags, "
    "or glossy/oiled material tags. "
    "Do not output numbered character-box labels. "
    "If a copyrighted character identity is uncertain, leave identity empty "
    "and describe appearance instead."
)

USER_PROMPT = (
    "Analyze this image and fill one JSON object with these SceneDraft keys: "
    "count_tag, themes, scene, camera, nsfw, nl, characters. "
    "appearance, clothing, pose, and expression are Danbooru short tag lists. "
    "nl is one English sentence for spatial layout and complex interaction only. "
    "Each character object uses: gender (girl|boy|other), identity, appearance, "
    "clothing, pose, expression, actions. "
    "Each action uses: role (source|target|mutual), verb."
)


class GrokVisionError(Exception):
    pass


def gateway_ready(base: str | None = None, opener=None, api_key: str | None = None) -> bool:
    if opener is None:
        opener = urllib.request.urlopen
    if api_key is None:
        api_key = os.environ.get("NAI5_TAGGER_VLM_KEY", "")
    url = _join(_resolve_base(base), "models")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        status, _body = _call(opener, req, _READY_TIMEOUT)
    except OSError:
        return False
    return status == 200


def analyze_image(
    image_bytes: bytes,
    *,
    base: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    opener=None,
) -> SceneDraft:
    if opener is None:
        opener = urllib.request.urlopen
    if model is None:
        model = os.environ.get("NAI5_TAGGER_VLM_MODEL", _DEFAULT_MODEL)
    if api_key is None:
        api_key = os.environ.get("NAI5_TAGGER_VLM_KEY", "")
    url = _join(_resolve_base(base), "chat/completions")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": _jpeg_data_url(image_bytes)},
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        status, body = _call(opener, req, _CHAT_TIMEOUT)
        if status == 429:
            time.sleep(_RETRY_SLEEP)
            status, body = _call(opener, req, _CHAT_TIMEOUT)
    except OSError as err:
        raise GrokVisionError(str(err)) from err
    if status == 401:
        raise GrokVisionError("check NAI5_TAGGER_VLM_KEY")
    if status != 200:
        raise GrokVisionError(_truncate_body(body))
    return _draft_from_response(body)


def _resolve_base(base: str | None) -> str:
    if base is None:
        return os.environ.get("NAI5_TAGGER_VLM_BASE", _DEFAULT_BASE)
    return base


def _join(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _call(opener, req, timeout: int) -> tuple[int, bytes]:
    try:
        resp = opener(req, timeout=timeout)
    except urllib.error.HTTPError as err:
        status = getattr(err, "status", None) or err.code
        try:
            body = err.read()
        except OSError:
            body = b""
        return int(status), body
    enter = getattr(resp, "__enter__", None)
    if enter is not None:
        with resp as opened:
            return opened.status, opened.read()
    return resp.status, resp.read()


def _jpeg_data_url(image_bytes: bytes) -> str:
    img = None
    try:
        with Image.open(io.BytesIO(image_bytes)) as src:
            src.load()
            img = src.convert("RGB")
    except OSError:
        img = None
    if img is None:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    img.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _truncate_body(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")[:_ERR_BODY_LIMIT]


def _draft_from_response(body: bytes) -> SceneDraft:
    try:
        envelope = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise GrokVisionError("response is not JSON") from err
    try:
        content = envelope["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as err:
        raise GrokVisionError("missing choices message content") from err
    if isinstance(content, dict):
        obj = content
    elif isinstance(content, str):
        obj = _loads_object(content)
    else:
        raise GrokVisionError("missing choices message content")
    try:
        return SceneDraft.from_dict(obj)
    except (KeyError, TypeError, ValueError) as err:
        raise GrokVisionError("invalid SceneDraft") from err


def _unwrap_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _loads_object(text: str) -> dict:
    text = _unwrap_fence(text)
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        idx = text.find("{")
        if idx >= 0:
            try:
                obj, _end = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        raise GrokVisionError("response is not JSON")
    return obj
