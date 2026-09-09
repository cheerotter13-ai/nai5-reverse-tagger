from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ExifTags import IFD

from nai5_tagger.types import MetadataResult

_MAGIC_COMP = b"stealth_pngcomp"
_MAGIC_INFO = b"stealth_pnginfo"
_USER_COMMENT = 0x9286


def read_metadata(path: str | Path) -> MetadataResult | None:
    try:
        with Image.open(path) as image:
            image.load()
            fmt = image.format
            if fmt == "PNG" and (
                image.info.get("Software") == "NovelAI" or "Comment" in image.info
            ):
                hit = _from_json_text(image.info.get("Comment"))
                if hit is not None:
                    return hit
            hit = _from_stealth(image)
            if hit is not None:
                return hit
            if fmt == "WEBP":
                hit = _from_webp_exif(image)
                if hit is not None:
                    return hit
            return None
    except OSError:
        return None


def _from_json_text(text: object) -> MetadataResult | None:
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(text, str) or not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _from_decoded_obj(obj)


def _from_decoded_obj(obj: object) -> MetadataResult | None:
    if not isinstance(obj, dict):
        return None
    hit = _from_nai_params(obj)
    if hit is not None:
        return hit
    comment = obj.get("Comment")
    if isinstance(comment, str):
        try:
            comment = json.loads(comment)
        except json.JSONDecodeError:
            return None
    if isinstance(comment, dict):
        return _from_nai_params(comment)
    return None


def _from_nai_params(obj: dict) -> MetadataResult | None:
    v4 = obj.get("v4_prompt")
    if not isinstance(v4, dict):
        return None
    caption = v4.get("caption")
    if not isinstance(caption, dict):
        return None
    base = caption.get("base_caption")
    chars = caption.get("char_captions")
    if not isinstance(base, str) or not base or not isinstance(chars, list):
        return None
    mapped: list[str] = []
    for item in chars:
        if isinstance(item, dict):
            cap = item.get("char_caption")
            mapped.append(cap if isinstance(cap, str) else "")
        else:
            mapped.append("")
    return MetadataResult(base_caption=base, char_captions=mapped, uc=_uc_from(obj))


def _uc_from(obj: dict) -> str:
    for key in ("uc", "undesired_content"):
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val
    v4n = obj.get("v4_negative_prompt")
    if isinstance(v4n, dict):
        cap = v4n.get("caption")
        if isinstance(cap, dict):
            base = cap.get("base_caption")
            if isinstance(base, str) and base:
                return base
    return ""


def _from_stealth(image: Image.Image) -> MetadataResult | None:
    try:
        data = _alpha_lsb_bytes(image)
        if not data:
            return None
        return _parse_stealth_bytes(data)
    except Exception:
        return None


def _alpha_lsb_bytes(image: Image.Image) -> bytes:
    if "A" not in image.mode:
        return b""
    alpha = np.asarray(image.convert("RGBA"))[..., 3]
    # Transpose: x-outer / y-inner, same as NovelAI stealth_pngcomp (alpha.T flatten).
    bits = np.bitwise_and(alpha.T.reshape(-1), 1)
    bits = bits[: bits.size // 8 * 8]
    return np.packbits(bits).tobytes()


def _parse_stealth_bytes(data: bytes) -> MetadataResult | None:
    if data.startswith(_MAGIC_COMP):
        raw = _stealth_body(data[len(_MAGIC_COMP) :], compressed=True)
    elif data.startswith(_MAGIC_INFO):
        raw = _stealth_body(data[len(_MAGIC_INFO) :], compressed=False)
    else:
        return None
    if not raw:
        return None
    return _from_json_text(raw.decode("utf-8"))


def _stealth_body(rest: bytes, *, compressed: bool) -> bytes | None:
    raw: bytes | None = None
    if len(rest) >= 4:
        nbits = int.from_bytes(rest[:4], "big")
        if nbits % 8 == 0:
            nbytes = nbits // 8
            if 0 < nbytes <= len(rest) - 4:
                raw = rest[4 : 4 + nbytes]
    if raw is None:
        raw = rest if compressed else rest.split(b"\x00", 1)[0]
    if compressed:
        raw = gzip.decompress(raw)
    return raw


def _from_webp_exif(image: Image.Image) -> MetadataResult | None:
    try:
        exif = image.getexif()
    except Exception:
        return None
    if not exif:
        return None
    values: list[object] = []
    uc = exif.get(_USER_COMMENT)
    if uc is not None:
        values.append(uc)
    try:
        ifd = exif.get_ifd(IFD.Exif)
        nested = ifd.get(_USER_COMMENT) if ifd else None
        if nested is not None:
            values.append(nested)
    except Exception:
        pass
    for value in values:
        hit = _from_json_text(_decode_user_comment(value))
        if hit is not None:
            return hit
    return None


def _decode_user_comment(value: object) -> str | None:
    if isinstance(value, str):
        if len(value) >= 8 and value[:8].rstrip("\x00") in {"ASCII", "UNICODE", "JIS"}:
            return value[8:]
        return value
    if not isinstance(value, bytes):
        return None
    if len(value) >= 8 and (
        value.startswith(b"ASCII")
        or value.startswith(b"UNICODE")
        or value.startswith(b"JIS")
        or value[:8] == b"\x00" * 8
    ):
        body = value[8:]
        if value.startswith(b"UNICODE"):
            try:
                return body.decode("utf-16")
            except UnicodeDecodeError:
                return None
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return None
