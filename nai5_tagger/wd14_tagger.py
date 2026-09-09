from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

from nai5_tagger.types import TagHit

_DEFAULT_DIR = ""
_IMAGE_SIZE = 448
_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

_session = None
_tag_rows: list[list[str]] | None = None


def tag_image(
    image,
    *,
    session=None,
    tag_rows=None,
    threshold: float = 0.35,
) -> list[TagHit]:
    try:
        sess = session if session is not None else _load_session()
        rows = tag_rows if tag_rows is not None else _load_tag_rows()
        inp = sess.get_inputs()[0]
        tensor = _preprocess(image, inp)
        scores = np.asarray(sess.run(None, {inp.name: tensor})[0]).reshape(-1)
        hits: list[TagHit] = []
        for row, score in zip(rows, scores):
            value = float(score)
            if value < threshold:
                continue
            name = row[1]
            hits.append(
                TagHit(
                    name=name.replace("_", " ") if len(name) > 3 else name,
                    score=value,
                    category=int(row[2]),
                )
            )
        return hits
    except Exception:
        return []


def _model_dir() -> Path:
    return Path(os.environ.get("NAI5_TAGGER_WD14_DIR", _DEFAULT_DIR).strip())


def _load_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        path = str(_model_dir() / "model.onnx")
        _session = ort.InferenceSession(path, providers=_PROVIDERS)
    return _session


def _load_tag_rows() -> list[list[str]]:
    global _tag_rows
    if _tag_rows is None:
        path = _model_dir() / "selected_tags.csv"
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            _tag_rows = [row for row in reader if row]
    return _tag_rows


def _input_hw(inp) -> tuple[int, int, bool]:
    """Return (height, width, nchw). FakeSession has no shape → 448 NHWC."""
    shape = getattr(inp, "shape", None)
    if not shape or len(shape) != 4:
        return _IMAGE_SIZE, _IMAGE_SIZE, False
    dims = [d if isinstance(d, int) else None for d in shape]
    if dims[1] == 3 and dims[3] != 3:
        h = dims[2] if dims[2] and dims[2] > 0 else _IMAGE_SIZE
        w = dims[3] if dims[3] and dims[3] > 0 else _IMAGE_SIZE
        return h, w, True
    h = dims[1] if dims[1] and dims[1] > 0 else _IMAGE_SIZE
    w = dims[2] if dims[2] and dims[2] > 0 else _IMAGE_SIZE
    return h, w, False


def _preprocess(image, inp) -> np.ndarray:
    height, width, nchw = _input_hw(inp)
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)[:, :, ::-1].copy()  # RGB → BGR, match lora-scripts / EVA02
    h, w = arr.shape[:2]
    side = max(h, w)
    pad_x = side - w
    pad_y = side - h
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    arr = np.pad(
        arr,
        ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    resized = Image.fromarray(arr).resize((width, height), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32)
    if nchw:
        arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...])
