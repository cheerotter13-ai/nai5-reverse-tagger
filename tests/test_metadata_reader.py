import json

from nai5_tagger.metadata_reader import read_metadata
from tests.fixtures.make_nai_png import write_comment_png


def test_read_v4_prompt_comment(tmp_path):
    payload = {
        "prompt": "ignored flat prompt",
        "uc": "lowres",
        "v4_prompt": {
            "caption": {
                "base_caption": "2girls, yuri, indoors",
                "char_captions": [
                    {"char_caption": "girl, silver hair, source#hug"},
                    {"char_caption": "girl, black hair, target#hug"},
                ],
            }
        },
    }
    path = write_comment_png(tmp_path / "nai.png", payload)
    result = read_metadata(path)
    assert result is not None
    assert result.base_caption == "2girls, yuri, indoors"
    assert result.char_captions == [
        "girl, silver hair, source#hug",
        "girl, black hair, target#hug",
    ]
    assert result.uc == "lowres"


def test_non_nai_png_returns_none(tmp_path):
    from PIL import Image
    path = tmp_path / "plain.png"
    Image.new("RGB", (8, 8), "blue").save(path)
    assert read_metadata(path) is None


def test_missing_base_caption_is_none(tmp_path):
    path = write_comment_png(tmp_path / "bad.png", {"v4_prompt": {"caption": {"base_caption": "", "char_captions": []}}})
    assert read_metadata(path) is None


def test_empty_char_captions_hits(tmp_path):
    payload = {
        "v4_prompt": {
            "caption": {
                "base_caption": "1girl, indoors",
                "char_captions": [],
            }
        },
        "uc": "lowres",
    }
    result = read_metadata(write_comment_png(tmp_path / "solo.png", payload))
    assert result is not None
    assert result.base_caption == "1girl, indoors"
    assert result.char_captions == []
    assert result.uc == "lowres"


def test_uc_from_undesired_content(tmp_path):
    payload = {
        "undesired_content": "blurry",
        "v4_prompt": {
            "caption": {"base_caption": "1girl", "char_captions": []},
        },
    }
    result = read_metadata(write_comment_png(tmp_path / "uc.png", payload))
    assert result is not None
    assert result.uc == "blurry"


def test_uc_from_v4_negative_prompt(tmp_path):
    payload = {
        "v4_prompt": {
            "caption": {"base_caption": "1girl", "char_captions": []},
        },
        "v4_negative_prompt": {
            "caption": {"base_caption": "lowres, worst quality", "char_captions": []},
        },
    }
    result = read_metadata(write_comment_png(tmp_path / "neg.png", payload))
    assert result is not None
    assert result.uc == "lowres, worst quality"


def test_stealth_pngcomp(tmp_path):
    from tests.fixtures.make_nai_png import write_stealth_png

    payload = {
        "uc": "lowres",
        "v4_prompt": {
            "caption": {
                "base_caption": "2girls, yuri, indoors",
                "char_captions": [
                    {"char_caption": "girl, silver hair, source#hug"},
                    {"char_caption": "girl, black hair, target#hug"},
                ],
            }
        },
    }
    path = write_stealth_png(tmp_path / "stealth.png", payload, compressed=True)
    result = read_metadata(path)
    assert result is not None
    assert result.base_caption == "2girls, yuri, indoors"
    assert result.char_captions == [
        "girl, silver hair, source#hug",
        "girl, black hair, target#hug",
    ]
    assert result.uc == "lowres"


def test_stealth_pnginfo(tmp_path):
    from tests.fixtures.make_nai_png import write_stealth_png

    payload = {
        "v4_prompt": {
            "caption": {"base_caption": "1girl, sitting", "char_captions": []},
        }
    }
    result = read_metadata(write_stealth_png(tmp_path / "info.png", payload, compressed=False))
    assert result is not None
    assert result.base_caption == "1girl, sitting"
    assert result.char_captions == []


def test_stealth_comment_wrapper(tmp_path):
    from tests.fixtures.make_nai_png import write_stealth_png

    inner = {
        "uc": "lowres",
        "v4_prompt": {
            "caption": {"base_caption": "1girl", "char_captions": []},
        },
    }
    wrapper = {"Software": "NovelAI", "Comment": json.dumps(inner)}
    result = read_metadata(write_stealth_png(tmp_path / "wrap.png", wrapper, compressed=True))
    assert result is not None
    assert result.base_caption == "1girl"
    assert result.uc == "lowres"


def test_webp_usercomment(tmp_path):
    from tests.fixtures.make_nai_png import write_webp_usercomment

    payload = {
        "uc": "lowres",
        "v4_prompt": {
            "caption": {"base_caption": "1girl, outdoors", "char_captions": []},
        },
    }
    result = read_metadata(write_webp_usercomment(tmp_path / "nai.webp", payload))
    assert result is not None
    assert result.base_caption == "1girl, outdoors"
    assert result.char_captions == []
    assert result.uc == "lowres"


def test_unreadable_file_returns_none(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    assert read_metadata(path) is None
