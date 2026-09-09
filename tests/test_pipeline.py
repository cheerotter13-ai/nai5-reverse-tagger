from PIL import Image
from nai5_tagger.pipeline import PipelineError, run_pipeline
from nai5_tagger.types import CompileOptions, MetadataResult, SceneDraft, TagHit
from tests.fixtures.gold_b import GOLD_B
from tests.fixtures.make_nai_png import write_comment_png


def test_metadata_short_circuit_skips_models(tmp_path):
    path = write_comment_png(
        tmp_path / "nai.png",
        {"uc": "lowres", "v4_prompt": {"caption": {"base_caption": "2girls, yuri", "char_captions": [{"char_caption": "girl, silver hair"}]}}},
    )
    calls = {"wd14": 0, "vision": 0}

    def wd14_fn(img):
        calls["wd14"] += 1
        return []

    def vision_fn(data):
        calls["vision"] += 1
        return GOLD_B

    prompt = run_pipeline(path, wd14_fn=wd14_fn, vision_fn=vision_fn)
    assert prompt.meta.source == "metadata"
    assert calls == {"wd14": 0, "vision": 0}
    assert "2girls" in prompt.base.tags
    assert prompt.characters[0].appearance


def test_grok_error_is_hard_failure(tmp_path):
    path = tmp_path / "x.png"
    Image.new("RGB", (8, 8), "red").save(path)

    def vision_fn(data):
        from nai5_tagger.grok_vision import GrokVisionError
        raise GrokVisionError("gateway down")

    try:
        run_pipeline(path, metadata_fn=lambda p: None, wd14_fn=lambda img: [], vision_fn=vision_fn)
        assert False, "expected PipelineError"
    except PipelineError as exc:
        assert exc.exit_code == 2
        assert "gateway down" in exc.message


def test_wd14_error_sets_skipped(tmp_path):
    path = tmp_path / "x.png"
    Image.new("RGB", (8, 8), "red").save(path)

    def wd14_fn(img):
        raise RuntimeError("onnx")

    prompt = run_pipeline(
        path,
        metadata_fn=lambda p: None,
        wd14_fn=wd14_fn,
        vision_fn=lambda data: GOLD_B,
        options=CompileOptions(include_nl=False),
    )
    assert prompt.meta.source == "reverse"
    assert prompt.meta.wd14_skipped is True
    assert len(prompt.characters) == 2
