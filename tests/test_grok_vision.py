import io
import json
from nai5_tagger.grok_vision import GrokVisionError, analyze_image, gateway_ready


class FakeResp:
    def __init__(self, status, body, headers=None):
        self.status = status
        self.body = body.encode() if isinstance(body, str) else body
        self.headers = headers or {}

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_analyze_image_parses_json_content():
    payload = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "count_tag": "2girls",
                    "themes": ["yuri"],
                    "scene": ["indoors"],
                    "camera": [],
                    "nsfw": True,
                    "nl": "two girls",
                    "characters": [
                        {"gender": "girl", "appearance": ["silver hair"], "actions": [{"role": "source", "verb": "sit"}]},
                        {"gender": "girl", "appearance": ["black hair"], "actions": [{"role": "target", "verb": "sit"}]},
                    ],
                })
            }
        }]
    }
    calls = []

    def opener(req, timeout=0):
        calls.append(json.loads(req.data.decode()))
        return FakeResp(200, json.dumps(payload))

    draft = analyze_image(b"\x89PNG", opener=opener)
    assert draft.count_tag == "2girls"
    assert draft.characters[0].actions[0].role == "source"
    assert calls[0]["model"] == "grok-4.6"
    assert "char1" not in json.dumps(calls[0]["messages"][0])
    assert "shiny skin" not in json.dumps(calls[0]["messages"][0]).lower()


def test_analyze_image_retries_once_on_429():
    payload = {
        "choices": [{"message": {"content": json.dumps({
            "count_tag": "1girl", "themes": [], "scene": [], "camera": [], "nsfw": False, "nl": "",
            "characters": [{"gender": "girl"}],
        })}}]
    }
    states = {"n": 0}

    def opener(req, timeout=0):
        states["n"] += 1
        if states["n"] == 1:
            return FakeResp(429, '{"error":"rate"}')
        return FakeResp(200, json.dumps(payload))

    draft = analyze_image(b"abc", opener=opener)
    assert draft.characters[0].gender == "girl"
    assert states["n"] == 2


def test_analyze_image_non_json_raises():
    def opener(req, timeout=0):
        return FakeResp(200, json.dumps({"choices": [{"message": {"content": "not json"}}]}))

    try:
        analyze_image(b"abc", opener=opener)
        assert False, "expected GrokVisionError"
    except GrokVisionError:
        pass


def test_gateway_ready_false_on_error():
    def opener(req, timeout=0):
        raise OSError("down")

    assert gateway_ready("http://127.0.0.1:8000/v1", opener=opener) is False


def test_gateway_ready_sends_authorization(monkeypatch):
    seen = {}

    def opener(req, timeout=0):
        seen["auth"] = req.headers.get("Authorization") or req.get_header("Authorization")
        return FakeResp(200, '{"data":[]}')

    monkeypatch.setenv("NAI5_TAGGER_VLM_KEY", "test-key")
    assert gateway_ready("http://127.0.0.1:8000/v1", opener=opener) is True
    assert "Bearer test-key" in str(seen["auth"])


def test_system_prompt_requires_cowboy_crop_not_low_angle():
    from nai5_tagger.grok_vision import SYSTEM_PROMPT

    text = SYSTEM_PROMPT.lower()
    assert "cowboy shot" in text
    assert "never low angle" in text or "never lower body, never low angle" in text
    assert "hair, ears, eyes" in text
    assert "char1" not in SYSTEM_PROMPT
    assert "shiny skin" not in text


def test_system_prompt_uses_danbooru_skeleton_and_spatial_nl():
    from nai5_tagger.grok_vision import SYSTEM_PROMPT, USER_PROMPT

    text = (SYSTEM_PROMPT + " " + USER_PROMPT).lower()
    assert "danbooru" in text
    assert "short tag" in text
    assert "one english sentence" in text or "exactly one" in text
    assert "spatial" in text
    assert "complex" in text
    assert "do not restate" in text or "not restate" in text
