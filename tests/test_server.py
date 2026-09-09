import json
import threading
from http.client import HTTPConnection
from nai5_tagger.server import URL, main, make_server
from nai5_tagger.types import Nai5Prompt, BasePrompt, CharacterPrompt, UcBlock, PromptMeta, Action


def _fake_prompt():
    return Nai5Prompt(
        base=BasePrompt(tags=["2girls", "yuri"], nl="nl"),
        characters=[
            CharacterPrompt("girl", "", ["silver hair"], ["nude"], ["sitting"], [], [Action("source", "sit")]),
            CharacterPrompt("girl", "", ["black hair"], ["nude"], ["all fours"], [], [Action("target", "sit")]),
        ],
        uc=UcBlock("default", []),
        unassigned=["bookshelf"],
        ignored=["masterpiece"],
        warnings=[],
        meta=PromptMeta("reverse", "2girls", False, 2),
    )


def test_ready_and_reverse_and_assign(monkeypatch, tmp_path):
    fake = _fake_prompt()
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake

    monkeypatch.setattr("nai5_tagger.server.run_pipeline", fake_run)
    monkeypatch.setattr("nai5_tagger.server.gateway_ready", lambda *args, **kwargs: True)
    httpd = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    conn = HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/api/ready")
    ready = json.loads(conn.getresponse().read())
    assert ready["ok"] is True
    conn.request("GET", "/api/config")
    cfg = json.loads(conn.getresponse().read())
    assert "base" in cfg
    assert "model" in cfg
    assert "has_key" in cfg
    conn.request("POST", "/api/ready", body=b'{"base":"https://example.test/v1","model":"x"}', headers={"Content-Type": "application/json"})
    posted = json.loads(conn.getresponse().read())
    assert posted["ok"] is True
    img = tmp_path / "x.png"
    from PIL import Image
    Image.new("RGB", (8, 8), "red").save(img)
    boundary = "----x"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\nContent-Type: image/png\r\n\r\n"
    ).encode() + img.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    conn.request("POST", "/api/reverse", body=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    reversed_json = json.loads(conn.getresponse().read())
    assert reversed_json["prompt"]["unassigned"] == ["bookshelf"]
    assert "Character 1:" in reversed_json["text"]
    options = captured["args"][1] if len(captured.get("args") or ()) > 1 else captured.get("kwargs", {}).get("options")
    assert options is not None
    assert options.include_nl is None
    conn.request(
        "POST",
        "/api/assign",
        body=json.dumps({"prompt": reversed_json["prompt"], "tag": "bookshelf", "dest": "char:0"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    assigned = json.loads(conn.getresponse().read())
    assert "bookshelf" in assigned["prompt"]["characters"][0]["appearance"]
    assert "template" not in reversed_json
    assert "slots" not in reversed_json
    httpd.shutdown()


def test_main_opens_browser_after_bind(monkeypatch):
    opened = []

    class FakeHttpd:
        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr("nai5_tagger.server._port_in_use", lambda host, port: False)
    monkeypatch.setattr("nai5_tagger.server.make_server", lambda host, port: FakeHttpd())
    monkeypatch.setattr("nai5_tagger.server.webbrowser.open", lambda url: opened.append(url))
    assert main() == 0
    assert opened == [URL]


def test_main_port_busy_skips_browser(monkeypatch, capsys):
    opened = []
    monkeypatch.setattr("nai5_tagger.server._port_in_use", lambda host, port: True)
    monkeypatch.setattr("nai5_tagger.server.webbrowser.open", lambda url: opened.append(url))
    assert main() == 0
    assert opened == []
    assert URL in capsys.readouterr().out
