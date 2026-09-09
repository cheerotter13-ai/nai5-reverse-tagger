import io
import json
from PIL import Image
from nai5_tagger.__main__ import main
from nai5_tagger.types import Nai5Prompt, BasePrompt, CharacterPrompt, UcBlock, PromptMeta


def test_cli_json_and_text(tmp_path, monkeypatch, capsys):
    path = tmp_path / "x.png"
    Image.new("RGB", (8, 8), "red").save(path)
    fake = Nai5Prompt(
        base=BasePrompt(tags=["2girls", "yuri"], nl="nl here"),
        characters=[CharacterPrompt("girl", "", ["silver hair"], [], ["sitting"], ["blush"], [])],
        uc=UcBlock("default", []),
        unassigned=[],
        ignored=[],
        warnings=[],
        meta=PromptMeta("reverse", "2girls", False, 1),
    )
    monkeypatch.setattr("nai5_tagger.__main__.run_pipeline", lambda *a, **k: fake)
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Prompt:")
    assert main([str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["base"]["tags"] == ["2girls", "yuri"]


def test_cli_missing_file_exit_1(capsys):
    assert main(["nope.png"]) == 1
