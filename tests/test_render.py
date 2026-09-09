from nai5_tagger.compiler import compile
from nai5_tagger.render import render
from nai5_tagger.types import CompileOptions
from tests.fixtures.gold_b import GOLD_B


def test_render_gold_b_has_boxes_and_no_char_labels():
    prompt = compile(GOLD_B, tags=[], options=CompileOptions(include_nl=True))
    text = render(prompt)
    assert text.startswith("Prompt:\n")
    assert "\nCharacter 1:\n" in text
    assert "\nCharacter 2:\n" in text
    assert "\nUC:\n" not in text
    assert "source#sitting on person" in text
    assert "target#sitting on person" in text
    assert "char1" not in text.lower().replace("character 1", "")
    assert "lowres" not in text
    lines = text.splitlines()
    prompt_idx = lines.index("Prompt:")
    char_idx = lines.index("Character 1:")
    block = "\n".join(lines[prompt_idx + 1 : char_idx]).strip()
    assert GOLD_B.nl in block


def test_render_omits_nl_when_empty():
    prompt = compile(GOLD_B, tags=[], options=CompileOptions(include_nl=False))
    text = render(prompt)
    assert GOLD_B.nl not in text
