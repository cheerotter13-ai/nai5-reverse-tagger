from nai5_tagger.ignore import is_ignored, normalize_tag
from nai5_tagger.types import SceneDraft, as_token_list


def test_normalize_collapses_underscore_and_case():
    assert normalize_tag("Shiny_Skin") == "shiny skin"


def test_shiny_and_quality_and_artist_ignored():
    assert is_ignored("shiny_skin")
    assert is_ignored("oiled")
    assert is_ignored("masterpiece")
    assert is_ignored("best quality")
    assert is_ignored("some artist", category=1)
    assert not is_ignored("wet clothes")
    assert not is_ignored("yuri")


def test_as_token_list_keeps_sentence_string_intact():
    assert as_token_list("long silver hair with bangs") == ["long silver hair with bangs"]
    assert as_token_list("silver hair, grey eyes") == ["silver hair", "grey eyes"]
    assert as_token_list(["silver hair", "grey eyes"]) == ["silver hair", "grey eyes"]


def test_from_dict_does_not_explode_appearance_string():
    draft = SceneDraft.from_dict(
        {
            "count_tag": "2girls",
            "themes": "yuri, femdom",
            "scene": ["indoors"],
            "camera": "from below",
            "nsfw": True,
            "nl": "",
            "characters": [
                {
                    "gender": "girl",
                    "appearance": "long silver hair with bangs, grey eyes",
                    "clothing": "black dress",
                    "pose": [],
                    "expression": [],
                    "actions": [],
                }
            ],
        }
    )
    assert draft.themes == ["yuri", "femdom"]
    assert draft.camera == ["from below"]
    assert draft.characters[0].appearance == ["long silver hair with bangs", "grey eyes"]
    assert "l" not in draft.characters[0].appearance


def test_from_dict_coerces_boolean_nl_to_empty_string():
    draft = SceneDraft.from_dict(
        {
            "count_tag": "2girls",
            "themes": [],
            "scene": [],
            "camera": [],
            "nsfw": True,
            "nl": True,
            "characters": [{"gender": "girl"}],
        }
    )
    assert draft.nl == ""
    draft_false = SceneDraft.from_dict(
        {
            "count_tag": "1girl",
            "themes": [],
            "scene": [],
            "camera": [],
            "nsfw": False,
            "nl": False,
            "characters": [{"gender": "girl"}],
        }
    )
    assert draft_false.nl == ""
