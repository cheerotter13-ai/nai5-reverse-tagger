from nai5_tagger.compiler import assign_unassigned, canonical_action_verb, compile, strip_weight
from nai5_tagger.types import Action, CharacterDraft, CompileOptions, SceneDraft, TagHit
from tests.fixtures.gold_a import GOLD_A
from tests.fixtures.gold_b import GOLD_B


def _all_text(prompt) -> str:
    parts = list(prompt.base.tags)
    for ch in prompt.characters:
        parts.extend([ch.gender, ch.identity, *ch.appearance, *ch.clothing, *ch.pose, *ch.expression])
        parts.extend([f"{a.role}#{a.verb}" for a in ch.actions])
    return ", ".join(p for p in parts if p).lower()


def test_compile_drops_sentence_length_character_tokens():
    draft = SceneDraft(
        count_tag="2girls",
        themes=["yuri"],
        scene=["narrow brick alley at night with large full moon in the sky"],
        camera=["cowboy shot"],
        nsfw=True,
        nl="The bound girl is pressed against the brick wall while the other stands hip-to-hip and covers her mouth.",
        characters=[
            CharacterDraft(
                gender="girl",
                appearance=["pink hair", "standing close with body touching the other character"],
                clothing=["white shorts"],
                pose=["bamboo gag in mouth"],
            ),
            CharacterDraft(
                gender="girl",
                appearance=["pink hair"],
                clothing=["white dress"],
                pose=["finger to mouth"],
            ),
        ],
    )
    prompt = compile(draft, tags=[], options=CompileOptions(include_nl=True))
    blob = _all_text(prompt)
    assert "standing close with body touching the other character" not in blob
    assert "narrow brick alley at night with large full moon in the sky" not in " ".join(prompt.base.tags)
    assert "pink hair" in prompt.characters[0].appearance
    assert "white shorts" in prompt.characters[0].clothing
    assert "pressed against the brick wall" in prompt.base.nl


def test_gold_a_three_characters_split():
    prompt = compile(GOLD_A, tags=[], options=CompileOptions(include_nl=True))
    assert "1girl" in prompt.base.tags
    assert "2boys" in prompt.base.tags
    assert "cuckolding theme" in prompt.base.tags
    assert "first person view" in prompt.base.tags
    assert "nsfw" in prompt.base.tags
    assert len(prompt.characters) == 3
    assert prompt.characters[0].gender == "girl"
    assert prompt.characters[1].gender == "boy"
    assert prompt.characters[2].gender == "boy"
    assert any(a.role == "target" and a.verb == "mocking small penis" for a in prompt.characters[0].actions)
    assert any(a.role == "source" and a.verb == "standing behind female" for a in prompt.characters[1].actions)
    assert any(a.role == "source" and a.verb == "watching from first person view" for a in prompt.characters[2].actions)
    assert "pov" in prompt.characters[2].pose
    blob = _all_text(prompt)
    assert "char1" not in blob
    assert "shiny skin" not in blob
    assert "oiled" not in blob
    assert "masterpiece" not in blob
    for ch in prompt.characters:
        assert "1girl" not in ch.appearance
        assert "2boys" not in ch.appearance


def test_gold_b_source_target_not_crossed():
    prompt = compile(GOLD_B, tags=[])
    assert prompt.base.tags[:2] == ["2girls", "yuri"] or (
        "2girls" in prompt.base.tags and "yuri" in prompt.base.tags
    )
    assert "indoors" in prompt.base.tags
    assert any(a.role == "source" and a.verb == "sitting on person" for a in prompt.characters[0].actions)
    assert any(a.role == "target" and a.verb == "sitting on person" for a in prompt.characters[1].actions)
    assert "long silver hair" in prompt.characters[0].appearance
    assert "long silver hair" not in prompt.characters[1].appearance
    assert "all fours" in prompt.characters[1].pose
    assert "all fours" not in prompt.characters[0].pose


def test_gold_c_silver_hair_stays_on_character_two():
    draft = SceneDraft(
        count_tag="2girls",
        themes=[],
        scene=[],
        camera=[],
        nsfw=False,
        nl="",
        characters=[
            CharacterDraft(gender="girl", appearance=["black hair"]),
            CharacterDraft(gender="girl", appearance=["silver hair"]),
        ],
    )
    tags = [TagHit(name="silver hair", score=0.9, category=0)]
    prompt = compile(draft, tags=tags, options=CompileOptions(include_nl=False))
    assert "silver hair" in prompt.characters[1].appearance
    assert "silver hair" not in prompt.characters[0].appearance
    assert "silver hair" not in prompt.base.tags
    assert "silver hair" not in prompt.unassigned


def test_wd14_explicit_becomes_single_nsfw():
    draft = SceneDraft(
        count_tag="1girl",
        themes=[],
        scene=[],
        camera=[],
        nsfw=False,
        nl="",
        characters=[CharacterDraft(gender="girl")],
    )
    tags = [
        TagHit(name="rating:explicit", score=0.99, category=5),
        TagHit(name="explicit", score=0.8, category=0),
        TagHit(name="nsfw", score=0.7, category=0),
    ]
    prompt = compile(draft, tags=tags)
    assert prompt.base.tags.count("nsfw") == 1
    assert "rating:explicit" not in prompt.base.tags
    assert "explicit" not in prompt.base.tags


def test_wd14_artist_and_masterpiece_ignored():
    draft = SceneDraft(
        count_tag="1girl",
        themes=[],
        scene=["indoors"],
        camera=[],
        nsfw=False,
        nl="",
        characters=[CharacterDraft(gender="girl", appearance=["masterpiece", "red hair"])],
    )
    tags = [
        TagHit(name="artist:foo", score=0.9, category=1),
        TagHit(name="masterpiece", score=0.95, category=0),
        TagHit(name="bookshelf", score=0.6, category=0),
    ]
    prompt = compile(draft, tags=tags)
    blob = _all_text(prompt)
    assert "masterpiece" not in blob
    assert "artist:foo" not in blob
    assert "red hair" in prompt.characters[0].appearance
    assert "bookshelf" in prompt.unassigned
    assert "indoors" in prompt.base.tags


def test_canonical_action_verb_unifies_voice():
    assert canonical_action_verb("sits on face") == canonical_action_verb("gets sat on face")
    assert canonical_action_verb("sits on face") == "sitting on face"
    assert canonical_action_verb("is being sat on face") == "sitting on face"
    assert canonical_action_verb("pins head to floor") == canonical_action_verb("is pinned head to floor")


def test_source_target_paraphrases_are_aligned():
    draft = SceneDraft(
        count_tag="2girls",
        themes=["yuri"],
        scene=[],
        camera=[],
        nsfw=True,
        nl="",
        characters=[
            CharacterDraft(
                gender="girl",
                actions=[Action(role="source", verb="sits on face")],
            ),
            CharacterDraft(
                gender="girl",
                actions=[Action(role="target", verb="gets sat on face")],
            ),
        ],
    )
    prompt = compile(draft, tags=[])
    assert prompt.characters[0].actions[0].verb == prompt.characters[1].actions[0].verb
    assert prompt.characters[0].actions[0].role == "source"
    assert prompt.characters[1].actions[0].role == "target"
    assert prompt.characters[0].actions[0].verb == "sitting on face"
    assert not any(w.startswith("unpaired_action:") for w in prompt.warnings)


def test_unpaired_hug_is_paired_on_the_other_character():
    draft = SceneDraft(
        count_tag="2girls",
        themes=[],
        scene=[],
        camera=[],
        nsfw=False,
        nl="",
        characters=[
            CharacterDraft(gender="girl", actions=[Action(role="source", verb="hug")]),
            CharacterDraft(gender="girl"),
        ],
    )
    prompt = compile(draft, tags=[])
    assert any(a.role == "source" and a.verb == "hug" for a in prompt.characters[0].actions)
    assert any(a.role == "target" and a.verb == "hug" for a in prompt.characters[1].actions)
    assert not any(w.startswith("unpaired_action:") for w in prompt.warnings)


def test_same_character_source_and_target_collapse_to_mutual():
    draft = SceneDraft(
        count_tag="1girl",
        themes=[],
        scene=[],
        camera=[],
        nsfw=False,
        nl="",
        characters=[
            CharacterDraft(
                gender="girl",
                actions=[
                    Action(role="source", verb="hug"),
                    Action(role="target", verb="hug"),
                ],
            )
        ],
    )
    prompt = compile(draft, tags=[])
    roles = [(a.role, a.verb) for a in prompt.characters[0].actions]
    assert roles == [("mutual", "hug")]
    assert any(w.startswith("collapsed_to_mutual:hug") for w in prompt.warnings)


def test_count_mismatch_rewrites_count_tag():
    draft = SceneDraft(
        count_tag="2girls",
        themes=[],
        scene=[],
        camera=[],
        nsfw=False,
        nl="",
        characters=[
            CharacterDraft(gender="girl"),
            CharacterDraft(gender="boy"),
            CharacterDraft(gender="boy"),
        ],
    )
    prompt = compile(draft, tags=[])
    assert "1girl" in prompt.base.tags
    assert "2boys" in prompt.base.tags
    assert "2girls" not in prompt.base.tags
    assert "count_mismatch" in prompt.warnings
    assert prompt.meta.count_tag == "1girl, 2boys"


def test_spaced_count_tokens_compacted_and_fake_char_dropped():
    draft = SceneDraft(
        count_tag="1 girl, 2 boys",
        themes=[],
        scene=[],
        camera=[],
        nsfw=False,
        nl="",
        characters=[
            CharacterDraft(gender="girl", appearance=["char_1"]),
            CharacterDraft(gender="boy"),
            CharacterDraft(gender="boy"),
        ],
    )
    prompt = compile(draft, tags=[])
    assert prompt.base.tags.count("1girl") == 1
    assert prompt.base.tags.count("2boys") == 1
    assert "1girl" in prompt.base.tags
    assert "2boys" in prompt.base.tags
    assert "1 girl" not in prompt.base.tags
    assert "2 boys" not in prompt.base.tags
    blob = _all_text(prompt)
    assert "char_1" not in blob
    assert "char 1" not in blob
    assert "char1" not in blob
    assert prompt.characters[0].appearance == []


def test_identity_dropped_when_wd14_character_conflicts():
    draft = SceneDraft(
        count_tag="1girl",
        themes=[],
        scene=[],
        camera=[],
        nsfw=False,
        nl="",
        characters=[CharacterDraft(gender="girl", identity="bronya zaychik (honkai: star rail)")],
    )
    tags = [TagHit(name="raiden_shogun", score=0.8, category=4)]
    prompt = compile(draft, tags=tags)
    assert prompt.characters[0].identity == ""
    assert "identity_unconfirmed" in prompt.warnings
    assert "raiden shogun" in [t.replace("_", " ") for t in prompt.unassigned] or "raiden_shogun" in prompt.unassigned


def test_two_named_characters_keep_matching_identities():
    draft = SceneDraft(
        count_tag="2girls",
        themes=[],
        scene=[],
        camera=[],
        nsfw=False,
        nl="",
        characters=[
            CharacterDraft(gender="girl", identity="bronya zaychik"),
            CharacterDraft(gender="girl", identity="seele vollerei"),
        ],
    )
    tags = [
        TagHit(name="bronya_zaychik", score=0.8, category=4),
        TagHit(name="seele_vollerei", score=0.8, category=4),
    ]
    prompt = compile(draft, tags=tags)
    assert prompt.characters[0].identity == "bronya zaychik"
    assert prompt.characters[1].identity == "seele vollerei"
    assert "identity_unconfirmed" not in prompt.warnings


def test_identity_kept_when_cat4_contains_identity():
    draft = SceneDraft(
        count_tag="1girl",
        themes=[],
        scene=[],
        camera=[],
        nsfw=False,
        nl="",
        characters=[CharacterDraft(gender="girl", identity="bronya zaychik")],
    )
    tags = [TagHit(name="bronya zaychik (honkai: star rail)", score=0.8, category=4)]
    prompt = compile(draft, tags=tags)
    assert prompt.characters[0].identity == "bronya zaychik"
    assert "identity_unconfirmed" not in prompt.warnings


def test_strip_weight_syntax():
    assert strip_weight("{{{shiny skin}}}") == "shiny skin"
    assert strip_weight("1.2::red hair::") == "red hair"
    assert strip_weight("{evil smile}") == "evil smile"


def test_wd14_substring_ignores_shared_nl_unless_solo():
    tags = [TagHit(name="bookshelf", score=0.6, category=0)]
    duo = SceneDraft(
        count_tag="2girls",
        themes=[],
        scene=[],
        camera=[],
        nsfw=False,
        nl="a bookshelf in the room",
        characters=[
            CharacterDraft(gender="girl", appearance=["red hair"]),
            CharacterDraft(gender="girl", appearance=["black hair"]),
        ],
    )
    duo_prompt = compile(duo, tags=tags)
    assert "bookshelf" in duo_prompt.unassigned
    assert all("bookshelf" not in ch.appearance for ch in duo_prompt.characters)

    solo = SceneDraft(
        count_tag="1girl",
        themes=[],
        scene=[],
        camera=[],
        nsfw=False,
        nl="a bookshelf in the room",
        characters=[CharacterDraft(gender="girl", appearance=["red hair"])],
    )
    solo_prompt = compile(solo, tags=tags)
    assert "bookshelf" in solo_prompt.characters[0].appearance
    assert "bookshelf" not in solo_prompt.unassigned


def test_assign_unassigned_to_character_is_stable():
    draft = SceneDraft(
        count_tag="2girls",
        themes=[],
        scene=[],
        camera=[],
        nsfw=False,
        nl="",
        characters=[CharacterDraft(gender="girl"), CharacterDraft(gender="girl")],
    )
    prompt = compile(draft, tags=[TagHit(name="bookshelf", score=0.5, category=0)])
    assert "bookshelf" in prompt.unassigned
    again = assign_unassigned(prompt, "bookshelf", "char:1")
    assert "bookshelf" in again.characters[1].appearance
    assert "bookshelf" not in again.unassigned
    third = assign_unassigned(again, "bookshelf", "char:1")
    assert third.characters[1].appearance.count("bookshelf") == 1


def test_shared_look_does_not_stay_in_base():
    draft = SceneDraft(
        count_tag="2girls",
        themes=["yuri"],
        scene=["pink hair", "elf ears", "night alley"],
        camera=["front view", "low angle"],
        nsfw=True,
        nl="",
        characters=[
            CharacterDraft(
                gender="girl",
                appearance=["pink hair", "elf ears"],
                clothing=["white corset"],
                pose=["standing"],
            ),
            CharacterDraft(
                gender="girl",
                appearance=["pink hair", "elf ears"],
                clothing=["frilly dress"],
                pose=["standing"],
            ),
        ],
    )
    prompt = compile(draft, tags=[])
    assert "pink hair" not in prompt.base.tags
    assert "elf ears" not in prompt.base.tags
    assert "night alley" in prompt.base.tags
    assert "low angle" not in prompt.base.tags
    assert "yuri" in prompt.base.tags


def test_distinct_actions_stay_on_owner():
    draft = SceneDraft(
        count_tag="2girls",
        themes=["yuri"],
        scene=[],
        camera=["cowboy shot"],
        nsfw=True,
        nl="",
        characters=[
            CharacterDraft(gender="girl", actions=[Action(role="target", verb="gagged")]),
            CharacterDraft(gender="girl", actions=[Action(role="source", verb="shushing")]),
        ],
    )
    prompt = compile(draft, tags=[])
    assert all(a.verb != "gagged" for a in prompt.characters[0].actions)
    assert all(a.verb != "shushing" for a in prompt.characters[1].actions)
    assert "gagged" in prompt.characters[0].pose
    assert "shushing" in prompt.characters[1].pose
    assert all(a.verb != "shushing" for a in prompt.characters[0].actions)
    assert all(a.verb != "gagged" for a in prompt.characters[1].actions)
    assert not any(f"{a.role}#{a.verb}" == "target#gagged" for a in prompt.characters[0].actions)


def test_paired_gag_person_is_still_a_state_not_an_action():
    draft = SceneDraft(
        count_tag="2girls",
        themes=[],
        scene=[],
        camera=[],
        nsfw=True,
        nl="",
        characters=[
            CharacterDraft(
                gender="girl",
                pose=["arms behind back"],
                clothing=["bamboo gag"],
                actions=[
                    Action(role="target", verb="hug person"),
                    Action(role="target", verb="gag person"),
                ],
            ),
            CharacterDraft(
                gender="girl",
                clothing=["hat"],
                actions=[
                    Action(role="source", verb="hug person"),
                    Action(role="source", verb="gag person"),
                ],
            ),
        ],
    )
    prompt = compile(draft, tags=[])
    assert any(a.role == "target" and a.verb.startswith("hug") for a in prompt.characters[0].actions)
    assert any(a.role == "source" and a.verb.startswith("hug") for a in prompt.characters[1].actions)
    assert all("gag" not in a.verb for a in prompt.characters[0].actions)
    assert all("gag" not in a.verb for a in prompt.characters[1].actions)


def test_restrained_character_is_hug_target_not_source():
    draft = SceneDraft(
        count_tag="2girls",
        themes=["yuri"],
        scene=[],
        camera=["cowboy shot"],
        nsfw=True,
        nl="",
        characters=[
            CharacterDraft(
                gender="girl",
                pose=["arms behind back"],
                clothing=["shibari over clothes"],
                actions=[Action(role="source", verb="hug person")],
            ),
            CharacterDraft(
                gender="girl",
                pose=["finger to mouth"],
                clothing=["hat"],
                actions=[Action(role="target", verb="hug person")],
            ),
        ],
    )
    prompt = compile(draft, tags=[])
    assert any(a.role == "target" and a.verb.startswith("hug") for a in prompt.characters[0].actions)
    assert any(a.role == "source" and a.verb.startswith("hug") for a in prompt.characters[1].actions)

