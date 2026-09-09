from nai5_tagger.types import SceneDraft

GOLD_A = SceneDraft.from_dict({
    "count_tag": "1girl, 2boys",
    "themes": ["small penis humiliation", "race contrast", "cuckolding theme"],
    "scene": ["indoors", "pink background"],
    "camera": ["first person view"],
    "nsfw": True,
    "nl": "The girl mocks the viewer while the standing man is behind her.",
    "characters": [
        {
            "gender": "girl",
            "identity": "",
            "appearance": ["shiny skin", "queen of spade symbol", "black tattoos"],
            "clothing": ["extremely revealing sailor uniform", "see-through top"],
            "pose": [],
            "expression": ["evil smile", "full-face blush"],
            "actions": [{"role": "target", "verb": "mocking small penis"}],
        },
        {
            "gender": "boy",
            "identity": "",
            "appearance": ["adult male", "black male", "muscular"],
            "clothing": ["shirtless", "black tight underwear"],
            "pose": [],
            "expression": [],
            "actions": [{"role": "source", "verb": "standing behind female"}],
        },
        {
            "gender": "boy",
            "identity": "",
            "appearance": [],
            "clothing": ["penis chastity"],
            "pose": ["pov"],
            "expression": [],
            "actions": [{"role": "source", "verb": "watching from first person view"}],
        },
    ],
})
