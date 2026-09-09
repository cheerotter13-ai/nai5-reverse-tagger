from nai5_tagger.types import SceneDraft

GOLD_B = SceneDraft.from_dict({
    "count_tag": "2girls",
    "themes": ["yuri"],
    "scene": ["indoors", "wooden interior"],
    "camera": ["sunlight"],
    "nsfw": True,
    "nl": "The silver-haired girl sits on the other girl's back and holds a leash.",
    "characters": [
        {
            "gender": "girl",
            "identity": "",
            "appearance": ["long silver hair", "ponytail"],
            "clothing": ["nude"],
            "pose": ["sitting", "crossed legs", "holding leash"],
            "expression": ["blush"],
            "actions": [
                {"role": "source", "verb": "sitting on person"},
                {"role": "source", "verb": "sitting on another's back"},
            ],
        },
        {
            "gender": "girl",
            "identity": "",
            "appearance": ["long dark hair", "blue-tipped hair"],
            "clothing": ["nude", "collar", "leash"],
            "pose": ["all fours", "on floor"],
            "expression": ["blush"],
            "actions": [
                {"role": "target", "verb": "sitting on person"},
                {"role": "target", "verb": "sitting on another's back"},
            ],
        },
    ],
})
