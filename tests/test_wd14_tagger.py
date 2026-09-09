from PIL import Image
from nai5_tagger.wd14_tagger import tag_image
from nai5_tagger.types import TagHit


class FakeSession:
    def get_inputs(self):
        return [type("I", (), {"name": "input"})()]

    def run(self, _, feeds):
        return [[[0.9, 0.8, 0.2]]]


def test_threshold_and_categories():
    rows = [
        ["0", "red_hair", "0"],
        ["1", "some_artist", "1"],
        ["2", "indoors", "0"],
    ]
    img = Image.new("RGB", (16, 16), "red")
    hits = tag_image(img, session=FakeSession(), tag_rows=rows, threshold=0.35)
    names = {h.name: h for h in hits}
    assert "red hair" in names or "red_hair" in names
    assert names[list(names)[0]].category == 0 or names["red hair"].category == 0
    assert all(h.score >= 0.35 for h in hits)
    assert not any(h.category == 1 and h.score < 0.35 for h in hits)
    assert any(h.category == 1 for h in hits)  # artist still returned; compiler ignores


def test_session_error_returns_empty():
    class Boom:
        def get_inputs(self):
            raise RuntimeError("cuda")

    img = Image.new("RGB", (8, 8))
    assert tag_image(img, session=Boom(), tag_rows=[["0", "a", "0"]]) == []
