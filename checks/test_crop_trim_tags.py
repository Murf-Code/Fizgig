"""Trim tags that don't fit on close-ups: characters / species stay only when the tagger sees them on the crop, fur
colour / marking / pattern tags need only half the keep score, and a few common e621 tags JTP-3 never learned (naked,
belly, armpits...) are judged by tags it knows instead of always staying. The real case: a sheath close-up of an
arcanine, cut from a picture of an arcanine and a growlithe, kept arcanine, growlithe, belly, armpits, petite, thin
waist, no nipples and naked. Its scores below are JTP-3 Hydra's own on that crop."""
import os

import pytest
from PIL import Image

from fizgig.dataset import crop_captions as C
from fizgig.dataset import jtp3_tagger as J
from fizgig.dataset import wd_tagger

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "..", "src", "fizgig", "web", "static")

CAPTION = ("@z4ush, 3 toes, abs, age difference, anthro, arcanine, armpits, balls, barefoot, belly, black fur, "
           "black nose, blonde hair, breasts, claws, duo, ear piercing, eyewear, feet, female, genital piercing, grin, "
           "growlithe, hair, half-closed eyes, half-erect, large balls, size difference, looking at viewer, "
           "looking away, male, multicolored fur, muscular, naked, navel, nintendo, nipples, no nipples, nude, "
           "open mouth, orange fur, paws, penis, petite, piercing, pokemon, purple eyes, pussy, sheath, "
           "simple background, sitting, small breasts, smile, stripes, tail, tan fur, thick thighs, thin waist, "
           "tongue, video games, white background, white fur, wide hips")
# JTP-3 Hydra on the 177x177 sheath crop (confidence, implications applied; tags under 0.005 left out)
SCORES = {"anthro": 0.861, "balls": 0.696, "black fur": 0.272, "butt": 0.009, "claws": 0.01, "female": 0.02,
          "genital piercing": 0.801, "hair": 0.066, "male": 0.996, "navel": 0.299, "nude": 0.153, "penis": 0.734,
          "piercing": 0.992, "sheath": 0.922, "simple background": 0.459, "smile": 0.009, "tail": 0.099,
          "white background": 0.288, "multicolored fur": 0.006}
UNKNOWN = {"@z4ush", "armpits", "belly", "large balls", "naked", "nintendo", "no nipples", "petite", "pokemon",
           "thin waist", "video games"}                  # not in JTP-3's tag list


def _cats():
    tags = [wd_tagger.norm(t) for t in CAPTION.split(",")]
    cats = {t: 0 for t in tags if t not in UNKNOWN}
    cats.update(arcanine=5, growlithe=5)
    for ss in J.STAND_INS.values():
        cats.update({s: 0 for s in ss})
    return cats


class FakeJTP3:
    """jtp3_tagger.Tagger's face for trim: scores(img), categories, who, stand_ins."""
    who = J.WHO
    stand_ins = J.STAND_INS

    def __init__(self):
        self.categories = _cats()

    def scores(self, img):
        return dict(SCORES)


def _sheath(keep_score):
    return C.CropCaptions(tagger=FakeJTP3(), keep_score=keep_score).text_for(CAPTION, Image.new("RGB", (8, 8)))


def test_the_sheath_close_up_keeps_only_what_is_in_it():
    out, dropped = _sheath(J.KEEP_SCORE)                                   # Chris's settings: JTP-3 at 0.20
    assert out == ("@z4ush, anthro, balls, black fur, genital piercing, large balls, male, navel, nintendo, penis, "
                   "piercing, pokemon, sheath, simple background, video games")
    for gone in ("arcanine", "growlithe", "armpits", "belly", "petite", "thin waist", "no nipples", "naked",
                 "multicolored fur", "orange fur", "stripes"):
        assert gone in dropped
    # at 0.30 navel and simple background go too, and black fur (a fur colour: half the bar) still stays
    out, _ = _sheath(0.30)
    assert out == ("@z4ush, anthro, balls, black fur, genital piercing, large balls, male, nintendo, penis, piercing, "
                   "pokemon, sheath, video games")


def test_before_the_fix_these_were_never_judged():
    """The old rule (species and unknown tags always stay) kept them."""
    out, _ = wd_tagger.trim(CAPTION, SCORES, _cats(), (), J.KEEP_SCORE, who=())
    for kept in ("arcanine", "growlithe", "armpits", "belly", "petite", "thin waist", "no nipples", "naked"):
        assert kept in out.split(", ")


def test_species_and_characters_need_to_be_seen():
    cats = {"fox": 5, "wolf": 5, "krystal": 4, "solo": 0}
    assert wd_tagger.trim("me, fox, wolf, krystal, solo", {"solo": 0.9, "fox": 0.2}, cats, (), 0.2,
                          who=(4, 5)) == ("me, fox, solo", ["wolf", "krystal"])
    # species aren't who tags for the WD tagger (its category 5 is meta), and the first tag always stays
    assert wd_tagger.WHO == (4,) and wd_tagger.Tagger.who == (4,)
    assert wd_tagger.trim("wolf, fox, krystal", {}, cats, (), 0.2) == ("wolf, fox", ["krystal"])


def test_fur_colours_markings_and_patterns_get_half_the_bar():
    for t in ("black fur", "Multicolored_Fur", "two tone body", "orange scales", "stripes", "spots", "facial markings",
              "chest tuft", "countershading", "white spots", "striped fur", "markings"):
        assert wd_tagger.is_pattern(t), t
    for t in ("fur", "hair", "blonde hair", "cum on body", "long body", "wet fur", "tail feathers", "glistening body",
              "striped thighhighs", "sheath"):
        assert not wd_tagger.is_pattern(t), t
    cats = {"black fur": 0, "stripes": 0, "smile": 0}
    out, dropped = wd_tagger.trim("me, black fur, stripes, smile, spotted fur", {"black fur": 0.16, "stripes": 0.14,
                                  "smile": 0.19}, cats, (), 0.30)
    assert out == "me, black fur, spotted fur" and dropped == ["stripes", "smile"]   # one it doesn't know stays


def test_stand_ins_judge_tags_the_tagger_does_not_know():
    cats = {"nude": 0, "navel": 0}
    st = {"naked": ("nude",), "belly": ("navel", "abs")}
    # a stand-in needs a little over the keep score (0.2 + 0.15)
    assert wd_tagger.trim("me, naked, belly", {"nude": 0.36, "navel": 0.3}, cats, (), 0.2,
                          stand_ins=st) == ("me, naked", ["belly"])
    assert wd_tagger.trim("me, naked, belly", {"nude": 0.36, "navel": 0.3}, cats, (), 0.2, stand_ins=st,
                          stand_in_extra=0.0) == ("me, naked, belly", [])
    # other unknown tags (artists, copyrights, your own) still stay
    assert wd_tagger.trim("me, @artist, pokemon, my tag", {}, cats, (), 0.2, stand_ins=st)[1] == []
    # a tag the tagger knows is judged by its own score, not a stand-in
    cats["naked"] = 0
    assert wd_tagger.trim("me, naked", {"nude": 0.9}, cats, (), 0.2, stand_ins=st) == ("me", ["naked"])


def test_stand_ins_name_real_jtp3_tags():
    """Every stand-in is an e621 tag (the Tagger drops any its list lacks); the tags they judge are lower case."""
    for tag, ss in J.STAND_INS.items():
        assert tag == wd_tagger.norm(tag) and ss and all(s == wd_tagger.norm(s) for s in ss)
    assert J.WHO == (4, 5)


def test_taggers_without_who_or_stand_ins_still_trim():
    class Plain:
        categories = {"solo": 0, "tail": 0, "krystal": 4}

        def scores(self, img):
            return {"solo": 0.9, "krystal": 0.5}
    cc = C.CropCaptions(tagger=Plain(), keep_score=0.35)
    assert cc.text_for("me, solo, tail, krystal", Image.new("RGB", (4, 4))) == ("me, solo, krystal", ["tail"])


def test_the_page_says_how_characters_and_species_are_trimmed():
    html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    js = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    assert "Characters and species stay only if it sees" in html and "need only half the score" in html
    assert "background tags 0.25 more" in html
    assert "species, character, artist and copyright tags and keeps" not in js
    assert "(naked, belly, armpits) by ones it does" in js


def test_background_tags_need_a_higher_score():
    for t in ("simple background", "white_background", "gradient background", "transparent background",
              "detailed background", "outside", "inside"):
        assert wd_tagger.is_background(t), t
    for t in ("white fur", "background character", "sheath"):
        assert not wd_tagger.is_background(t), t
    assert wd_tagger.background_bar(0.2) == pytest.approx(0.45) and wd_tagger.background_bar(0.9) == 0.95
    cats = {"simple background": 0, "white background": 0, "smile": 0}
    out, dropped = wd_tagger.trim("me, simple background, white background, smile",
                                  {"simple background": 0.46, "white background": 0.4, "smile": 0.3}, cats, (), 0.2)
    assert out == "me, simple background, smile" and dropped == ["white background"]


def test_the_sheath_close_up_backgrounds():
    out, dropped = _sheath(0.20)                  # simple background 0.459 just clears 0.45; white 0.288 doesn't
    assert "simple background" in out.split(", ") and "white background" in dropped
    out, dropped = _sheath(0.30)
    assert "simple background" in dropped and "white background" in dropped


def test_crops_get_close_up(tmp_path):
    pic = tmp_path / "a.png"
    Image.new("RGB", (64, 64)).save(pic)
    (tmp_path / "a.txt").write_text("me, solo, tail")
    logs = []
    cc = C.CropCaptions(keep=["me"], closeup=True)
    cc.to_crop(str(pic), str(tmp_path / "Crop_tail_001.png"), logs.append)
    assert (tmp_path / "Crop_tail_001.txt").read_text() == "me, solo, tail, close-up"
    assert "  caption: added close-up\n" in logs
    # not twice, not on sentence captions, and only when turned on
    assert cc.caption_for("me, Close_Up, solo", None) == ("me, Close_Up, solo", [], [])
    s = "A fox standing in a field, looking at the camera."
    assert cc.caption_for(s, None) == (s, [], [])
    assert C.CropCaptions().caption_for("me, solo", None) == ("me, solo", [], [])


def test_a_close_up_that_replaced_its_picture_gets_it_but_a_resized_picture_does_not(tmp_path):
    cc = C.CropCaptions(closeup=True)
    for name, cropped, want in (("a", True, "me, solo, close-up"), ("b", False, "me, solo")):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{name}.png")
        (tmp_path / f"{name}.txt").write_text("me, solo")
        cc.in_place(str(tmp_path / f"{name}.png"), str(tmp_path / f"{name}.png"), str(tmp_path / "originals"),
                    lambda m: None, None, cropped=cropped)
        assert (tmp_path / f"{name}.txt").read_text() == want
    assert (tmp_path / "originals" / "a.txt").read_text() == "me, solo"      # its first version is kept
    assert not (tmp_path / "originals" / "b.txt").exists()


def test_close_up_is_on_unless_turned_off(tmp_path, monkeypatch):
    from fizgig import user_data
    from fizgig.web import prep_jobs
    monkeypatch.setattr(user_data, "MODELS_DIR", str(tmp_path))
    c, _ = prep_jobs.caption_settings({"copy": True})
    assert c["closeup"] is True and prep_jobs.crop_captions(c).closeup is True
    c, _ = prep_jobs.caption_settings({"copy": True, "closeup": False})
    assert c["closeup"] is False and prep_jobs.crop_captions(c).closeup is False
    html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    js = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    assert 'id="prepCapCloseup" checked>' in html and "Tag crops as close-up" in html
    assert "closeup: p.capCloseup" in js and "saved.capCloseup !== false" in js


def test_missing_tags_the_tagger_is_sure_of():
    cats = {"solo": 0, "mammal": 0, "simple background": 0, "explicit": 7, "fox": 5, "krystal": 4, "star fox": 3,
            "animal vulva": 0, "tail": 0}
    sc = {"solo": 0.92, "mammal": 1.0, "simple background": 0.8, "explicit": 0.99, "fox": 0.75, "krystal": 0.9,
          "star fox": 0.9, "animal vulva": 0.95, "tail": 0.69}
    got = wd_tagger.missing(["me", "Solo", "animal pussy"], sc, cats, 0.70, who=(4, 5),
                            aliases={"animal pussy": "animal vulva"})
    # surest first; no rating / meta / copyright; an older name in the caption counts; background needs 0.95
    assert got == ["mammal", "krystal", "fox"]
    # the WD tagger's who is characters only (no fox); a background at 0.8 clears 0.55 + 0.25
    assert wd_tagger.missing([], sc, cats, 0.55) == ["mammal", "animal vulva", "solo", "krystal", "simple background",
                                                     "tail"]
    assert wd_tagger.missing([], sc, cats, 0.70, names=["tail", "solo"]) == ["solo"]   # only the tagger's own tags


def test_crop_captions_add_what_the_tagger_is_sure_of():
    class T:
        categories = {"solo": 0, "tail": 0, "long hair": 0, "safe": 9}
        names = ["solo", "tail", "long hair", "safe"]

        def scores(self, img):
            return {"solo": 0.95, "tail": 0.1, "long hair": 0.9, "safe": 0.99}
    img = Image.new("RGB", (4, 4))
    cc = C.CropCaptions(keep=["me"], tagger=T(), keep_score=0.35, closeup=True, add_score=0.70)
    assert cc.caption_for("me, solo, tail", img) == ("me, solo, long hair, close-up", ["tail"], ["long hair", "close-up"])
    assert cc.caption_for("me, solo_x, tail", img)[0] == "me, solo_x, solo, long_hair, close-up"   # underscore style
    s = "A fox standing in a field, looking at the camera."
    assert cc.caption_for(s, img) == (s, [], [])                                    # sentences: as written
    off = C.CropCaptions(tagger=T(), keep_score=0.35)
    assert off.caption_for("me, solo, tail", img) == ("me, solo", ["tail"], [])     # off by default
    assert cc.text_for("me, solo, tail", img) == ("me, solo", ["tail"])            # text_for only trims


def test_sheath_close_up_with_add_missing_tags():
    class T(FakeJTP3):
        names = None
    sc = dict(SCORES, mammal=1.0, genitals=0.97, canid=0.93, solo=0.92, canine=0.9, **{"brown body": 0.83,
              "explicit": 0.98, "close-up": 0.6})
    t = T()
    t.categories.update(mammal=0, genitals=0, canid=0, solo=0, canine=0, explicit=7, **{"brown body": 0, "close-up": 0})
    t.scores = lambda img: dict(sc)
    text, _, added = C.CropCaptions(tagger=t, keep_score=0.2, closeup=True, add_score=0.89).caption_for(
        CAPTION, Image.new("RGB", (8, 8)))
    assert added == ["mammal", "genitals", "canid", "solo", "canine", "close-up"]
    assert text.endswith("video games, mammal, genitals, canid, solo, canine, close-up")


def test_add_missing_tags_on_the_page(tmp_path, monkeypatch):
    from fizgig import user_data
    from fizgig.web import prep_jobs
    monkeypatch.setattr(user_data, "MODELS_DIR", str(tmp_path))
    d = tmp_path / wd_tagger.FOLDER
    d.mkdir()
    for f in wd_tagger.FILES:
        (d / f).write_text("")
    c, _ = prep_jobs.caption_settings({"trim": True, "add": True, "add_score": "0.89"})
    assert c["add"] is True and c["add_score"] == 0.89 and prep_jobs.crop_captions(c, "T").add_score == 0.89
    c, _ = prep_jobs.caption_settings({"trim": True, "add": True, "add_score": ""})
    assert c["add_score"] == 0.70
    assert prep_jobs.crop_captions({**c, "add": False}, "T").add_score is None     # off
    assert prep_jobs.crop_captions({**c, "trim": False}, "T").add_score is None    # needs Trim's tagger
    for bad in ("0", "1.5", "lots"):
        with pytest.raises(ValueError, match="Add missing tags scored at least"):
            prep_jobs.caption_settings({"trim": True, "add": True, "add_score": bad})
    prep_jobs.caption_settings({"trim": True, "add": False, "add_score": "lots"})   # ignored while off
    html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    js = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    assert '<input type="checkbox" id="prepCapAdd">' in html and "Add missing tags the tagger is sure of" in html
    # the same kind of box as Keep tags scored at least, on its own labelled row
    assert ('<div class="field-row"><span class="field-label">Add tags scored at least:</span>\n'
            '                      <input type="text" id="prepAddScore" class="w-num" value="0.70">') in html
    assert 'id="prepKeepScore" class="w-num"' in html
    assert "It only adds tags when Add missing tags is on." in html and "never adds any" not in html
    assert "capAdd: $(\"prepCapAdd\").checked" in js and "add: p.capAdd, add_score: p.addScore" in js
    assert "saved.addScore || \"0.70\"" in js and "Add tags scored at least: a number between 0 and 1" in js
