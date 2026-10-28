"""Trim tags that don't fit, on close-ups of one of two characters: only the characters / species the tagger sees on
the crop (or the likeliest of them) stay, and a few common e621 tags JTP-3 never learned (naked, belly, armpits...)
are judged by tags it knows instead of always staying. The real case: a sheath close-up of an arcanine, cut from a
picture of an arcanine and a growlithe, kept growlithe, belly, armpits, petite, thin waist, no nipples and naked.
Its scores below are JTP-3 Hydra's own on that crop."""
import os

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
          "white background": 0.288}
RANKS = {"arcanine": 0.242, "growlithe": 0.065}         # the network's own probabilities (both under confidence 0)
UNKNOWN = {"@z4ush", "armpits", "belly", "large balls", "naked", "nintendo", "no nipples", "petite", "pokemon",
           "thin waist", "video games"}                  # not in JTP-3's tag list


def _cats():
    tags = [wd_tagger.norm(t) for t in CAPTION.split(",")]
    cats = {t: 0 for t in tags if t not in UNKNOWN}
    cats.update(arcanine=5, growlithe=5)
    for ss in J.STAND_INS.values():
        cats.update({s: 0 for s in ss})
    return cats


class FakeJudge:
    """jtp3_tagger.Tagger's face for trim: judge(img) -> (scores, ranks), categories, who, stand_ins."""
    who = J.WHO
    stand_ins = J.STAND_INS

    def __init__(self):
        self.categories = _cats()

    def judge(self, img):
        return dict(SCORES), dict(RANKS)

    def scores(self, img):
        return dict(SCORES)


def test_the_sheath_close_up_keeps_only_what_is_in_it():
    cc = C.CropCaptions(tagger=FakeJudge(), keep_score=J.KEEP_SCORE)        # Chris's settings: JTP-3 at 0.20
    out, dropped = cc.text_for(CAPTION, Image.new("RGB", (8, 8)))
    assert out == ("@z4ush, anthro, arcanine, balls, black fur, genital piercing, large balls, male, navel, nintendo, "
                   "penis, piercing, pokemon, sheath, simple background, video games, white background")
    for gone in ("growlithe", "armpits", "belly", "petite", "thin waist", "no nipples", "naked"):
        assert gone in dropped


def test_before_the_fix_these_were_never_judged():
    """The old rule (species and unknown tags always stay) kept them; the new one with no who / stand_ins does too."""
    out, _ = wd_tagger.trim(CAPTION, SCORES, _cats(), (), J.KEEP_SCORE, who=())
    for kept in ("growlithe", "armpits", "belly", "petite", "thin waist", "no nipples", "naked"):
        assert kept in out.split(", ")


def test_who_tags_are_judged_together():
    cats = {"fox": 5, "wolf": 5, "krystal": 4, "solo": 0}
    # neither seen: the likeliest stays, one far less likely goes
    out, dropped = wd_tagger.trim("me, fox, wolf, solo", {"solo": 0.9}, cats, (), 0.2, who=(4, 5),
                                  ranks={"fox": 0.3, "wolf": 0.05})
    assert out == "me, fox, solo" and dropped == ["wolf"]
    # about as likely (at least half the likeliest) stays too, and one the tagger sees always stays
    assert wd_tagger.trim("me, fox, wolf", {"wolf": 0.25}, cats, (), 0.2, who=(5,),
                          ranks={"fox": 0.9, "wolf": 0.3}) == ("me, fox, wolf", [])
    assert wd_tagger.trim("me, fox, wolf", {}, cats, (), 0.2, who=(5,),
                          ranks={"fox": 0.9, "wolf": 0.5}) == ("me, fox, wolf", [])
    # a lone species or character always stays, whatever it scores
    assert wd_tagger.trim("me, fox, krystal", {}, cats, (), 0.2, who=(4, 5)) == ("me, fox, krystal", [])
    # species aren't who tags for the WD tagger (its category 5 is meta), and the first tag always stays
    assert wd_tagger.WHO == (4,) and wd_tagger.Tagger.who == (4,)
    assert wd_tagger.trim("wolf, fox", {}, cats, (), 0.2, who=(5,), ranks={"fox": 0.9}) == ("wolf, fox", [])


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


def test_old_taggers_without_judge_still_trim():
    class Plain:
        categories = {"solo": 0, "tail": 0, "krystal": 4}

        def scores(self, img):
            return {"solo": 0.9}
    cc = C.CropCaptions(tagger=Plain(), keep_score=0.35)
    assert cc.text_for("me, solo, tail, krystal", Image.new("RGB", (4, 4))) == ("me, solo, krystal", ["tail"])


def test_the_page_says_how_characters_and_species_are_trimmed():
    html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    js = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    assert "Only the characters and species it sees in" in html and "the likeliest one stays" in html
    assert "species, character, artist and copyright tags and keeps" not in js
    assert "(naked, belly, armpits) by ones it does" in js
