"""Include sub-folders (the Start tab tick under the training folder, off by default): each sub-folder becomes its own dataset
block, and the loss watch keys every picture by its folder - so auto-recaption, exclusions and the per-image view
find the right file, and same-named pictures in two folders stay apart."""
import json
import os
import types

import pytest
from PIL import Image

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family
from fizgig.training.loss_logger import where


def _pic(path, caption="a photo"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (32, 32)).save(path)
    if caption is not None:
        with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
            f.write(caption)


@pytest.fixture
def tree(tmp_path):
    d = tmp_path / "data"
    _pic(str(d / "a.png"))
    _pic(str(d / "outfits" / "red" / "a.png"))             # the same name as the training folder's own
    _pic(str(d / "outfits" / "blue" / "b.png"))
    _pic(str(d / "excluded_by_look" / "x.png"))            # Fizgig's own folders, hidden ones, the output folder
    _pic(str(d / "originals" / "x.png"))
    _pic(str(d / ".hidden" / "x.png"))
    _pic(str(d / "out" / "sample" / "s.png"))
    os.makedirs(d / "empty")
    return d


def _st(d, **kw):
    return {"image_folder": str(d), "LORA_OUTPUT_DIR": str(d / "out"), "megapixels": 0.25,
            "cache_root": str(d.parent / "cache"), "subfolders": True, **kw}


def test_subfolders_skip_fizgigs_own_hidden_output_and_empty(tree):
    got = [rel for rel, _f in launch.subfolders(str(tree), (str(tree / "out"),))]
    assert got == ["outfits/blue", "outfits/red"]


def test_each_subfolder_is_its_own_block_with_its_prefix(tree):
    toml = launch.dataset_toml(get_family("krea2"), _st(tree))
    assert toml.count("[[datasets]]") == 3
    assert 'item_prefix = "outfits/red/"' in toml and 'item_prefix = "outfits/blue/"' in toml
    assert toml.count("cache_directory") == 3                # each its own cache folder


def test_off_by_default_and_off_means_one_block(tree):
    assert form.DEFAULTS["TRAIN_SUBFOLDERS"] is False
    assert launch.dataset_toml(get_family("krea2"), _st(tree, subfolders=False)).count("[[datasets]]") == 1


def test_a_training_folder_of_only_subfolders_has_no_empty_block(tmp_path):
    d = tmp_path / "data"
    _pic(str(d / "one" / "a.png"))
    _pic(str(d / "two" / "b.png"))
    toml = launch.dataset_toml(get_family("krea2"), _st(d))
    assert toml.count("[[datasets]]") == 2 and f'image_directory = "{str(d).replace(os.sep, "/")}"\n' not in toml
    assert not any("No caption files" in p for p in launch.start_problems(get_family("krea2"), _st(d)))


def test_an_uncaptioned_subfolder_is_refused_by_name(tree):
    _pic(str(tree / "outfits" / "green" / "c.png"), caption=None)
    probs = launch.start_problems(get_family("krea2"), _st(tree))
    assert any("the sub-folder outfits/green" in p for p in probs)


def test_tick_hidden_with_subjects_dop_isolation_edit_slider():
    desc = get_family("klein")
    for v in ({"FAMILY_MULTICONCEPT": True}, {"FAMILY_DOP": True}, {"FAMILY_MULTI_PERSON": True}):
        assert "TRAIN_SUBFOLDERS" in form.state(desc, v)["hidden"]
    assert "TRAIN_SUBFOLDERS" not in form.state(desc, {})["hidden"]
    assert not launch.subfolders_on(desc, {"subfolders": True, "multiconcept": True})


def test_the_tick_says_what_it_leaves_out_and_what_it_trains(tree):
    desc = get_family("krea2")
    v = {"LORA_OUTPUT_DIR": str(tree / "out")}
    off, act = form.subfolders_note(desc, v, str(tree))
    assert "2 sub-folders with 2 more pictures are not trained on" in off and act is None
    _pic(str(tree / "outfits" / "blue" / "c.png"), caption=None)
    on, act = form.subfolders_note(desc, {**v, "TRAIN_SUBFOLDERS": True}, str(tree))
    assert on.startswith("Training on 3 folders, 4 pictures.") and "Start will refuse them: outfits/blue (1)" in on
    assert "outfits/red" not in on and act["label"] == "Show all folders"            # the list waits for the button
    full, act = form.subfolders_note(desc, {**v, "TRAIN_SUBFOLDERS": True, **act["values"]}, str(tree))
    assert "All folders: the training folder 1 · outfits/blue 2 · outfits/red 1." in full
    assert act["label"] == "Hide the folders"


def test_where_takes_the_longest_prefix_and_refuses_unknown_folders():
    f = {"": "MAIN", "outfits/": "O", "outfits/red/": "R", "s2:": "S2"}
    assert where("a", f) == ("", "MAIN", "a")
    assert where("outfits/red/a", f) == ("outfits/red/", "R", "a")
    assert where("s2:a", f) == ("s2:", "S2", "a")
    assert where("other/a", {"": "MAIN"}) == (None, None, None)
    assert where("s3:a", {"": "MAIN"}) == (None, None, None)


def test_exclusions_are_kept_in_each_pictures_own_folder(tree, tmp_path):
    pytest.importorskip("torch")
    from fizgig.training.loss_logger import PerImageLossWatch
    folders = {"": str(tree), "outfits/red/": str(tree / "outfits" / "red")}
    w = PerImageLossWatch(str(tmp_path / "run"), folders=folders, family="krea2")
    w._record_exclusion("outfits/red/a", 5)
    red = json.load(open(tree / "outfits" / "red" / "fizgig_excluded.json", encoding="utf-8"))
    assert list(red) == ["a"] and red["a"]["caption"] == "a photo"
    assert not os.path.exists(tree / "fizgig_excluded.json") or "a" not in json.load(open(tree / "fizgig_excluded.json"))
    again = PerImageLossWatch(str(tmp_path / "run2"), folders=folders, family="krea2")
    assert "outfits/red/a" in again._known_hard and "a" not in again._known_hard


def test_auto_recaption_writes_the_subfolders_caption(tree, monkeypatch):
    pytest.importorskip("torch")
    from fizgig.families import loss_watch
    from fizgig.krea2 import embedder, utils
    monkeypatch.setattr(utils, "load_krea2_text_encoder", lambda *a, **k: types.SimpleNamespace(to=lambda d: None))
    monkeypatch.setattr(embedder, "generate_caption", lambda *a, **k: "a red outfit")
    folders = {"": str(tree), "outfits/red/": str(tree / "outfits" / "red")}
    obj = types.SimpleNamespace(captioner_path="x", instr=None, instr_detailed=None, trigger_word="zmain",
                                triggers={"outfits/red/": "zred"},
                                trigger_position="start", caption_ext=".txt", auto_recaption=True,
                                watch=types.SimpleNamespace(where=lambda k: where(k, folders)))
    rows = loss_watch.Watch._recaption(obj, [("outfits/red/a", None, 1)], {"outfits/red/a": "item"}, "cpu")
    # its own folder's trigger word, never the training folder's
    assert rows and open(tree / "outfits" / "red" / "a.txt", encoding="utf-8").read() == "zred, a red outfit"
    assert open(tree / "a.txt", encoding="utf-8").read() == "a photo"           # the same-named one is untouched


def test_the_per_image_view_finds_each_picture_and_caption(tree, tmp_path):
    from fizgig.web import image_loss
    # nothing stored about the folders: a run moved to a pod (or back) finds them from its own training folder
    out = tmp_path / "run"
    os.makedirs(out / "loss_log")
    run = {"output_dir": str(out), "image_folder": str(tree), "result": "done",
           "values": {"FAMILY_CONCEPT_DIR": str(tree / "outfits" / "blue")}}
    pics = image_loss.image_paths(run, ["a", "outfits/red/a"])
    assert pics["outfits/red/a"] == os.path.join(str(tree / "outfits" / "red"), "a.png")
    assert pics["a"] == os.path.join(str(tree), "a.png")
    image_loss.save_caption(run, "outfits/red/a", "fixed", ".txt")
    assert image_loss.read_caption(run, "outfits/red/a", ".txt") == ("fixed", True)
    assert image_loss.read_caption(run, "a", ".txt") == ("a photo", True)
    assert image_loss.read_caption(run, "s2:b", ".txt") == ("a photo", True)              # subject 2's folder


def test_an_upload_keeps_its_subfolder_and_answers_with_the_top_folder(tmp_path, monkeypatch):
    import io
    from fizgig.web import server
    monkeypatch.setattr(server, "uploads_root", lambda: str(tmp_path))
    r = server._save_upload("probe/group/one", "x.txt", io.BytesIO(b"hi"), 2)
    assert (tmp_path / "probe" / "group" / "one" / "x.txt").is_file()
    assert r["folder"] == os.path.join(str(tmp_path), "probe")
    for bad in ("probe/../evil", "probe//x", "../x"):
        with pytest.raises(ValueError):
            server._save_upload(bad, "x.txt", io.BytesIO(b"hi"), 2)


def test_a_finetunes_regularisation_folder_inside_is_never_a_subfolder(tree):
    desc = get_family("krea2")
    reg = str(tree / "outfits" / "blue")
    v = {"FAMILY_KIND": dict(form.kinds(desc))["finetune"], "TRAIN_SUBFOLDERS": True, "FAMILY_FT_REG_DIR": reg,
         "LORA_OUTPUT_DIR": str(tree / "out")}
    assert form.subfolders_note(desc, v, str(tree))[0].startswith("Training on 2 folders")
    toml = launch.dataset_toml(desc, _st(tree, FAMILY_FT=True, FAMILY_FT_REG_DIR=reg))
    assert 'item_prefix = "outfits/blue/"' not in toml and 'item_prefix = "reg:"' in toml and "is_reg = true" in toml
