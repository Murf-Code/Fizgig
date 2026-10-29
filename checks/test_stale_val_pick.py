"""The Check button warns when picked validation images aren't in this training folder (Chris, 29 Oct 2026: an old
pick from another dataset quietly fell back to the automatic pick)."""
from types import SimpleNamespace

from fizgig.families import launch

DESC = SimpleNamespace()


def _st(folder, picked, **kw):
    return {"image_folder": str(folder), "VALIDATION_PICKED": picked, **kw}


def test_picks_from_another_dataset_are_named(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "run_subfolders", lambda d, st: [])
    for n in ("a.jpg", "b.jpg"):
        (tmp_path / n).write_bytes(b"x")
    w = launch.stale_validation_picks(DESC, _st(tmp_path, ["044a9b90.png", "zz.png"]))
    assert "2 of your 2" in w and "044a9b90.png" in w and "automatic pick" in w and "Re-pick" in w


def test_one_stale_pick_of_several(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "run_subfolders", lambda d, st: [])
    (tmp_path / "a.jpg").write_bytes(b"x")
    w = launch.stale_validation_picks(DESC, _st(tmp_path, ["a.png", "gone.png"]))
    assert w.startswith("1 of your 2") and "automatic" not in w


def test_picks_in_the_folder_or_a_ticked_subfolder_are_fine(tmp_path, monkeypatch):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "Peen").mkdir()
    (tmp_path / "Peen" / "p.jpg").write_bytes(b"x")
    monkeypatch.setattr(launch, "run_subfolders", lambda d, st: [("Peen", str(tmp_path / "Peen"))])
    assert launch.stale_validation_picks(DESC, _st(tmp_path, ["a.jpg", "Peen/p.jpg"])) == ""
    assert launch.stale_validation_picks(DESC, _st(tmp_path, [])) == ""


def test_likeness_picks_from_another_dataset_are_named(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(launch, "run_subfolders", lambda d, st: [])
    (tmp_path / "a.jpg").write_bytes(b"x")
    pairs = json.dumps([{"index": 0, "hash": "1", "image": "a.jpg"}, {"index": 1, "hash": "2", "image": "zaush.png"}])
    w = launch.stale_likeness_picks(DESC, {"image_folder": str(tmp_path), "SAMPLE_PAIRS": pairs})
    assert w.startswith("1 of your 2 Likeness picks") and "zaush.png" in w and "Re-pick" in w
    ok = json.dumps([{"index": 0, "hash": "1", "image": "a.jpg"}])
    assert launch.stale_likeness_picks(DESC, {"image_folder": str(tmp_path), "SAMPLE_PAIRS": ok}) == ""
    assert launch.stale_likeness_picks(DESC, {"image_folder": str(tmp_path)}) == ""
