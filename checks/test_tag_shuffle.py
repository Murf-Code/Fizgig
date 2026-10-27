"""Tag shuffle / keep first N / tag dropout / caption dropout (dataset/tag_variants.py) on tiny CPU data: variants are
deterministic, the kept head and the trigger stay first, the drop rates come out about right, each step's pick is a
pure function of (seed, epoch, image, repeat) so a resume repeats it, validation keeps the original caption, a caption
edit makes the old variants unusable and a rebuild encodes only what's new, Batch Size > 1 picks per image, and the
launch / form / cache wiring."""
import hashlib
import json
import os

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from fizgig.dataset import tag_variants as tv  # noqa: E402
from fizgig.dataset.image_dataset import BucketBatchManager, ItemInfo, original_text_batch  # noqa: E402
from fizgig.families import form, launch  # noqa: E402
from fizgig.families.registry import get  # noqa: E402

TAGS = "zwxem, 1girl, red dress, smiling, outdoors, sky, tree, sunlight, long hair, blue eyes, looking at viewer"
PROSE = "A photo of a woman in a red dress smiling in a park. The sun is shining through the trees."


def _enc_one(text, dim=8):
    """A stand-in text encoder: one row per tag (at least one), values from the text's hash - the same text always
    encodes the same, another text differently."""
    n = max(1, len([t for t in text.split(",") if t.strip()]))
    g = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(text.encode()).digest()[:6], "big"))
    return {"hidden_states": torch.randn(n, dim, generator=g)}


class _Encoder:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls += list(texts)
        return [_enc_one(t) for t in texts]


def _te(path, caption):
    save_file({f"cond__{k}": v for k, v in _enc_one(caption).items()}, str(path),
              metadata={"caption1": caption, "architecture": "t"})
    return str(path)


# ---- captions -----------------------------------------------------------------------------------------------------
def test_tags_and_natural_language():
    assert tv.split_tags(TAGS)[:3] == ["zwxem", "1girl", "red dress"]
    assert tv.split_tags(PROSE) is None                                 # no commas
    assert tv.split_tags("a woman standing in a park, holding a red umbrella over her head, under a grey sky") is None
    assert tv.split_tags("Smiling. She stands, waves. Then leaves, quickly now.") is None   # sentences
    assert tv.split_tags("solo") is None


def test_variants_deterministic_per_seed():
    s = tv.TagSettings.from_params(shuffle=True, keep_first=1, tag_dropout=0.1, variants=6)
    a, b = tv.variant_captions(TAGS, s, seed=1), tv.variant_captions(TAGS, s, seed=1)
    assert a == b and len(a) == 6 and len(set(a)) > 1
    assert tv.variant_captions(TAGS, s, seed=2) != a
    more = tv.variant_captions(TAGS, tv.TagSettings.from_params(shuffle=True, keep_first=1, tag_dropout=0.1,
                                                                 variants=8), seed=1)
    assert more[:6] == a                       # a bigger count keeps the variants it shares
    assert tv.variant_captions(PROSE, s) == []                          # natural language: as written
    assert tv.variant_captions(TAGS, tv.TagSettings.from_params(caption_dropout=0.05)) == []   # nothing to vary


def test_keep_first_n_and_trigger_first():
    tags = tv.split_tags(TAGS)
    for n in (0, 1, 3):
        s = tv.TagSettings.from_params(shuffle=True, keep_first=n, tag_dropout=0.3, variants=20)
        for v in tv.variant_captions(TAGS, s):
            assert v.split(", ")[:n] == tags[:n]
            assert set(v.split(", ")) <= set(tags)
    # the folder's trigger is kept, first, wherever it sits in the caption
    cap = "1girl, red dress, zwxem, smiling, outdoors, sky"
    s = tv.TagSettings.from_params(shuffle=True, keep_first=1, tag_dropout=0.5, variants=30, keep_words=["ZWXEM"])
    for v in tv.variant_captions(cap, s):
        assert v.split(", ")[:2] == ["zwxem", "1girl"]


def test_tag_dropout_rate():
    s = tv.TagSettings.from_params(shuffle=False, keep_first=1, tag_dropout=0.2, variants=1)
    tail = len(tv.split_tags(TAGS)) - 1
    kept = [len(tv.make_variant(TAGS, s, i).split(", ")) - 1 for i in range(3000)]
    rate = 1 - sum(kept) / (len(kept) * tail)
    assert abs(rate - 0.2) < 0.02, rate
    v = tv.make_variant(TAGS, s, 5)                                     # no shuffle: the order is kept
    order = [t for t in tv.split_tags(TAGS) if t in v.split(", ")]
    assert v.split(", ") == order


# ---- the per-step pick ----------------------------------------------------------------------------------------------
def _items(tmp_path, caps, settings, enc=None):
    infos = []
    enc = enc or _Encoder()
    for i, cap in enumerate(caps):
        lat = tmp_path / f"img{i}_0064x0064_t.safetensors"
        save_file({"latent_4x4": torch.randn(4, 4, 4, generator=torch.Generator().manual_seed(i))}, str(lat))
        te = _te(tmp_path / f"img{i}_t_te.safetensors", cap)
        tv.write_variants(enc, te, cap, settings, arch="t")
        it = ItemInfo(f"img{i}", cap, (64, 64), (64, 64), latent_cache_path=str(lat))
        it.text_encoder_output_cache_path = te
        infos.append(it)
    return infos, enc


def test_caption_dropout_rate_and_picks_are_seeded(tmp_path):
    s = tv.TagSettings.from_params(shuffle=True, tag_dropout=0.1, caption_dropout=0.1, variants=4)

    def run(name, seed):
        d = tmp_path / name
        d.mkdir()
        items, _ = _items(d, [TAGS], s)
        vp = tv.VariantPicker(s, seed)
        return [vp.choose("img0", items[0].text_encoder_output_cache_path, e, occ) for e in range(1, 400)
                for occ in range(5)]
    a = run("a", 42)
    assert a == run("b", 42)                                             # deterministic per seed
    assert a != run("c", 7)
    empty = a.count(tv.EMPTY) / len(a)
    assert abs(empty - 0.1) < 0.02, empty
    picks = [p for p in a if p != tv.EMPTY]
    assert set(picks) == {0, 1, 2, 3}                                   # every variant comes up
    assert all(abs(picks.count(i) / len(picks) - 0.25) < 0.04 for i in range(4))


def _epoch(bm, epoch):
    bm.epoch = epoch
    return [bm[i] for i in range(len(bm))]


def test_batch_loads_the_picked_variant_and_resume_repeats_it(tmp_path):
    s = tv.TagSettings.from_params(shuffle=True, keep_first=1, tag_dropout=0.2, caption_dropout=0.2, variants=4)
    caps = [TAGS, "zwxem, 1boy, hat, beach", PROSE, "zwxem, cat, sofa, window, night"]
    items, _ = _items(tmp_path, caps, s)
    bm = BucketBatchManager({(64, 64): items + items[:2]}, 2)          # two images repeat (num_repeats)
    bm.variants = tv.VariantPicker(s, seed=42)
    first = {e: _epoch(bm, e) for e in (1, 2, 3)}
    for e, batches in first.items():
        for b in batches:
            assert len(b["caption_pick"]) == len(b["item_keys"])
            for j, (key, pick, te) in enumerate(zip(b["item_keys"], b["caption_pick"], b["item_te_paths"])):
                cap = caps[int(key[3:])]
                want = ("" if pick == tv.EMPTY else cap if pick == tv.ORIG else
                        tv.variant_captions(cap, s)[pick])
                row = _enc_one(want)["hidden_states"]
                got = b["cond__hidden_states"][j]
                valid = b.get("cond__hidden_states__valid")
                n = int(valid[j].sum()) if valid is not None else got.shape[0]
                assert n == row.shape[0] and torch.equal(got[:n], row), (key, pick)
    # the prose caption never gets a tag variant: only its own or the empty caption
    prose_picks = {p for bs in first.values() for b in bs for k, p in zip(b["item_keys"], b["caption_pick"])
                   if k == "img2"}
    assert prose_picks <= {tv.ORIG, tv.EMPTY}
    # a resume: a fresh picker (and manager) at epoch 3 picks exactly what the run did - no state to carry
    bm2 = BucketBatchManager({(64, 64): items + items[:2]}, 2)
    bm2.variants = tv.VariantPicker(s, seed=42)
    again = _epoch(bm2, 3)
    assert [b["caption_pick"] for b in again] == [b["caption_pick"] for b in first[3]]
    assert all(torch.equal(a["cond__hidden_states"], b["cond__hidden_states"]) for a, b in zip(again, first[3]))
    # a repeated image may pick differently on each of its steps in an epoch
    reps = [(k, p) for b in first[1] for k, p in zip(b["item_keys"], b["caption_pick"]) if k == "img0"]
    assert len(reps) == 2


def test_no_settings_is_as_before(tmp_path):
    items, _ = _items(tmp_path, [TAGS, "a, b, c"], tv.TagSettings())
    bm = BucketBatchManager({(64, 64): items}, 1)
    b = bm[0]
    assert "caption_pick" not in b
    assert not list(tmp_path.glob("*_tags.safetensors"))               # nothing written when off


def test_validation_keeps_the_original_caption(tmp_path):
    from fizgig.training.val_loss import FixedValidation
    s = tv.TagSettings.from_params(shuffle=True, tag_dropout=0.3, caption_dropout=0.5, variants=4)
    items, _ = _items(tmp_path, [TAGS, "zwxem, 1boy, hat, beach, sand, sea"], s)
    bm = BucketBatchManager({(64, 64): items}, 2)
    bm.variants = tv.VariantPicker(s, seed=3)
    b = next(b for e in range(1, 50) for b in _epoch(bm, e) if any(p != tv.ORIG for p in b["caption_pick"]))
    orig = original_text_batch(b)
    for j, k in enumerate(orig["item_keys"]):
        row = _enc_one(items[int(k[3:])].caption)["hidden_states"]
        valid = orig.get("cond__hidden_states__valid")
        n = int(valid[j].sum()) if valid is not None else orig["cond__hidden_states"].shape[1]
        assert torch.equal(orig["cond__hidden_states"][j][:n], row)
    v = FixedValidation(["img0", "img1"], images=2)
    v.offer(b)
    kept = v.batches["img0"]
    assert kept["caption_pick"] == ["orig", "orig"]
    assert torch.equal(kept["cond__hidden_states"], orig["cond__hidden_states"])


def test_caption_edit_invalidates_and_rebuild_encodes_only_whats_new(tmp_path):
    s = tv.TagSettings.from_params(shuffle=True, keep_first=1, tag_dropout=0.1, caption_dropout=0.1, variants=4)
    enc = _Encoder()
    te = _te(tmp_path / "img0_t_te.safetensors", TAGS)
    n0 = tv.write_variants(enc, te, TAGS, s)
    assert n0 == len(set(tv.variant_captions(TAGS, s)) - {TAGS}) + 1      # the variants + the empty caption
    assert tv.write_variants(enc, te, TAGS, s) == 0                     # unchanged: nothing re-encoded
    s6 = tv.TagSettings.from_params(shuffle=True, keep_first=1, tag_dropout=0.1, caption_dropout=0.1, variants=6)
    new = set(tv.variant_captions(TAGS, s6)[4:]) - set(tv.variant_captions(TAGS, s)) - {TAGS}
    assert tv.write_variants(enc, te, TAGS, s6) == len(new)             # more variants: only the new ones
    vp = tv.VariantPicker(s6, 1)
    assert vp.info(te) == (6, True)
    # a caption fix / auto-recaption rewrites the text cache: the old variants are no longer used...
    fixed = "zwxem, 1girl, blue dress, frowning, indoors"
    _te(tmp_path / "img0_t_te.safetensors", fixed)
    os.utime(te, (1, 1))                     # a new mtime even on a coarse clock
    assert vp.info(te) == (0, True)
    assert all(vp.choose("img0", te, e) in (tv.ORIG, tv.EMPTY) for e in range(1, 60))
    # ...until they're rebuilt for the new caption (the empty caption is reused)
    enc.calls.clear()
    tv.write_variants(enc, te, fixed, s6)
    assert "" not in enc.calls and enc.calls
    assert vp.info(te)[0] == 6
    md = tv._header(tv.variants_path(te))[0]
    assert md["source"] == fixed and json.loads(md["captions"]) == tv.variant_captions(fixed, s6)


def test_wiring():
    """The trainer seeds the picks with the run's seed and logs what each dataset does; the cache pass builds the
    variants with the encoder it loaded; a caption fix / auto-recaption remakes the item's variants."""
    import inspect
    from fizgig.families import cache, loss_watch, train
    t = inspect.getsource(train.train_family)
    assert "variant_seed = int(seed)" in t and "_tag_summary(group.datasets)" in t
    assert t.index("variant_seed = int(seed)") < t.index("generate_dataset_group_by_blueprint(blueprint.dataset_group")
    assert "_encode_tag_variants(args, datasets, driver, te, desc)" in inspect.getsource(cache.main)
    c = inspect.getsource(loss_watch.Watch._captions)
    assert c.index("cache.save_cond") < c.index("write_variants")


# ---- Batch Size > 1 through a real driver ----------------------------------------------------------------------------
def test_moss_batch_of_variants_trains(tmp_path):
    from fizgig.moss_image.model import MOSSImageDiT
    s = tv.TagSettings.from_params(shuffle=True, tag_dropout=0.3, caption_dropout=0.2, variants=4)
    caps = [TAGS, "zwxem, 1boy, hat, beach", "zwxem, cat, sofa, window, night, lamp"]
    infos = []
    for i, cap in enumerate(caps):
        lat = tmp_path / f"img{i}_lat.safetensors"
        save_file({"latent_4x4": torch.randn(16, 4, 4, generator=torch.Generator().manual_seed(i))}, str(lat))
        te = tmp_path / f"img{i}_t_te.safetensors"
        save_file({"cond__hidden_states": _enc_one(cap, 48)["hidden_states"]}, str(te), metadata={"caption1": cap})
        tv.write_variants(lambda ts: [_enc_one(t, 48) for t in ts], str(te), cap, s)
        it = ItemInfo(f"img{i}", cap, (64, 64), (64, 64), latent_cache_path=str(lat))
        it.text_encoder_output_cache_path = str(te)
        infos.append(it)
    bm = BucketBatchManager({(64, 64): infos}, 3)
    bm.variants = tv.VariantPicker(s, seed=5)
    torch.manual_seed(0)
    dit = MOSSImageDiT(dim=64, ffn_dim=96, num_heads=4, num_layers=2, text_dim=48, freq_dim=32, latent_channels=16)
    drv = get("moss_image").load_driver()
    seen = set()
    for e in range(1, 6):
        b = _epoch(bm, e)[0]
        seen |= set(b["caption_pick"])
        loss, info = drv.training_loss(dit, b["latents"], drv.batch_cond(b, "cpu"), torch.Generator().manual_seed(e))
        loss.backward()
        assert torch.isfinite(loss)
    assert len(seen) > 1


# ---- the cache pass over a real dataset --------------------------------------------------------------------------------
def _dataset_toml(tmp_path, desc_key="sdxl", **vals):
    from PIL import Image
    img = tmp_path / "pics"
    img.mkdir()
    for i, cap in enumerate([TAGS, "1girl, zwxem, hat, beach, sea", PROSE]):
        Image.new("RGB", (64, 64), (i * 40, 0, 0)).save(img / f"p{i}.png")
        (img / f"p{i}.txt").write_text(cap, encoding="utf-8")
    inputs = {"megapixels": "0.25", "image_folder": str(img), "cache_root": str(tmp_path / "cache"),
              "batch_size": 1, "DOP_FOLDER_TRIGGER": "zwxem", **vals}
    return launch.dataset_toml(get(desc_key), inputs)


def test_dataset_toml_and_cache_pass(tmp_path):
    import toml
    from fizgig.dataset.config import BlueprintGenerator, ConfigSanitizer, generate_dataset_group_by_blueprint
    text = _dataset_toml(tmp_path, TAG_SHUFFLE=True, TAG_KEEP_FIRST="1", TAG_DROPOUT="10", CAPTION_DROPOUT="5%",
                         TAG_VARIANTS="4")
    cfg = toml.loads(text)
    assert cfg["general"]["tag_shuffle"] is True and cfg["general"]["tag_dropout"] == 10
    assert cfg["general"]["caption_dropout"] == 5 and cfg["general"]["caption_variants"] == 4
    assert cfg["datasets"][0]["keep_words"] == ["zwxem"]
    import argparse
    bp = BlueprintGenerator(ConfigSanitizer()).generate(cfg, argparse.Namespace(), architecture="sdxl")
    ds = generate_dataset_group_by_blueprint(bp.dataset_group).datasets[0]
    st = ds.tag_settings
    assert st.shuffle and st.keep_first == 1 and abs(st.tag_dropout - 0.1) < 1e-9 and abs(st.caption_dropout - 0.05) < 1e-9
    assert st.keep_words == ("zwxem",)
    enc = _Encoder()
    # the item text caches (the shared cache pass writes these first)
    for b in ds.retrieve_text_encoder_output_cache_batches(1):
        for it in b:
            os.makedirs(os.path.dirname(it.text_encoder_output_cache_path), exist_ok=True)
            _te(it.text_encoder_output_cache_path, it.caption)
    out = tv.build_for_datasets([ds], enc, arch="sdxl", batch=4)
    assert out["items"] == 3 and out["prose"] == 1 and out["encoded"] > 0
    files = sorted(os.path.basename(p) for p in (tmp_path / "cache").rglob("*_tags.safetensors"))
    assert files == ["p0_sdxl_tags.safetensors", "p1_sdxl_tags.safetensors", "p2_sdxl_tags.safetensors"]
    # the trigger leads p1's variants, though its caption has it second
    md = tv._header(str(next((tmp_path / "cache").rglob("p1_sdxl_tags.safetensors"))))[0]
    assert all(c.startswith("zwxem, 1girl") for c in json.loads(md["captions"]))
    # the prose caption: no variants, only the empty caption for dropout
    md2 = tv._header(str(next((tmp_path / "cache").rglob("p2_sdxl_tags.safetensors"))))[0]
    assert json.loads(md2["captions"]) == [] and md2["empty"] == "1"
    assert tv.build_for_datasets([ds], enc, arch="sdxl")["encoded"] == 0      # a second pass: all reused
    # a cache file the resume check reads as a latent? never
    assert launch.uncached_pictures(text) == 3                         # no latents yet: all three still need one


def test_launch_rules():
    on = {"TAG_SHUFFLE": True, "TAG_DROPOUT": "10", "CAPTION_DROPOUT": "0", "TAG_KEEP_FIRST": "2", "TAG_VARIANTS": "8"}
    for k in ("sdxl", "anima", "krea2", "moss_image"):
        assert launch.tag_settings(get(k), on)["caption_variants"] == 8, k
    assert launch.tag_settings(get("minimax"), on) is None           # its own text cache and caption dropout
    if get("acestep") is not None:
        assert launch.tag_settings(get("acestep"), on) is None
    assert launch.tag_settings(get("sdxl"), {}) is None               # all off by default
    assert launch.tag_settings(get("sdxl"), {"CAPTION_DROPOUT": "5"})["caption_dropout"] == 5
    assert launch.tag_problems(get("sdxl"), {**on, "TAG_DROPOUT": "lots"})
    assert launch.tag_problems(get("sdxl"), {**on, "TAG_DROPOUT": "150"})
    assert not launch.tag_problems(get("sdxl"), on)
    assert "tag_shuffle" not in launch.dataset_toml(get("sdxl"), {"megapixels": "0.25", "image_folder": "",
                                                                  "batch_size": 1})


def test_form_card_and_cost_note():
    sec = {s["id"]: s for s in form.form(get("sdxl"))["sections"]}
    assert [r["key"] for r in sec["captions"]["rows"]] == list(launch.TAG_KEYS)
    assert all(not form.DEFAULTS[k] or form.DEFAULTS[k] in ("0", "1", "4") for k in launch.TAG_KEYS)
    assert "captions" not in {s["id"] for s in form.form(get("minimax"))["sections"]}
    st = form.state(get("sdxl"), dict(form.DEFAULTS))
    assert {"TAG_KEEP_FIRST", "TAG_VARIANTS"} <= set(st["disabled"]) and "TAG_VARIANTS" not in st["notes"]
    st = form.state(get("sdxl"), {**form.DEFAULTS, "TAG_SHUFFLE": True, "CAPTION_DROPOUT": "5"})
    assert "5 encoding(s) a caption" in st["notes"]["TAG_VARIANTS"]
    assert form.tag_cost_note({"TAG_SHUFFLE": True, "TAG_VARIANTS": "4"}, 30).count("120 extra") == 1
    st = form.state(get("sdxl"), {**form.DEFAULTS, "FAMILY_KIND": "Edit (original + edited photo pairs)"})
    if get("sdxl").edit_training:
        assert "captions" in st["hidden"]
