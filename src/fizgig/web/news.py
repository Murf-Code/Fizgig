"""The Start tab's News card: newest first. Each item is a date, a title, a body (plain text, paragraphs split on a
blank line) and optional links ({"text", "href"}). The card shows the newest few; the full articles are docs/NEWS.md."""

_ARTICLE = "https://github.com/shootthesound/Fizgig/blob/master/docs/NEWS.md"

NEWS = [
    {
        "date": "20 Oct 2026",
        "title": "MOSS-Image likeness sooner, and Base previews with the SFT look",
        "body": (
            "A new Clean-end share default for MOSS-Image, Slide, brings likeness in early and keeps it: on our most "
            "difficult dataset it was the best of eight settings we tried.\n\n"
            "Give Base previews the SFT look (Samples tab) gives your Base previews much of SFT's look with no model "
            "swap. We still recommend leaving the Turbo LoRA off for previews. And update ComfyUI: many people's "
            "Fizgig LoRAs work better since its latest update."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#20-oct-2026-moss-image-likeness-sooner-and-base-previews-with-the-sft-look"},
                  {"text": "Watch the video", "href": "https://www.youtube.com/watch?v=UCoCtM8yrBk"}],
    },
    {
        "date": "20 Oct 2026",
        "title": "MOSS-Image Turbo previews: fast, but don't judge likeness on them",
        "body": (
            "The MOSS-Image Turbo LoRAs aren't mature yet: they can reduce likeness a lot in previews. Use them to "
            "check a run is on track quickly, then judge likeness on full-quality previews: set Turbo LoRA strength to "
            "0 on the Samples tab (Steps and CFG go back with it). The gallery's Turbo line now says this too.\n\n"
            "New for every model: How often to preview and How often to save. Preview every 5 epochs, say, then from "
            "epoch 10 every epoch."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#20-oct-2026-moss-image-turbo-previews-fast-but-dont-judge-likeness-on-them"}],
    },
    {
        "date": "20 Oct 2026",
        "title": "Kromoss out as a preview model",
        "body": (
            "Kromoss is no longer offered as the MOSS-Image preview model: it's too different from MOSS-Image Base for "
            "reliable previews of a LoRA trained on Base, and as a base to train on it overcooks. If you set Kromoss on "
            "the preview model or Base DiT row, press Download models for me in Preferences: it points both back to "
            "SFT and MOSS-Image Base.\n\n"
            "The 8-step Turbo LoRA stays, for fast previews of your Base: untick Use the SFT model for samples and set "
            "Turbo LoRA strength to 1 (12 steps)."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#20-oct-2026-kromoss-out-as-a-preview-model-the-turbo-lora-stays"}],
    },
    {
        "date": "19 Oct 2026",
        "title": "MOSS-Image training, fixed",
        "body": (
            "MOSS-Image LoRAs now learn likeness much sooner, and the picture holds its composition from epoch to "
            "epoch. MOSS's own schedule put only about 11% of training steps where faces and detail are learned; "
            "Fizgig's new Clean-end share puts 30% there. By eye the pictures are far sharper and more accurate, with "
            "no body horror. It's the default: there's nothing to set."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#19-oct-2026-moss-image-training-fixed"}],
    },
]
