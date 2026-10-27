"""The JTP-3 Hydra network, vendored for jtp3_tagger.py (Image Prep's "Trim tags that don't fit").

From Project RedRocket's Hydra repository, https://huggingface.co/RedRocket/Hydra (formerly RedRocket/JTP-3), revision
cfa9b0a1ffcf2b8df8553be7673210fd60fba23b (23 Aug 2026). Copyright Project RedRocket (RedHotTensors and contributors).
Licensed under the Apache License, Version 2.0 (the repository's licence: apache-2.0). Only the network modules are
taken: cufork.py, glu.py, head.py, pool.py and siglip2.py. Their image loader (hydra/image.py, which needs pyvips),
model loader, calibration and GUI are not; jtp3_tagger.py loads the weights and prepares pictures itself with Pillow.

Fizgig changes, marked "Fizgig change" in the files: they import and run on Python 3.10 (typing.Self and NotRequired
fall back, and siglip2.py's three `x[*idxs]` subscripts are spelled `x[tuple(...)]`). Nothing else is altered."""
