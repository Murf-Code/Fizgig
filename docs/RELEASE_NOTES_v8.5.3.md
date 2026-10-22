# Fizgig v8.5.3: Linux fixes

**Linux: Fizgig no longer stops by itself when it opens in the browser.** Without a window engine, launch_fizgig.sh opens the web UI in your browser instead of its own window, but the web UI then shut down about two minutes later. The file picker, the model downloader and saving Preferences stopped working, and refreshing gave "unable to connect". It now stays up until you stop it.

**Linux: the app window works straight after install.** The installers (NVIDIA and AMD) and the updater now install Qt for the app window inside Fizgig's venv, so `./launch_fizgig.sh` opens Fizgig in its own window without installing anything else. If Qt can't install on your system, Fizgig opens in your browser instead, and the message says how to get the window.

Already installed on Linux (NVIDIA)? Run `git pull` once, then `./update_fizgig.sh`, the new Linux updater; use it for every update from now on. On Linux AMD, run `./install_fizgig_rocm.sh` again.

Thanks to u/WiseDuck on Reddit for the report.

---

## Fizgig v8.5.2: maintenance

**MOSS-Image presets: Detect problem images and per-image LR are off.** I'm turning them off until I've had a chance to test them further on this model. Both are still on the Training tab if you want them.

**Fixes and small improvements:**
- **Captions tab:** an image model's caption list no longer shows the folder's recordings and clips. Training already skipped them.
- **Samples gallery:** starting a run on a new dataset no longer shows an error for the previous run's likeness photos; pick three from the new dataset when you want scoring.
- **File picker:** Name, Date modified and Size columns, sorting by name, date modified, date created, size or type (click a column to sort, again to reverse), and a filter box. Folders stay first and your sort is remembered.
- **Start tab:** the Upload a training folder card only shows when you're not on the machine running Fizgig (a pod, your phone or another computer). On the machine itself, Choose… picks the folder directly.

---

## Fizgig v8.5.0: MOSS-Image

**MOSS-Image is now in Fizgig.** It's OpenMOSS's new 5B open image model (Apache-2.0), with a large world knowledge, strong prompt following and good text rendering. Fizgig trains LoRAs on MOSS-Image Base, the checkpoint OpenMOSS recommend for training, and previews them on MOSS-Image SFT, the one most people generate with.

Thanks to @Murf-Code, whose pull request brought MOSS-Image to Fizgig, and to @l1934332574-maker for the careful scope discussion that started it.

**Getting started:** pick MOSS-Image on the Training tab, then press Download models for me in Preferences. It fetches the Base model, the SFT model for previews, the Qwen3.5-9B text encoder and the FLUX.2 autoencoder (about 35 GB in all). Already have the files from ComfyUI? Point the paths at them instead.

**What you get:**
- **Every LoRA kind:** Standard LoRA, LoKR, sliders (from photo pairs or from prompts), full fine-tune, Multi Concept, DOP and Fizgig LoRA Isolation.
- **The workbench:** Repair Studio (with Turbo Preview), LoRA Royale, the Profiler and Extract.
- **Previews on SFT:** training samples and the workbench render on the SFT model, so you see your LoRA where you'll use it. A tick on the Samples tab switches samples back to the Base model being trained.
- **Better sampling settings:** previews use res_multistep at 30 steps and CFG 4, picked from side-by-side tests against the reference's 50-step Euler. The small FLUX.2 decoder MOSS-Image ships with avoids the spots the full decoder can leave.

**Presets:** Fast (rank 16, learning rate 1e-4, 50 epochs at 0.25 MP) is the recommended start and gives excellent results. Standard (rank 32), Style (rank 16 at 5e-5), Slider and Fine-tune are there too. They use a fixed learning rate, which suits MOSS-Image better than Adaptive LR; Adaptive LR is still on the Training tab if you want it.

**Memory:** Auto trains on an INT8 base (NF4 on smaller cards) and compiles the blocks for about 25% faster steps on longer runs. The text encoder runs at full precision on any card: without room for it, it streams through system memory one layer at a time, so even an 8 GB card caches captions exactly.

**Optional Turbo LoRA:** Preferences has an empty row for a turbo LoRA (lodestones' 8-step DMD LoRA today) if you want faster, softer previews. Set it, then raise Turbo LoRA strength on the Samples tab and set Steps and CFG for it.

Guide: [MOSS-Image in Fizgig](https://github.com/shootthesound/Fizgig/blob/master/docs/MOSS_IMAGE.md) · CLI: [docs/CLI.md](https://github.com/shootthesound/Fizgig/blob/master/docs/CLI.md)
