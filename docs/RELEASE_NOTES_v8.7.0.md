# Fizgig v8.7.0: MOSS-Image Clean-end share

**MOSS-Image likeness arrives much sooner.** It's 2:43 in the morning here as I write this. I went to bed, then got up again, because something from my MiniMax H3 training work wouldn't leave me alone. On H3 I had to move the share of training steps at the clean end of the noise range, where faces and fine detail get refined, before likeness came good. So I compared MOSS's training sigmas with Krea 2's, and it was like looking at MiniMax all over again. MOSS's own schedule puts only about 11% of steps at the clean end (sigma below 0.5), against 18–28% for Krea 2. The rest go to layout and composition, which explains the big jumps from epoch to epoch and why it's been so hard to nail the likeness.

A new Training-tab option, **Clean-end share**, now sets that share, and it's on by default at 60%. My test run is going as I write: the previews are holding their composition between epochs, and at epoch 12 it has already passed the best likeness my previous settings reached in 30 epochs. 60 is an experimental figure, so please play with it and share what you find in the Reddit discussion linked under the option. As a community we can find the right setting: we're starting at 60 and will probably come down, and 30 (about where Krea 2 sits) is a sensible next try. Empty the box to train on the model's own schedule. Update Fizgig and retrain to get it.

**MOSS-Image Style preset: EMA off.** A style LoRA at 5e-5 learns slowly enough on its own, so the Style preset no longer adds EMA.

---

## Fizgig v8.6.1: Qwen Image 2.1 previews fixed

**Qwen Image 2.1 previews are no longer noisy.** With the Turbo LoRA set, the Samples tab could put Steps and CFG back to the plain model's 25 steps and CFG 3 while previews still used the Turbo LoRA, which made them very noisy. Steps and CFG now always match what the previews render with: the Turbo LoRA's 8 steps at CFG 1, a ticked Turbo checkpoint's own settings, or the plain model's 25 steps at CFG 3. A form saved with the wrong values is corrected the next time you open it, and a value you typed yourself is left alone. MOSS-Image, which has the same two settings, is covered too. Update Fizgig, then restart (or resume) a run that was affected. Thanks to the Reddit user who reported it.

---

## Fizgig v8.6.0: improved MOSS-Image likeness

**Improved MOSS-Image likeness.** LoRAs now train all 42 of MOSS-Image's blocks. Block 22 used to be left out, as AI Toolkit does; in side-by-side runs on the same dataset, including it improved likeness from the first epochs on. Update Fizgig and retrain to get it; LoRAs you already have work as before.

**Training on MOSS-Image SFT.** A new tick on the Training tab, **Base is SFT or a fine-tune of it**, is for training on SFT or a model fine-tuned from it: the LoRA then trains attention only, since training SFT's feed-forward layers breaks anatomy within a few epochs. In our tests so far Base trains more reliably than SFT, so the recommended way stays the same: train on Base, then use the LoRA on SFT in ComfyUI.

**MOSS-Image Fast preset.** The Fast preset is back: Standard's settings at 0.25 MP and rank 16, for quicker runs and smaller cards. With face-cropped datasets (Image Prep's face crop) it performs very well. In 8.5.5 earlier today I said 0.25 MP wasn't the best size for MOSS-Image, and I was wrong. Getting here has been a bit of a puzzle, and I'm sure there's still progress to be made, but it feels like we're getting there.

**MOSS-Image presets run 70 epochs.** MOSS-Image LoRAs are slow to converge: likeness often seems to level off, then keeps improving with more training. In our tests a 53-image dataset was still improving at 1,600 steps (epoch 30), so Standard and Fast now run 70 epochs, enough for a 25-image dataset to get there. Look at the late epochs in LoRA Royale before settling on an early peak.

**What you see in Fizgig is what you get in ComfyUI.** Train on Base with previews on SFT (Fizgig's defaults), then run the LoRA on SFT in ComfyUI with the same seed, resolution and sampler settings: the images match Fizgig's previews, with the same ArcFace likeness scores. It's an easy check to do yourself: in ComfyUI, use CFG 4, res_multistep with the simple scheduler, 30 steps, and SLG on, as in ComfyUI's MOSS-Image template.

**MOSS-Image previews with block swap now work.** On cards where the SFT preview model streams blocks, previews failed with "expected mat1 and mat2 to have the same dtype", and INT8 training with block swap failed the same way. Both are fixed. Thanks to **[@Ferocit](https://github.com/Ferocit)** for the report (#201).

---

## Fizgig v8.5.9: faster MOSS-Image Studio, safer state saves

**MOSS-Image renders faster in the workbench tools.** Repair Studio and the other workbench tabs keep MOSS-Image's text encoder in RAM between prompts instead of reloading it from disk each time, and, when there is room beside it on the GPU, no longer move the model off the GPU and back first. A new prompt on a 32 GB card takes about 25 s instead of 37 s, and the first render about 34 s instead of 59 s. Cards that stream the text encoder gain too (about 34 s to 27 s on 16 GB). The encoder is kept only when the computer has the RAM free (about 15 GB plus headroom); otherwise it loads as before.

**A failed resume-state save no longer ends the run.** If writing an epoch's resume state fails (a full disk, or another program such as a sync tool or antivirus touching the file), Fizgig tries again, and if that fails too it warns and keeps training. The epoch's LoRA is already saved by then; only resuming from that epoch is lost. Pausing still needs the state, so a pause stops if it can't be saved.

---

## Fizgig v8.5.8: MOSS-Image block swap fix

**MOSS-Image training with block swap now works.** A MOSS-Image run that needed block swap (most 12 GB cards) crashed on its first step with "'LoRALinear' object has no attribute 'weight'". On bigger cards the run went on, but previews, which turn block swap on for their own render, failed. Both are fixed: update Fizgig and start the run again.

Thanks to **[@russjr08](https://github.com/russjr08)** and **[@Ferocit](https://github.com/Ferocit)** for the reports (#199).

---

## Fizgig v8.5.7: fixes

**MiniMax H3 fine-tuning starts again.** Since 8.4, an H3 full fine-tune stopped at the start with "unexpected keyword argument 'held_grads'", even with DOP and Isolation off. It now starts and plans its windows as before.

**Fizgig LoRA Isolation:** the side-by-side pictures are rebuilt when you edit or replace a training photo under the same name. Before, they kept using the old version of that photo.

Thanks to **[@mabseyuk](https://github.com/mabseyuk)** for both reports.

---

## Fizgig v8.5.6: MOSS-Image status line

**MOSS-Image support is new and improving daily.** The Training tab now says so under the model picker when MOSS-Image is selected. Update Fizgig regularly to get the latest fixes.

---

## Fizgig v8.5.5: MOSS-Image at 1 megapixel

**MOSS-Image presets now train at 1 megapixel, and the Fast preset (0.25 MP) is gone.** MOSS-Image was trained at 1 megapixel (its last training stage is 1024² in mixed aspect ratios), and in our tests it doesn't hold up at lower sizes as well as other models do. 0.75 and 0.5 MP are clearly better than going down to 0.25 MP, but the best results really come at 1 MP. Standard (rank 32), Style, Slider and Fine-tune all use 1 MP now, and Standard is the first preset. 1 MP fits a 12 GB card on the INT8 base, and smaller cards on NF4 with a few blocks swapped. If you need to go smaller, lower Target Megapixels on the Training tab.

**Queue fixes:**
- After you resume a paused run, the queue no longer shows as held, under the run or in the side bar. It carries on by itself once that run finishes.
- Remove, the arrows and Start queue could sometimes miss a click, because the list redrew under the pointer. It now redraws only when the queue changes.

---

## Fizgig v8.5.4: MOSS-Image previews

**MOSS-Image previews now render the way ComfyUI does.** This fixes some issues with the previews, which weren't rendering quite the same way as ComfyUI. Training samples, the SFT previews and the workbench tools now use res_multistep on ComfyUI's simple schedule (30 steps, CFG 4), with block 22 skipped in the negative for the first half of the steps, the same as ComfyUI's MOSS-Image template with its SLG switch on. What you see in Fizgig is what you'll get in ComfyUI.

**Queue:** after you resume a paused run, the queue no longer shows as held. It carries on by itself once that run finishes.

---

## Fizgig v8.5.3: Linux fixes

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
