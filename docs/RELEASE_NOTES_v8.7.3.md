# Fizgig v8.7.3: MOSS-Image fine-tunes on 16 GB cards, fine-tune previews fixed

**A MOSS-Image fine-tune fits a 16 GB card, the whole model at once.** I've now run real fine-tunes on this, and they need far less memory than Fizgig was planning for: a full Base fine-tune at 1 MP peaked at 12.6 GB, and **Train attention only** at 8.3 GB. Fizgig now plans with these figures, so a 16 GB card trains the whole model in one go, nothing rotates, and a 12 GB card does an attention-only fine-tune the same way.

| Card | Base (all parts) | Train attention only |
|:--|:--|:--|
| 16 GB and up | The whole model at once, nothing rotates (peak 12.6 GB) | The whole model at once |
| 12 GB | 2 parts per rotation | The whole model at once (peak 8.3 GB) |
| 10 GB | 9 parts per rotation | Many small parts, with blocks streamed from system RAM (works, but slow) |

The 16 GB figures are measured, and a preview peaks lower than training (9.9 GB); the 12 GB and 10 GB rows are Fizgig's plan from the same measurements, not yet run on those cards.

**Fine-tune previews work again.** A fine-tune with a preview checkpoint set (MOSS-Image's SFT previews, on by default) had its first preview fail, and then trained on with previews off. A fine-tune now previews on the model it's training.

**"Train attention only".** The MOSS-Image tick that was called "Base is SFT or a fine-tune of it" is now **Train attention only**, since that's what it does. It's still the one to tick on SFT or a fine-tune of it, where training the feed-forward layers breaks anatomy. It also uses less VRAM, so a fine-tune fits a smaller card, but on Base it may cost quality or slow convergence.

**Samples gallery: each preview shows its own prompt.** If you change the preview prompt mid-run, the gallery now shows the prompt each preview was actually made with, not the run's first one.

---

## Fizgig v8.7.2: MOSS-Image ready for LoRA training and fine-tuning

**MOSS-Image now ships with Clean-end share at 30%.** That brings its training schedule close to Krea 2's (18-28%) and Qwen Image's (25%), the models that pick up likeness fastest in my own training, against MOSS's own 11%. This is about training only: how MOSS-Image generates images is unchanged, and you sample it exactly as before.

**Where MOSS-Image training stands now.** Since it arrived in 8.5, these are the settings we've fixed:

- Previews render the way ComfyUI does, so what you see during training is what you get.
- LoRAs train all 42 blocks, block 22 included, which gave clearly better likeness.
- The text adapter stays frozen with the text encoder, as OpenMOSS's guide advises; training it exaggerated expressions epoch by epoch.
- On an SFT base, LoRAs and now fine-tunes train attention only.
- Clean-end share puts 30% of training steps where faces and fine detail are refined, against the model's own 11%.

The result, on the same dataset, ArcFace likeness against 3 reference photos:

| Epoch | 3 | 6 | 9 | 12 | 15 | 20 |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Before (model's own schedule) | 40% | 44% | 52% | 58% | 63% | 60% |
| Now (Clean-end 30%) | 42% | 59% | 63% | 66% | 69% | 69% |

The old settings peaked at 67% after 30 epochs; Clean-end 30% passed that by epoch 15. **By eye the Clean-end 30% pictures are vastly better too: far sharper and more accurate, with no body horror.** **On the model's own schedule the composition hopped massively from epoch to epoch; with Clean-end it holds steady.**

**It holds at 1 MP too.** A second dataset on the Standard preset (rank 32, 1 MP, Clean-end 30%) reached 66% ArcFace likeness by epoch 12 and 72% by epoch 32, above the 71% that the dataset's own photos score against the same three reference photos. **By eye those epochs were superb.**

**Clean-end share applies to fine-tunes too.** A MOSS-Image fine-tune uses the same 30% as a LoRA. And with **Base is SFT or a fine-tune of it** ticked, a fine-tune now trains attention only, as a LoRA does, because training SFT's feed-forward layers breaks anatomy. Fine-tuning is probably best done from Base, though: SFT has already been taken a long way in one direction.

**Which cards can fine-tune MOSS-Image, at 1 MP.** Fizgig plans this for you: it trains as much of the model at once as your card holds, and rotates through the rest. Measured on real fine-tunes:

| Card | Base (all parts) | Train attention only |
|:--|:--|:--|
| 16 GB and up | The whole model at once, nothing rotates (peak 12.6 GB) | The whole model at once |
| 12 GB | 2 parts per rotation | The whole model at once (peak 8.3 GB) |
| 10 GB | 9 parts per rotation | Many small parts, with blocks streamed from system RAM (works, but slow) |

A fine-tune also needs plenty of system RAM; the Log says how much when it starts.

That opens MOSS-Image up properly for LoRA training as well as fine-tuning.

---

## Fizgig v8.7.1: MOSS-Image Clean-end share 30 by default

**Clean-end share now defaults to 30%.** Since 8.7.0 I've trained the same dataset at 60 and at 30, side by side. 30 starts slower, pulls ahead after about 300-400 steps and stays there, with some of the best likeness I've had from this dataset on any model. Composition and shape also come in sooner than at 60, which spends so many steps on the clean end that layout gets less attention.

| Epoch | 3 | 6 | 9 | 12 | 15 |
|:--|:-:|:-:|:-:|:-:|:-:|
| Old schedule (~11%) | 40% | 44% | 52% | 58% | 63% |
| Clean-end 60% (overnight run) | 50% | 63% | 66% | 68% | 69% |
| Clean-end 60% (today, alongside the 30% run) | 65% | 54% | 59% | 63% | 51% |
| Clean-end 30% | 42% | 59% | 63% | 66% | 69% |

ArcFace likeness against 3 reference photos, one preview per epoch, same dataset (the 30% run's previews were upper-body shots until epoch 14, headshots from 15). **By eye the Clean-end pictures were vastly better too: far sharper and more accurate, with no body horror.** **On the model's own schedule the composition hopped massively from epoch to epoch; with Clean-end it holds steady.** It's still a figure to explore, so please keep sharing what you find in the [Reddit discussion](https://www.reddit.com/r/StableDiffusion/comments/1x9khtm/a_potential_fix_for_moss_training_issues/). If the box still shows 60 from an earlier run, set it to 30. On the command line it's `--family_option lownoise_pct=30`.

**The Clean-end share box stands out in the web UI,** so it's easy to find on the Training tab.

---

## Fizgig v8.7.0: MOSS-Image Clean-end share

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
