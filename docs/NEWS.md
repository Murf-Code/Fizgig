# Fizgig news

## 20 Oct 2026: MOSS-Image likeness sooner, and Base previews with the SFT look

**▶ [Watch: Training MOSS-Image LoRAs in Fizgig 8.8.1](https://www.youtube.com/watch?v=UCoCtM8yrBk)** (2 minutes)

**A new Clean-end share default for MOSS-Image: Slide.** It puts 60% of training steps on nearly clean images, where faces and fine detail are learned, until epoch 7, eases down to 30% by epoch 15, and then sends every 2nd epoch back to 60% so likeness doesn't fade. Likeness comes in early, then composition. On the dataset that had given us the most trouble, it was the best of eight settings we tried: around 70% ArcFace likeness from epoch 20 to epoch 60, a very high score (different real photos of the same person often score lower). It's tuned on datasets of up to about 55 images; with many more, a steady 40% with **Single** may suit better. Every MOSS-Image preset uses Slide. A saved form keeps its old value: pick a preset, or pick **Slide** on the Clean-end share control.

**Give Base previews the SFT look** (Samples tab): with **Use the SFT model for samples** off, Fizgig's SFT look LoRA gives your Base previews much of MOSS-Image SFT's look, with no model swap, so previews are much quicker. It's not as good as SFT itself, but a good guide. It downloads itself the first time you tick it (about 2.1 GB). **We still recommend not using the Turbo LoRA for previews.**

**Each enlarged preview says how it was made:** SFT, the Base with the SFT look LoRA, or the Base alone, and whether the Turbo LoRA was on.

**Update ComfyUI.** Many people tell us their Fizgig LoRAs work better since ComfyUI's update this week.

## 20 Oct 2026: MOSS-Image Turbo previews: fast, but don't judge likeness on them

**The MOSS-Image Turbo LoRAs aren't mature yet: they can reduce likeness a lot in previews.** Several people judged a run's likeness on Turbo previews and thought it wasn't learning. Use the Turbo LoRA to check quickly that a run is on track, and judge likeness on full-quality previews: set **Turbo LoRA strength** to 0 on the Samples tab, and Steps and CFG go back with it. With **Change the previews of the run going now** ticked, that applies from the run's next preview. The gallery's amber Turbo line now says this too.

**New for every model: How often to preview and How often to save.** Each is one control: **Every** N epochs, and, switched on, **then from epoch** Y, **every** Z epochs. Switching it on fills in every 5, then from epoch 10 every 1: few previews or checkpoints early, every epoch once the run gets close. The last epoch is always saved, and a fine-tune keeps saving at whole rotations.

## 20 Oct 2026: Kromoss out as a preview model, the Turbo LoRA stays

**Kromoss is no longer offered as the MOSS-Image preview model.** It's too different from MOSS-Image Base for reliable previews of a LoRA trained on Base: side-by-side renders of the same LoRA, prompt and seed on SFT, Kromoss and Base showed it. Previews render on **MOSS-Image SFT** again, and the Preferences row is back to **MOSS-Image SFT DiT (previews)**.

**If you set Kromoss on a MOSS-Image row** (the preview model in place of SFT, or the Base DiT in place of Base), go to Preferences → MOSS-Image and press **Download models for me**: it points each row back to SFT and MOSS-Image Base, downloading whichever you don't have. Your Kromoss files stay on disk.

**Train on MOSS-Image Base, not on Kromoss:** in our test, a LoRA trained on Kromoss began to overbake after about 750 steps (epoch 12 on a 62-image dataset), where the same training on Base held up.

**The Turbo LoRA stays.** lodestones' 8-step Turbo LoRA, distilled from MOSS-Image Base, gives fast previews of your Base: **Download** on the Turbo LoRA row in Preferences, untick **Use the SFT model for samples** on the Samples tab, and set **Turbo LoRA strength** to 1. Steps move to 12 (much better than its card's 8) and CFG to 1. You can turn it on mid-run too: it joins from the next preview. Previews on SFT never take it.

## 19 Oct 2026: MOSS-Image training, fixed

**MOSS-Image LoRAs now learn likeness much sooner, and the picture holds its composition from epoch to epoch.** There's nothing to set: update Fizgig and it's the default.

**What was wrong.** A diffusion model trains at every noise level, from nearly clean pictures to pure noise. The clean end is where faces and fine detail are learned; the noisy end is layout and composition. MOSS-Image's own training schedule puts only about 11% of steps at the clean end. That's why MOSS LoRAs were slow to pick up likeness, and why the picture jumped about between epochs.

**What Fizgig does now.** A new Training-tab setting, **Clean-end share**, puts 30% of training steps at the clean end, close to where Krea 2 and Qwen Image sit, the models that pick up likeness fastest in my own training. It changes training only; how MOSS-Image generates pictures is unchanged.

ArcFace likeness against 3 reference photos, same dataset:

| Epoch | 3 | 6 | 9 | 12 | 15 | 20 |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Before (the model's own schedule) | 40% | 44% | 52% | 58% | 63% | 60% |
| Now (Clean-end 30%) | 42% | 59% | 63% | 66% | 69% | 69% |

The old settings peaked at 67% after 30 epochs. On the Standard preset (rank 32, 1 MP) a second dataset reached 72% by epoch 32, above the 71% that the dataset's own photos score against the same reference photos.

**By eye the difference is bigger still: the pictures are far sharper and more accurate, with no body horror, and the composition holds steady instead of hopping between epochs.**

**Fine-tunes too.** A MOSS-Image fine-tune uses the same Clean-end share, and it fits a 16 GB card, the whole model at once: a full Base fine-tune peaked at 12.6 GB at 1 MP.

30 is still a figure to explore; share what you find in the [Reddit discussion](https://www.reddit.com/r/StableDiffusion/comments/1x9khtm/a_potential_fix_for_moss_training_issues/). Empty the box for the model's own schedule.

