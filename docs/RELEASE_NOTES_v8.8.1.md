# Fizgig v8.8.1: better MOSS-Image likeness, quicker previews that look like SFT

**▶ [Watch: Training MOSS-Image LoRAs in Fizgig 8.8.1](https://www.youtube.com/watch?v=UCoCtM8yrBk)** (2 minutes)

- **MOSS-Image learns likeness sooner and holds it.** Clean-end share now starts high and eases down: **Slide**, 60% of training steps on nearly-clean images until epoch 7, then a few points less each epoch to 30% by epoch 15, and from then on every 2nd epoch goes back to 60% for one epoch (a tick, on by default), which keeps likeness from fading on a small dataset. Likeness comes in early, then composition. On the dataset that had given us the most trouble, it reached 72% likeness by epoch 20 and still held 71% at epoch 60; the best of eight settings we tried there. It works well with datasets of up to about 55 images; with many more, start dropping below 60 earlier. Every MOSS-Image preset uses it. The Clean-end share control (Training tab) has two modes: **Slide** and **Single** (one value for the whole run).
- **Give Base previews the SFT look** (Samples tab): with **Use the SFT model for samples** off, Fizgig's SFT look LoRA renders your Base previews with much of MOSS-Image SFT's look. It's not as good as SFT itself, but a really good indicator, and previews are much quicker, with no model swap. It's used for previews only, never in training. The first time you tick it, Fizgig downloads it for you (about 2.1 GB), with progress under the tick. **We still recommend not using the Turbo LoRA for previews.**
- **Each enlarged preview says how it was made.** Click a MOSS-Image preview in the gallery and a line above it says what it was rendered with: SFT, the Base with the SFT look LoRA, or the Base alone, and whether the Turbo LoRA was on, with what gives a better read of quality.
- **Update ComfyUI.** Many people tell us their Fizgig LoRAs work better since ComfyUI's update this week.
- **Fixed:** with two Fizgig web servers running at once, starting the second one marked a run on the first as stopped in Runs & outputs. The training itself carried on.
- **Fixed:** the folder boxes that show the training folder (an Edit run's **Edited folder**, a slider's **+1 end folder**) can now be changed where they are, with **Choose…**: it changes the training folder on the Start tab too. Thanks @tudal for the report (#217).
- **Fixed:** in LoRA Royale, a comparison sheet of more than 24 pictures renders again: the **Render it** button no longer disappears before you can press it. Thanks @Nessa2026691 for the report and the fix (#215).
- **Krea 2 on 16 GB cards: Auto no longer picks a slow setup.** **Base precision: Auto** now takes NF4 with no block swap whenever it fits, instead of INT8 with a lot of block swapping, which used far more CPU and can be much slower. At 1 MP, Krea 2 training on NF4 sits close to the limit of a 16 GB card that also runs your display, and going over makes it slower and uneven: set **Weight averaging (EMA)** to Off (about 0.5 GB less) or train at 0.75 MP. Thanks @mikehdt for the detailed report (#216).

## Updating from an earlier version

- **Your saved MOSS-Image form keeps its Clean-end value.** To use Slide, pick a MOSS-Image preset, or pick **Slide** on the Clean-end share control.
- **On a RunPod pod:** restart the pod to update.
