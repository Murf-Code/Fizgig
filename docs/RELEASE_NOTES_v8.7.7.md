# Fizgig v8.7.7: Kromoss out, a preview schedule

- **Kromoss is no longer offered for MOSS-Image, as the preview model or as a base to train on.** As the preview model, its previews were unreliable: it's too different from MOSS-Image Base, so a LoRA trained on Base doesn't show reliably on it. As a base to train on, it overcooks: in our test a LoRA trained on Kromoss began to overbake after about 750 steps, where the same training on Base held up. Previews render on **MOSS-Image SFT** again, and you train on **MOSS-Image Base**.
- **The 8-step Turbo LoRA stays,** for fast previews of your Base: untick **Use the SFT model for samples** on the Samples tab and set **Turbo LoRA strength** to 1 (12 steps, CFG 1). You can turn it on mid-run: it joins from the next preview.
- **Preview less often later in a run, on every model.** Under the Samples tab's **Every N epochs**, tick **Then preview less often** and set **After epoch** and **Preview every N epochs**: every 1 epoch up to epoch 10, then every 5, previews epochs 1-10, then 15, 20, 25. Off by default, and its boxes stay greyed until you tick it. Resume picks up changes, and so does **Change the previews of the run going now** while a run trains.
- **MOSS-Image on a small dataset:** with fewer than 50 pictures, the Training tab notes beside **Max Epochs** that MOSS-Image can take longer to converge than other models, so allow more epochs before judging it.

## Updating from an earlier version

- **Set Kromoss on a MOSS-Image row? Put it back.** If you changed the **preview model** row from SFT to Kromoss, or the **Base DiT** row from Base to Kromoss, go to Preferences → MOSS-Image and press **Download models for me**. It points each row back (Kromoss → SFT on the preview row, Kromoss → MOSS-Image Base on the Base row) and downloads whichever of the two you don't already have. Your Kromoss files stay on disk, and a Base row pointing at your own fine-tune is left alone. Then check the Samples tab: **Use the SFT model for samples** should be ticked.
- **On a RunPod pod:** restart the pod to update, then do the same in Preferences.
