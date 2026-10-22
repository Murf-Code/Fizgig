# Fizgig v8.7.8: preview and save schedules, a Turbo likeness warning

- **MOSS-Image Turbo previews can reduce likeness a lot.** The MOSS-Image Turbo LoRAs aren't mature yet, so judge likeness on full-quality previews: set **Turbo LoRA strength** to 0 on the Samples tab. The samples gallery's amber Turbo line now says so for MOSS-Image.
- **How often to preview** (Samples tab) is now one control: **Every** N epochs, and, switched on, **then from epoch** Y, **every** Z epochs. Switching it on fills in every 5, then from epoch 10 every 1. It replaces 8.7.7's **Then preview less often**, and works either way: fewer previews later, or more. Resume and **Change the previews of the run going now** both pick up changes.
- **How often to save** (Training tab) works the same way for checkpoints. The last epoch is always saved. A fine-tune still saves at whole rotations, so it doesn't use it. Resume picks up changes.
- **Fizgig LoRA Isolation needs two or more subjects.** With one subject it's greyed out, with a note under the choices saying why; **Off** and **Leave the rest of the model alone** stay. A saved one-subject form on Isolation moves to Leave the rest of the model alone, with a note. The Subjects card's intro is clearer too.
- **On a phone,** the new controls fold onto extra lines instead of running off the screen.

**On a RunPod pod:** restart the pod to update.
