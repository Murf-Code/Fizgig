# Fizgig v8.7.9: maintenance

- **MOSS-Image's workbench renders where its previews do.** Repair Studio, LoRA the Explorer, LoRA Royale and the Profiler render on MOSS-Image SFT while **Use the SFT model for samples** is ticked on the Samples tab, and on the Base when it isn't. They used to render on SFT whenever its file was set. Nothing changes for other models.
- **MOSS-Image renders with a LoRA that covers the whole model.** A LoRA extracted from a full fine-tune (Extract → From a fine-tune) also adapts MOSS-Image's output layer, and previews with it failed with "'LoRALinear' object has no attribute 'weight'". They render now.
- **Extracting from a MOSS-Image fine-tune keeps the text adapter.** Its two blocks were left out of the extracted LoRA, so Repair Studio's Text adapter sliders had nothing to work on. Extract again to include them.

**On a RunPod pod:** restart the pod to update.
