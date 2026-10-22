# Fizgig v8.8.3: maintenance

- **Resume uses the Learning Rate you set.** Change the Learning Rate on the Training tab before pressing **Resume** and the run carries on at the new rate, with its schedule picking up where it paused. With Adaptive LR on, a new Min / Max is used from there on. The console says when the rate changed, and with nothing changed a resume trains exactly as before. Thanks @Murf-Code (#214).
- **Per-image loss on the run card.** With **Track how each image trains** ticked (Training Parameters; it was "Detect problem images"), the run card has a **Per-image loss** button while the run trains: each image's thumbnail, how its loss is going, its verdict, and its caption to edit, re-encoded at the next epoch. **Show: Worth a look** lists the images that stay hard. Most images do their work early and are simply learned by the end, so a flagged image isn't necessarily a bad one. Works on every model with the per-image loss watch, MOSS-Image included. Thanks @CognitiveDiffusion (#213).
- **Fixed:** after switching to another training folder, the Look Filter kept the last folder's 3 baselines and wouldn't let you pick new ones until a restart. Thanks @Ezdub69 for the report and the diagnosis (#218).
- **Fixed:** continuing a MOSS-Image fine-tune kept its Clean-end schedule from the start instead of from the epochs already done, and the **Give Base previews the SFT look** tick now reaches a run that's already going, and a resume. Thanks @mabseyuk for the report.

**On a RunPod pod:** restart the pod to update.
