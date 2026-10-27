# Masked training

Train on the part of each picture that matters: the person, not the room around them. A black-and-white mask beside a picture says which part counts. White is trained fully, grey partly, and black only as much as you choose. The picture itself still trains whole, so the LoRA sees your subject in their scene; it just stops copying the scene.

It works for Klein, Krea 2, Qwen Image 2.1, Flux 3, MOSS-Image, Z-Image, SDXL and Anima (photos; not clips or songs).

In our tests of character LoRAs, masks significantly raised both the best likeness a run reached, by eye and in likeness scores, and how quickly it got there, on every recent model. It seems especially effective on MOSS-Image with its default Clean-end share, Slide.

## How do I…?

### …turn it on?

Training tab → **Masked training** card → tick **Train on the parts your masks show**. The line under it says how many pictures have a mask. Pictures without one train as normal.

![The Masked training card, on the Training tab](../assets/masked_training/training_card.png)

Start won't run with it ticked and no masks in the training folder.

### …make masks automatically?

Image Prep tab → **Masks for masked training** card. Pick what the masks show:

![Make masks, on the Image Prep tab](../assets/masked_training/make_masks.png)

- **The person (hair to clothes)**: for a person LoRA. The background counts far less.
- **The person, not their clothes**: a person LoRA that shouldn't learn their outfits. Clothes, accessories and background count far less.
- **Just the face and hair**: likeness only. Clothes, body and background count far less.
- **The body and clothes, not the face or hair**: an outfit or body LoRA that shouldn't learn the face. Face, hair and background count far less.
- **The body's skin, not the face, hair or clothes**: a body or skin LoRA (skin texture, tattoos) that shouldn't learn the face. Face, hair, clothes and background count far less.
- **Just the clothes**: for an outfit LoRA. Clothes and accessories train fully; the person wearing them and the background count far less.
- **Everything but the faces**: for an outfit, body or style LoRA that shouldn't learn anyone's face.
- **Just the background, not the people**: for a place or scene LoRA. The people in the pictures count far less; a picture with nobody in it trains whole.

Press **Make masks**. It runs on the CPU, a fraction of a second a picture. The first time, it downloads Google's person finder (MediaPipe Selfie Multiclass, 16 MB).

Masks already in the folder are kept unless you tick **Replace masks already there**, so a mask you painted yourself is safe.

Make masks after resizing and face crops. A crop made later carries its mask with it, but this way the masks you check are the final ones.

### …make masks of drawn, anime or furry characters?

Choose **Quick finder**, then **Who's in your pictures**:

- **Real people (photos)**: Google's person finder, the default and the same as always. It can tell hair, skin and clothes apart, but it misses most drawn and furry characters.
- **Drawn or furry characters**: a character finder made for drawings, anime and furry art (SkyTNT anime-seg, Apache-2.0). It also does well on most fursuit photos. It cuts out the whole character, fur, tail and all. Press **Download the character finder (176 MB)** once first.
- **A mix: decide for each picture**: runs both on every picture. A photo of a real person gets the person finder's mask, anything else the character finder's; the log says which each picture got.

For a character, the choices are **The whole character**, **Just the head**, **The character without the head**, **Everything but the heads** and **Just the background**. The head comes from the YOLO head model in the face settings (Image Prep → 1 · What to do → Anime/furry/drawn characters: Fdetailer when it's there, otherwise deepghs' anime head model, downloaded the first time); head masks follow a box, so their bottom edge is a straight line. The clothes choices are off for drawn pictures: the finder can't tell clothes from fur. Use SAM 3.1 from words for those (`shirt, jacket`).

**Preview on 2 random pictures** works with the quick finder too. For **Only my subject**, also set the face finder and look match to Anime/furry/drawn.

With **SAM 3.1: from words**, tick **My pictures are drawn or furry characters**: SAM then keeps finds it's 40% sure of instead of 50%, and Only my subject outlines each figure with `anthro character, furry character, animal character` instead of `person`. Ticked while Mask these says `person`, it fills those words in.

### …make masks of anything, by naming it (SAM 3.1)?

Tick **Use SAM 3.1: masks from words** on the same card. Instead of the list above you type what the masks show:

- **Mask these**: a few words for each thing, separated by commas: `person`, `hair, face`, `the dog`, `red dress`.
- **Leave these out** (optional): what to take away from that: `glasses`, `phone`, `hat`.

**Preview on 2 pictures** shows the first two pictures with their masks over them before you make any. Then press **Make masks**.

SAM 3.1 is Meta's Segment Anything model. It finds whatever you name, with sharper edges than the person finder (around hair especially). The first time, press **Download** on the card: it fetches the model (1.75 GB) into Fizgig's models folder, once, and it stays there through updates.

It runs on the GPU, a fraction of a second a picture once it has loaded; on a computer without one it takes several seconds a picture. Make masks when nothing is training. **Only my subject** works with it too.

### …keep only my subject when there are other people in the picture?

Tick **Only my subject, when there are several people**. Fizgig finds your subject by their face, using the Look Filter's 3 baselines (the last card on the Image Prep tab: choose them and scan the folder first). The other people count as background.

### …make masks by hand, or use masks from another trainer?

Save a black-and-white picture beside each photo, named after it with `-masklabel` added:

```
beach.jpg
beach-masklabel.png
```

Paint white over what you want learned and black over the rest; grey for in between and soft edges. A mask smaller than its picture is fine (it's stretched to fit), as is one painted white on a transparent layer (transparent counts as black). Other trainers that use this naming make masks that work as they are.

### …check a mask?

Captions tab: a picture with a mask has a **Show mask** button.

### …choose how much the background counts?

**How much outside the mask counts**, under the tick. 0.1 (recommended) keeps a little of the background, so the LoRA still knows your subject sits in a scene. 0 ignores it completely; then a very small mask trains hard on a few spots.

### …keep the model's own look outside the mask?

**Outside the mask, keep the model's own picture (prior preservation)**, under the tick. Off, the part outside the mask still learns your photos a little (as much as the setting above). On, it learns to draw there what the model itself would draw: at each step Fizgig also asks the model, with your LoRA switched off, what it sees outside the mask, and teaches the LoRA to give the same answer. So the LoRA learns your subject and leaves backgrounds and the model's own look alone. The number is how firmly; 1 counts the outside as much as the inside. It still works with **How much outside the mask counts** at 0: that stops the outside learning your photos, and this keeps it to the model's own picture.

Each step takes longer: one extra pass through the model, without the LoRA. In our tests that added from a few percent to about half again, depending on the model. It's for LoRAs: a slider or a fine-tune trains without it. This is known as masked prior preservation.

## What to watch out for

**A mask Fizgig won't use.** These are named under the tick and in the training log, and their pictures train without a mask:

- **All black.** It leaves nothing to train on.
- **Not its picture's shape.** The picture was cropped or rotated after the mask was made.

**Masks go with their pictures.** Remove on the Captions tab, Move Marked in the Look Filter, Image Prep's resize and face crops, Move a run and a training share all take a picture's mask along.

**Each picture counts the same.** A picture with a small mask isn't trained more weakly than one with a big mask: the loss is scaled so every picture's weight is the same as without masks.

**The saved LoRA says it was masked.** Its metadata carries `ss_masked_training`, `ss_mask_floor` and how many pictures had a mask.
