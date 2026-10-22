# Fizgig v8.7.6: cache fixes

- **A resume rebuilds a cache you deleted.** If you deleted cache files while a run was paused, Resume stopped at once with "No training items". It now caches any pictures whose cache is missing, then carries on. A resume with its cache intact starts as fast as before. Thanks to **[@mart-hill](https://github.com/mart-hill)** for the report (#210).
- **A folder refilled with someone else's photos trains on the new photos.** Fizgig re-encoded a photo only when it was newer than its cache, and photos copied into a folder keep their own dates, so older photos under the same file names kept the previous person's cache. Each cache now records its photo's size and date, and any change re-encodes it.
- **MOSS-Image's Samples tab** has a note above the Prompts box: MOSS-Image doesn't respond well to short one-line prompts, and they can make the previews look worse.
- **Subjects card:** **Leave the rest of the model alone** now says it isn't ideal with just one person; use **Off** for a one-person LoRA.

**On a RunPod pod:** restart the pod to update.
