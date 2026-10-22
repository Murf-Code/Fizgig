# Fizgig v8.8.2: maintenance

- **The web UI no longer freezes when you open it during a long run.** A new tab, or the page after your computer wakes from sleep, replays the run's log, and on a long run the console redrew itself thousands of times in a row, so the tab stopped responding (the training itself carried on). The same replay now takes a fraction of a second.

**On a RunPod pod:** restart the pod to update.
