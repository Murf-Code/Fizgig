# Fizgig v8.8.0: maintenance

- **Resuming a run works on Python 3.10.** Check settings on a resume failed with "ModuleNotFoundError: No module named 'tomllib'" on installs running Python 3.10. Pick the state folder under Memory & Precision, raise Max Epochs, and the run carries on.

**On a RunPod pod:** restart the pod to update.
