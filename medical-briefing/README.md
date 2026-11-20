# medical-briefing (app folder)

This folder is the Condition Research Briefing app. See the [root README](../README.md) for what it does, how it works, how to run and test it, and how to deploy it.

Folder-specific note: refresh the prebuilt quick-pick corpora with

```bash
python scripts/fetch_sources.py
```

The script does not set a browser `User-Agent` (CDC returns HTTP 403 to one) and keeps NCBI calls at or under 3 per second.
