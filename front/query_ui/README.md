# Dataset Viewer Query UI

Simple frontend for querying your deployment with renamed-dataset handling.

## Features

- Query `/splits` from your custom server URL
- Auto-resolve renamed dataset IDs via Hugging Face Hub API
- Optional polling for `ResponseNotReady`
- Optional HF Bearer token for private/gated datasets
- Optional insecure TLS mode for self-signed certs

## Run

From `front/query_ui`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open:

- http://localhost:7860

## Suggested settings for your current deployment

- `Dataset Viewer Server`: `https://54.198.247.234`
- `Auto-resolve renamed dataset IDs`: enabled
- `Skip TLS verification`: enabled (until certificate trust is fully configured)
