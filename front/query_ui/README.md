# Dataset Viewer Workbench

Frontend workbench for fully exercising a dataset-viewer deployment.

## What this UI covers

- Full endpoint explorer for:
  - `/is-valid`
  - `/splits`
  - `/info`
  - `/first-rows`
  - `/rows`
  - `/search`
  - `/filter`
  - `/parquet`
  - `/size`
  - `/statistics`
  - `/croissant-crumbs`
  - `/opt-in-out-urls`
  - `/presidio-entities`
- Dataset profile mode (multi-endpoint summary in one click)
- Tabular rendering for row-based and list-based responses (`/rows`, `/first-rows`, `/search`, `/filter`, `/splits`, `/parquet`, `/statistics`, `/size`)
- OpenAPI-driven endpoint metadata refresh from your deployment (`/openapi.json`)
- Saved presets (store and reload endpoint/query configurations)
- Format inspector for source files (CSV/JSON/Parquet/image/audio/etc.) via Hub metadata
- Auto-resolve renamed dataset IDs through the Hub API
- Optional polling for `ResponseNotReady`
- Optional bearer token for private/gated datasets
- Optional insecure TLS mode for self-signed certificates
- Equivalent `curl` output for every request

## Why this supports more datasets

Datasets on the Hub can be stored as CSV, JSON/JSONL, Parquet, images, audio, and more.
This frontend supports broad compatibility by combining:

- Dataset-viewer API checks (`/is-valid`, `/splits`, `/first-rows`, etc.)
- Source file-type detection through Hub dataset metadata
- Endpoint-level controls so you can choose preview/search/filter/statistics depending on what is available for that dataset

For non-parquet source datasets, dataset-viewer can still support them through backend conversion and cached processing steps.

## Local run

From `front/query_ui`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

## Docker run

From `front/query_ui`:

```bash
docker build -t dataset-viewer-workbench .
docker run --rm -p 7860:7860 dataset-viewer-workbench
```

Open `http://<your-ec2-public-ip>:7860`.

## Example EC2 deployment (Docker Compose)

This repository is already wired to run the UI as `query-ui` and proxy it through nginx.
After pulling the latest changes on EC2, run:

```bash
cd ~/dataset-viewer
docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d --build query-ui reverse-proxy
docker-compose -f docker-compose.ec2.yml --env-file .env.production ps
```

Access through your existing public endpoint:

- `https://54.198.247.234/query-ui/`

If you want the minimal compose snippet, it looks like:

```yaml
services:
  query-ui:
    build:
      context: ./front/query_ui
      dockerfile: Dockerfile
    restart: always
    expose:
      - "7860"
```

## Presets storage

Presets are saved to:

- `front/query_ui/presets.json`

You can override this location with:

- `QUERY_UI_PRESETS_FILE=/path/to/presets.json`

## Suggested settings for your current deployment

- `Dataset Viewer Server`: `https://54.198.247.234`
- `Auto-resolve renamed dataset IDs`: enabled
- `Skip TLS verification`: enabled until certificate trust is fully configured
