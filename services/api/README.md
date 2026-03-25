# Dataset viewer API

> API for HuggingFace 🤗 dataset viewer

## Configuration

The service can be configured using environment variables. They are grouped by scope.

### API service

See [../../libs/libapi/README.md](../../libs/libapi/README.md) for more information about the API configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Endpoints

See [dataset-viewer docs](https://huggingface.co/docs/dataset-viewer)

- /healthcheck: Ensure the app is running
- /metrics: Return a list of metrics in the Prometheus format
- /croissant-crumbs: Return (parts of) the [Croissant](https://huggingface.co/docs/dataset-viewer/croissant) metadata for a dataset.
- /is-valid: Tell if a dataset is [valid](https://huggingface.co/docs/dataset-viewer/valid)
- /splits: List the [splits](https://huggingface.co/docs/dataset-viewer/splits) names for a dataset
- /first-rows: Extract the [first rows](https://huggingface.co/docs/dataset-viewer/first_rows) for a dataset split
- /parquet: List the [parquet files](https://huggingface.co/docs/dataset-viewer/parquet) auto-converted for a dataset
- /opt-in-out-urls: Return the number of opted-in/out image URLs. See [Spawning AI](https://api.spawning.ai/spawning-api) for more information.
- /statistics: Return some basic statistics for a dataset split.
- /local-datasets: List datasets uploaded directly to this application.
- /local-datasets/upload (POST): Upload a local CSV/JSON/JSONL/Parquet dataset file.
- /local-datasets/{dataset_id}/info: Get metadata (format, columns, row count) for an uploaded dataset.
- /local-datasets/{dataset_id}/rows: Read paginated rows from an uploaded dataset.
- /local-datasets/{dataset_id}/search: Search text in an uploaded dataset.
- /local-datasets/{dataset_id}/filter: Apply SQL-like row filtering and ordering on an uploaded dataset.
- /local-datasets/{dataset_id} (DELETE): Delete an uploaded dataset.

## Local Dataset Security Notes

- Local dataset endpoints require `Authorization: Bearer <token>` by default.
- The requirement can be disabled with `LOCAL_DATASETS_REQUIRE_BEARER_TOKEN=false` for local development/testing.
- Uploaded datasets are namespaced by bearer token hash to avoid cross-user listing/read/delete.
- Default max upload size is `LOCAL_DATASETS_MAX_UPLOAD_SIZE_BYTES=200000000` (200 MB).
