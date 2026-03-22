import json
import inspect
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import requests

DEFAULT_SERVER = os.environ.get("DEFAULT_SERVER", "https://54.198.247.234")
DEFAULT_HF_ENDPOINT = os.environ.get("DEFAULT_HF_ENDPOINT", "https://huggingface.co")
PRESETS_FILE = Path(os.environ.get("QUERY_UI_PRESETS_FILE", str(Path(__file__).with_name("presets.json"))))

# fallback metadata used when /openapi.json cannot be fetched from the target server
DEFAULT_ENDPOINT_METADATA: dict[str, dict[str, Any]] = {
    "/is-valid": {
        "summary": "Check if dataset capabilities are available.",
        "required": ["dataset"],
        "optional": ["config", "split"],
    },
    "/splits": {
        "summary": "List configs and splits.",
        "required": ["dataset"],
        "optional": [],
    },
    "/info": {
        "summary": "Get dataset metadata for a config.",
        "required": ["dataset", "config"],
        "optional": [],
    },
    "/first-rows": {
        "summary": "Preview first rows for a split.",
        "required": ["dataset", "config", "split"],
        "optional": [],
    },
    "/rows": {
        "summary": "Get a row slice.",
        "required": ["dataset", "config", "split"],
        "optional": ["offset", "length"],
    },
    "/search": {
        "summary": "Full-text search over string columns.",
        "required": ["dataset", "config", "split", "query"],
        "optional": ["offset", "length"],
    },
    "/filter": {
        "summary": "SQL-like filtering over split rows.",
        "required": ["dataset", "config", "split"],
        "optional": ["where", "orderby", "offset", "length"],
    },
    "/parquet": {
        "summary": "List converted parquet files.",
        "required": ["dataset"],
        "optional": [],
    },
    "/size": {
        "summary": "Get row and size metrics.",
        "required": ["dataset"],
        "optional": [],
    },
    "/statistics": {
        "summary": "Get descriptive statistics for a split.",
        "required": ["dataset", "config", "split"],
        "optional": [],
    },
    "/croissant-crumbs": {
        "summary": "Get croissant metadata crumbs.",
        "required": ["dataset"],
        "optional": ["full"],
    },
    "/opt-in-out-urls": {
        "summary": "Get opt-in/opt-out URL counts.",
        "required": ["dataset"],
        "optional": ["config", "split"],
    },
    "/presidio-entities": {
        "summary": "Get PII entity counts from sampled rows.",
        "required": ["dataset"],
        "optional": [],
    },
}


def _read_presets() -> dict[str, dict[str, Any]]:
    if not PRESETS_FILE.exists():
        return {}
    try:
        content = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        if isinstance(content, dict):
            return {name: value for name, value in content.items() if isinstance(name, str) and isinstance(value, dict)}
    except Exception:
        pass
    return {}


def _write_presets(presets: dict[str, dict[str, Any]]) -> None:
    PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PRESETS_FILE.write_text(json.dumps(presets, indent=2, ensure_ascii=True), encoding="utf-8")


def _extract_canonical_dataset_id(final_url: str) -> Optional[str]:
    match = re.search(r"/api/datasets/([^?#]+)", final_url)
    if not match:
        return None
    return urllib.parse.unquote(match.group(1)).strip("/")


def resolve_dataset_id(dataset: str, hf_endpoint: str, timeout_seconds: float) -> tuple[str, str, str]:
    dataset = dataset.strip()
    encoded = urllib.parse.quote(dataset, safe="/")
    url = f"{hf_endpoint.rstrip('/')}/api/datasets/{encoded}"
    try:
        response = requests.get(url, timeout=timeout_seconds, allow_redirects=True)
        response.raise_for_status()
        canonical = _extract_canonical_dataset_id(response.url) or dataset
        if canonical != dataset:
            return dataset, canonical, f"Renamed dataset detected: {dataset} -> {canonical}"
        return dataset, canonical, "Dataset ID is already canonical."
    except Exception as err:
        return dataset, dataset, f"Could not resolve dataset ID from Hub API: {err}"


def _headers(token: str) -> dict[str, str]:
    stripped = token.strip()
    if not stripped:
        return {}
    return {"Authorization": f"Bearer {stripped}"}


def _to_pretty_response(response: requests.Response) -> str:
    try:
        return json.dumps(response.json(), indent=2, ensure_ascii=True)
    except Exception:
        return response.text


def _build_curl(url: str, params: dict[str, str], insecure_tls: bool, token: str) -> str:
    query = urllib.parse.urlencode(params, doseq=True)
    token_arg = " -H \"Authorization: Bearer ${HF_TOKEN}\"" if token.strip() else ""
    return f"curl -i {'-k ' if insecure_tls else ''}\"{url}?{query}\"{token_arg}"


def _normalize_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return "" if value is None else str(value)
    return json.dumps(value, ensure_ascii=True)


def _extract_table(endpoint: str, payload: Any) -> tuple[list[str], list[list[Any]], str]:
    if not isinstance(payload, dict):
        return [], [], "Response is not a JSON object."

    if isinstance(payload.get("rows"), list):
        rows = payload["rows"]
        all_columns: list[str] = ["row_idx"]
        for item in rows:
            if isinstance(item, dict) and isinstance(item.get("row"), dict):
                for key in item["row"].keys():
                    if key not in all_columns:
                        all_columns.append(key)

        table_data: list[list[Any]] = []
        for item in rows:
            row_obj = item.get("row", {}) if isinstance(item, dict) else {}
            row_data: list[Any] = []
            for col in all_columns:
                if col == "row_idx":
                    row_data.append(item.get("row_idx") if isinstance(item, dict) else "")
                else:
                    row_data.append(_normalize_value(row_obj.get(col) if isinstance(row_obj, dict) else None))
            table_data.append(row_data)
        return all_columns, table_data, f"Rendered {len(table_data)} rows from {endpoint}."

    if isinstance(payload.get("splits"), list):
        columns = ["dataset", "config", "split"]
        rows = []
        for item in payload["splits"]:
            if isinstance(item, dict):
                rows.append([item.get("dataset", ""), item.get("config", ""), item.get("split", "")])
        return columns, rows, f"Rendered {len(rows)} split records."

    if isinstance(payload.get("parquet_files"), list):
        columns = ["dataset", "config", "split", "filename", "size", "url"]
        rows = []
        for item in payload["parquet_files"]:
            if isinstance(item, dict):
                rows.append(
                    [
                        item.get("dataset", ""),
                        item.get("config", ""),
                        item.get("split", ""),
                        item.get("filename", ""),
                        item.get("size", ""),
                        item.get("url", ""),
                    ]
                )
        return columns, rows, f"Rendered {len(rows)} parquet file records."

    if isinstance(payload.get("statistics"), list):
        columns = ["column_name", "column_type", "nan_count", "n_unique", "min", "max", "mean"]
        rows = []
        for item in payload["statistics"]:
            if isinstance(item, dict):
                stats = item.get("column_statistics", {}) if isinstance(item.get("column_statistics"), dict) else {}
                rows.append(
                    [
                        item.get("column_name", ""),
                        item.get("column_type", ""),
                        stats.get("nan_count", ""),
                        stats.get("n_unique", ""),
                        stats.get("min", ""),
                        stats.get("max", ""),
                        stats.get("mean", ""),
                    ]
                )
        return columns, rows, f"Rendered {len(rows)} statistics rows."

    if isinstance(payload.get("size"), dict):
        size = payload["size"]
        columns = ["scope", "dataset", "config", "split", "num_rows", "num_columns", "num_bytes_parquet_files"]
        rows = []
        if isinstance(size.get("dataset"), dict):
            ds = size["dataset"]
            rows.append(["dataset", ds.get("dataset", ""), "", "", ds.get("num_rows", ""), "", ds.get("num_bytes_parquet_files", "")])
        if isinstance(size.get("configs"), list):
            for cfg in size["configs"]:
                if isinstance(cfg, dict):
                    rows.append([
                        "config",
                        cfg.get("dataset", ""),
                        cfg.get("config", ""),
                        "",
                        cfg.get("num_rows", ""),
                        cfg.get("num_columns", ""),
                        cfg.get("num_bytes_parquet_files", ""),
                    ])
        if isinstance(size.get("splits"), list):
            for sp in size["splits"]:
                if isinstance(sp, dict):
                    rows.append([
                        "split",
                        sp.get("dataset", ""),
                        sp.get("config", ""),
                        sp.get("split", ""),
                        sp.get("num_rows", ""),
                        sp.get("num_columns", ""),
                        sp.get("num_bytes_parquet_files", ""),
                    ])
        return columns, rows, f"Rendered {len(rows)} size records."

    if endpoint == "/is-valid":
        columns = ["preview", "viewer", "search", "filter", "statistics"]
        row = [payload.get("preview", ""), payload.get("viewer", ""), payload.get("search", ""), payload.get("filter", ""), payload.get("statistics", "")]
        return columns, [row], "Rendered capability flags from /is-valid."

    return [], [], "No tabular projection for this response."


def _extract_openapi_metadata(server: str, timeout_seconds: float, insecure_tls: bool) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    url = f"{server.rstrip('/')}/openapi.json"
    response = requests.get(url, timeout=timeout_seconds, verify=not insecure_tls)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return DEFAULT_ENDPOINT_METADATA

    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        return DEFAULT_ENDPOINT_METADATA

    for path, defn in paths.items():
        if not isinstance(path, str) or not path.startswith("/"):
            continue
        get_def = defn.get("get") if isinstance(defn, dict) else None
        if not isinstance(get_def, dict):
            continue

        required_params: list[str] = []
        optional_params: list[str] = []
        for param in get_def.get("parameters", []):
            if not isinstance(param, dict):
                continue
            name = param.get("name")
            if not isinstance(name, str):
                continue
            if param.get("required") is True:
                required_params.append(name)
            else:
                optional_params.append(name)

        metadata[path] = {
            "summary": get_def.get("summary", ""),
            "required": required_params,
            "optional": optional_params,
        }

    for endpoint, fallback in DEFAULT_ENDPOINT_METADATA.items():
        metadata.setdefault(endpoint, fallback)
    return metadata


def refresh_endpoint_metadata(
    server: str,
    timeout_seconds: float,
    insecure_tls: bool,
    current_endpoint: str,
) -> tuple[dict[str, dict[str, Any]], Any, str]:
    try:
        metadata = _extract_openapi_metadata(
            server=server,
            timeout_seconds=timeout_seconds,
            insecure_tls=insecure_tls,
        )
        choices = sorted(metadata.keys())
        value = current_endpoint if current_endpoint in metadata else (choices[0] if choices else current_endpoint)
        return metadata, gr.update(choices=choices, value=value), "Loaded endpoint metadata from /openapi.json."
    except Exception as err:
        choices = sorted(DEFAULT_ENDPOINT_METADATA.keys())
        value = current_endpoint if current_endpoint in DEFAULT_ENDPOINT_METADATA else (choices[0] if choices else current_endpoint)
        return DEFAULT_ENDPOINT_METADATA, gr.update(choices=choices, value=value), f"Using fallback endpoint metadata: {err}"


def endpoint_help(endpoint: str, metadata: dict[str, dict[str, Any]]) -> str:
    details = metadata.get(endpoint) or DEFAULT_ENDPOINT_METADATA.get(endpoint) or {"summary": "", "required": [], "optional": []}
    required = ", ".join(details.get("required", [])) or "none"
    optional = ", ".join(details.get("optional", [])) or "none"
    summary = details.get("summary", "")
    return (
        "### Endpoint Metadata\n\n"
        f"- Endpoint: `{endpoint}`\n"
        f"- Summary: {summary or 'n/a'}\n"
        f"- Required params: {required}\n"
        f"- Optional params: {optional}"
    )


def _request_with_poll(
    *,
    server: str,
    endpoint: str,
    params: dict[str, str],
    token: str,
    timeout_seconds: float,
    poll_seconds: int,
    insecure_tls: bool,
) -> tuple[requests.Response, list[str], str]:
    url = f"{server.rstrip('/')}{endpoint}"
    status_lines: list[str] = [f"Request URL: {url}"]
    started = time.monotonic()
    response: Optional[requests.Response] = None

    while True:
        response = requests.get(
            url,
            params=params,
            headers=_headers(token),
            timeout=timeout_seconds,
            verify=not insecure_tls,
        )
        error_code = response.headers.get("x-error-code", "n/a")
        status_lines.append(f"HTTP {response.status_code} (x-error-code={error_code})")

        if response.status_code == 200:
            break
        if error_code != "ResponseNotReady":
            break
        if (time.monotonic() - started) >= poll_seconds:
            status_lines.append("Stopped polling: poll window elapsed.")
            break
        time.sleep(2)

    assert response is not None
    return response, status_lines, url


def _status_markdown(lines: list[str], resolve_message: str, original_dataset: str, used_dataset: str) -> str:
    full_lines = [
        f"Original dataset: {original_dataset}",
        f"Dataset used: {used_dataset}",
        resolve_message,
        *lines,
    ]
    return "### Query Status\n\n```text\n" + "\n".join(full_lines) + "\n```"


def query_endpoint(
    server: str,
    hf_endpoint: str,
    token: str,
    timeout_seconds: float,
    poll_seconds: int,
    auto_resolve: bool,
    insecure_tls: bool,
    endpoint: str,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    query: str,
    where: str,
    orderby: str,
    full: bool,
    metadata: dict[str, dict[str, Any]],
) -> tuple[str, str, str, str, str, Any]:
    original_dataset = dataset.strip()
    canonical_dataset = original_dataset
    resolve_msg = "Auto-resolve disabled."

    if auto_resolve and original_dataset:
        _, canonical_dataset, resolve_msg = resolve_dataset_id(
            dataset=original_dataset,
            hf_endpoint=hf_endpoint,
            timeout_seconds=timeout_seconds,
        )

    params_source = {
        "dataset": canonical_dataset,
        "config": config.strip(),
        "split": split.strip(),
        "offset": str(int(offset)),
        "length": str(int(length)),
        "query": query.strip(),
        "where": where.strip(),
        "orderby": orderby.strip(),
        "full": "true" if full else "false",
    }
    details = metadata.get(endpoint) or DEFAULT_ENDPOINT_METADATA.get(endpoint) or {"required": [], "optional": []}
    expected_params = set(details.get("required", []) + details.get("optional", []))
    params = {name: value for name, value in params_source.items() if name in expected_params and value != ""}

    for required_name in details.get("required", []):
        if not params.get(required_name):
            status_md = _status_markdown(
                [f"Missing required parameter '{required_name}' for {endpoint}."],
                resolve_msg,
                original_dataset,
                canonical_dataset,
            )
            return status_md, "", canonical_dataset, "", "No table rendered due to missing required parameters.", gr.update(visible=False)

    try:
        response, status_lines, url = _request_with_poll(
            server=server,
            endpoint=endpoint,
            params=params,
            token=token,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            insecure_tls=insecure_tls,
        )
    except Exception as err:
        status_md = _status_markdown(
            [f"Request failed: {err}"],
            resolve_msg,
            original_dataset,
            canonical_dataset,
        )
        return status_md, "", canonical_dataset, "", "No table rendered because the request failed.", gr.update(visible=False)

    status_md = _status_markdown(status_lines, resolve_msg, original_dataset, canonical_dataset)
    response_text = _to_pretty_response(response)
    curl_cmd = _build_curl(url=url, params=params, insecure_tls=insecure_tls, token=token)
    table_headers: list[str] = []
    table_rows: list[list[Any]] = []
    table_message = "No table rendered."
    try:
        payload = response.json()
        table_headers, table_rows, table_message = _extract_table(endpoint=endpoint, payload=payload)
    except Exception:
        table_message = "No table rendered: response was not JSON."

    table_update = gr.update(headers=table_headers, value=table_rows, visible=bool(table_headers))
    return status_md, response_text, canonical_dataset, curl_cmd, table_message, table_update


def inspect_dataset_formats(
    server: str,
    hf_endpoint: str,
    token: str,
    timeout_seconds: float,
    insecure_tls: bool,
    dataset: str,
) -> tuple[str, str, Any]:
    dataset = dataset.strip()
    if not dataset:
        return "Dataset is required.", "", gr.update(visible=False)

    headers = _headers(token)
    encoded = urllib.parse.quote(dataset, safe="/")
    dataset_info_url = f"{hf_endpoint.rstrip('/')}/api/datasets/{encoded}"
    viewer_valid_url = f"{server.rstrip('/')}/is-valid"

    summary_lines = [f"Dataset: {dataset}"]
    details: dict[str, Any] = {}

    try:
        info_response = requests.get(dataset_info_url, headers=headers, timeout=timeout_seconds, verify=not insecure_tls)
        summary_lines.append(f"Hub /api/datasets: HTTP {info_response.status_code}")
        info_json = info_response.json() if info_response.headers.get("content-type", "").startswith("application/json") else {}
        details["hub_dataset_info"] = info_json
        siblings = info_json.get("siblings", []) if isinstance(info_json, dict) else []
    except Exception as err:
        summary_lines.append(f"Hub metadata request failed: {err}")
        details["hub_dataset_info_error"] = str(err)
        siblings = []

    extension_counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}

    for sibling in siblings:
        if not isinstance(sibling, dict):
            continue
        path = sibling.get("rfilename") or sibling.get("path")
        if not isinstance(path, str):
            continue
        lower = path.lower()
        ext = "other"
        if lower.endswith(".parquet"):
            ext = "parquet"
        elif lower.endswith(".csv") or lower.endswith(".tsv"):
            ext = "csv_tsv"
        elif lower.endswith(".json") or lower.endswith(".jsonl"):
            ext = "json_jsonl"
        elif lower.endswith(".arrow"):
            ext = "arrow"
        elif lower.endswith(".txt"):
            ext = "text"
        elif lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")):
            ext = "image"
        elif lower.endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
            ext = "audio"
        elif lower.endswith((".mp4", ".webm", ".mov", ".mkv")):
            ext = "video"

        extension_counts[ext] = extension_counts.get(ext, 0) + 1
        if ext not in samples:
            samples[ext] = []
        if len(samples[ext]) < 3:
            samples[ext].append(path)

    try:
        is_valid_response = requests.get(
            viewer_valid_url,
            params={"dataset": dataset},
            headers=headers,
            timeout=timeout_seconds,
            verify=not insecure_tls,
        )
        summary_lines.append(f"Dataset viewer /is-valid: HTTP {is_valid_response.status_code}")
        details["viewer_is_valid"] = is_valid_response.json()
    except Exception as err:
        summary_lines.append(f"/is-valid request failed: {err}")
        details["viewer_is_valid_error"] = str(err)

    rows = []
    for ext, count in sorted(extension_counts.items(), key=lambda item: item[0]):
        rows.append([ext, count, " | ".join(samples.get(ext, []))])

    if not rows:
        rows = [["unknown", 0, "No file samples available from Hub metadata response"]]

    summary_lines.append("Tip: dataset-viewer can auto-convert many source formats (CSV/JSON/etc.) to parquet for fast querying.")
    summary_md = "### Format Support Summary\n\n```text\n" + "\n".join(summary_lines) + "\n```"
    raw_json = json.dumps(details, indent=2, ensure_ascii=True)
    table_update = gr.update(headers=["detected_type", "count", "sample_paths"], value=rows, visible=True)
    return summary_md, raw_json, table_update


def save_preset(
    preset_name: str,
    server: str,
    hf_endpoint: str,
    endpoint: str,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    query: str,
    where: str,
    orderby: str,
    full: bool,
    auto_resolve: bool,
    insecure_tls: bool,
) -> tuple[str, Any]:
    name = preset_name.strip()
    if not name:
        return "Preset name is required.", gr.update()
    presets = _read_presets()
    presets[name] = {
        "server": server,
        "hf_endpoint": hf_endpoint,
        "endpoint": endpoint,
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": int(offset),
        "length": int(length),
        "query": query,
        "where": where,
        "orderby": orderby,
        "full": bool(full),
        "auto_resolve": bool(auto_resolve),
        "insecure_tls": bool(insecure_tls),
    }
    _write_presets(presets)
    names = sorted(presets.keys())
    return f"Saved preset '{name}'.", gr.update(choices=names, value=name)


def delete_preset(name: str) -> tuple[str, Any]:
    if not name:
        return "Select a preset to delete.", gr.update()
    presets = _read_presets()
    if name not in presets:
        return "Preset does not exist.", gr.update(choices=sorted(presets.keys()))
    del presets[name]
    _write_presets(presets)
    names = sorted(presets.keys())
    return f"Deleted preset '{name}'.", gr.update(choices=names, value=(names[0] if names else None))


def load_preset(name: str) -> tuple[str, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    presets = _read_presets()
    preset = presets.get(name)
    if not preset:
        return (
            "Preset not found.",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    return (
        f"Loaded preset '{name}'.",
        gr.update(value=preset.get("server", DEFAULT_SERVER)),
        gr.update(value=preset.get("hf_endpoint", DEFAULT_HF_ENDPOINT)),
        gr.update(value=preset.get("endpoint", "/splits")),
        gr.update(value=preset.get("dataset", "")),
        gr.update(value=preset.get("config", "")),
        gr.update(value=preset.get("split", "")),
        gr.update(value=preset.get("offset", 0)),
        gr.update(value=preset.get("length", 10)),
        gr.update(value=preset.get("query", "")),
        gr.update(value=preset.get("where", "")),
        gr.update(value=preset.get("orderby", "")),
        gr.update(value=preset.get("full", True)),
        gr.update(value=preset.get("auto_resolve", True)),
        gr.update(value=preset.get("insecure_tls", True)),
    )


def dataset_profile(
    server: str,
    hf_endpoint: str,
    token: str,
    timeout_seconds: float,
    poll_seconds: int,
    auto_resolve: bool,
    insecure_tls: bool,
    dataset: str,
    config: str,
    split: str,
) -> tuple[str, str, str]:
    original_dataset = dataset.strip()
    canonical_dataset = original_dataset
    resolve_msg = "Auto-resolve disabled."
    if auto_resolve and original_dataset:
        _, canonical_dataset, resolve_msg = resolve_dataset_id(
            dataset=original_dataset,
            hf_endpoint=hf_endpoint,
            timeout_seconds=timeout_seconds,
        )

    checks: list[tuple[str, dict[str, str]]] = [
        ("/is-valid", {"dataset": canonical_dataset}),
        ("/splits", {"dataset": canonical_dataset}),
        ("/parquet", {"dataset": canonical_dataset}),
        ("/size", {"dataset": canonical_dataset}),
    ]
    if config.strip():
        checks.append(("/info", {"dataset": canonical_dataset, "config": config.strip()}))
    if config.strip() and split.strip():
        checks.append(
            (
                "/statistics",
                {"dataset": canonical_dataset, "config": config.strip(), "split": split.strip()},
            )
        )

    summary_lines = [
        f"Dataset profile for: {canonical_dataset}",
        f"Resolve status: {resolve_msg}",
    ]
    payload: dict[str, Any] = {}

    for endpoint, params in checks:
        try:
            response, _, _ = _request_with_poll(
                server=server,
                endpoint=endpoint,
                params=params,
                token=token,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
                insecure_tls=insecure_tls,
            )
            summary_lines.append(
                f"{endpoint}: HTTP {response.status_code} (x-error-code={response.headers.get('x-error-code', 'n/a')})"
            )
            try:
                payload[endpoint] = response.json()
            except Exception:
                payload[endpoint] = response.text
        except Exception as err:
            summary_lines.append(f"{endpoint}: request failed ({err})")
            payload[endpoint] = {"error": str(err)}

    return (
        "### Profile Summary\n\n```text\n" + "\n".join(summary_lines) + "\n```",
        json.dumps(payload, indent=2, ensure_ascii=True),
        canonical_dataset,
    )


with gr.Blocks(title="Dataset Viewer Workbench") as demo:
    gr.Markdown("## Dataset Viewer Workbench")
    gr.Markdown("A complete frontend for exploring your dataset-viewer deployment from one place.")

    with gr.Row():
        server = gr.Textbox(label="Dataset Viewer Server", value=DEFAULT_SERVER)
        hf_endpoint = gr.Textbox(label="HF Endpoint", value=DEFAULT_HF_ENDPOINT)

    with gr.Row():
        dataset = gr.Textbox(label="Dataset", value="ag_news", placeholder="e.g. ag_news or username/dataset")
        token = gr.Textbox(label="HF Token (optional)", type="password", value="")

    with gr.Row():
        config = gr.Textbox(label="Config (subset)", value="")
        split = gr.Textbox(label="Split", value="")

    with gr.Row():
        timeout_seconds = gr.Slider(label="Request timeout (seconds)", minimum=1, maximum=120, value=15, step=1)
        poll_seconds = gr.Slider(label="Poll window for ResponseNotReady (seconds)", minimum=0, maximum=240, value=30, step=2)

    with gr.Row():
        auto_resolve = gr.Checkbox(label="Auto-resolve renamed dataset IDs", value=True)
        insecure_tls = gr.Checkbox(label="Skip TLS verification (for self-signed certs)", value=True)

    endpoint_metadata_state = gr.State(DEFAULT_ENDPOINT_METADATA)

    with gr.Tab("Dataset Profile"):
        gr.Markdown("Run a quick multi-endpoint check to understand what is available for a dataset.")
        profile_button = gr.Button("Run Dataset Profile", variant="primary")
        profile_status = gr.Markdown()
        profile_resolved_dataset = gr.Textbox(label="Resolved dataset", interactive=False)
        profile_payload = gr.Code(label="Profile payload", language="json")

    with gr.Tab("Endpoint Explorer"):
        with gr.Row():
            refresh_metadata_button = gr.Button("Refresh Endpoints From OpenAPI")
            endpoint_metadata_status = gr.Markdown()

        with gr.Row():
            endpoint = gr.Dropdown(
                label="Endpoint",
                choices=sorted(DEFAULT_ENDPOINT_METADATA.keys()),
                value="/splits",
            )
            full = gr.Checkbox(label="full (only for /croissant-crumbs)", value=True)
        endpoint_info = gr.Markdown()

        with gr.Row():
            offset = gr.Number(label="offset", value=0, precision=0)
            length = gr.Number(label="length", value=10, precision=0)

        with gr.Row():
            query = gr.Textbox(label="query (for /search)", value="")
            orderby = gr.Textbox(label="orderby (for /filter)", value="")

        where = gr.Textbox(
            label='where (for /filter, e.g. "label"=1)',
            value="",
        )

        run_button = gr.Button("Run Endpoint Query", variant="primary")
        status_md = gr.Markdown()
        resolved_dataset = gr.Textbox(label="Resolved dataset", interactive=False)
        curl_cmd = gr.Code(label="Equivalent curl", language="shell")
        response_json = gr.Code(label="Response", language="json")
        table_status = gr.Markdown()
        result_table = gr.Dataframe(label="Tabular View", visible=False, interactive=False)

    with gr.Tab("Format Inspector"):
        gr.Markdown("Inspect source file types (CSV/JSON/Parquet/etc.) and viewer support for a dataset.")
        inspect_button = gr.Button("Inspect Dataset Formats", variant="primary")
        inspect_status = gr.Markdown()
        inspect_raw = gr.Code(label="Inspector raw payload", language="json")
        inspect_table = gr.Dataframe(label="Detected source types", interactive=False)

    with gr.Tab("Presets"):
        preset_names = sorted(_read_presets().keys())
        with gr.Row():
            preset_name = gr.Textbox(label="Preset name", placeholder="e.g. common_voice_search")
            preset_selector = gr.Dropdown(label="Saved presets", choices=preset_names, value=(preset_names[0] if preset_names else None))
        with gr.Row():
            save_preset_button = gr.Button("Save Current Settings", variant="primary")
            load_preset_button = gr.Button("Load Selected Preset")
            delete_preset_button = gr.Button("Delete Selected Preset")
        preset_status = gr.Markdown()

    profile_button.click(
        fn=dataset_profile,
        inputs=[
            server,
            hf_endpoint,
            token,
            timeout_seconds,
            poll_seconds,
            auto_resolve,
            insecure_tls,
            dataset,
            config,
            split,
        ],
        outputs=[profile_status, profile_payload, profile_resolved_dataset],
    )

    refresh_metadata_button.click(
        fn=refresh_endpoint_metadata,
        inputs=[server, timeout_seconds, insecure_tls, endpoint],
        outputs=[endpoint_metadata_state, endpoint, endpoint_metadata_status],
    )

    endpoint.change(
        fn=endpoint_help,
        inputs=[endpoint, endpoint_metadata_state],
        outputs=[endpoint_info],
    )

    demo.load(
        fn=endpoint_help,
        inputs=[endpoint, endpoint_metadata_state],
        outputs=[endpoint_info],
    )

    run_button.click(
        fn=query_endpoint,
        inputs=[
            server,
            hf_endpoint,
            token,
            timeout_seconds,
            poll_seconds,
            auto_resolve,
            insecure_tls,
            endpoint,
            dataset,
            config,
            split,
            offset,
            length,
            query,
            where,
            orderby,
            full,
            endpoint_metadata_state,
        ],
        outputs=[status_md, response_json, resolved_dataset, curl_cmd, table_status, result_table],
    )

    inspect_button.click(
        fn=inspect_dataset_formats,
        inputs=[server, hf_endpoint, token, timeout_seconds, insecure_tls, dataset],
        outputs=[inspect_status, inspect_raw, inspect_table],
    )

    save_preset_button.click(
        fn=save_preset,
        inputs=[
            preset_name,
            server,
            hf_endpoint,
            endpoint,
            dataset,
            config,
            split,
            offset,
            length,
            query,
            where,
            orderby,
            full,
            auto_resolve,
            insecure_tls,
        ],
        outputs=[preset_status, preset_selector],
    )

    delete_preset_button.click(
        fn=delete_preset,
        inputs=[preset_selector],
        outputs=[preset_status, preset_selector],
    )

    load_preset_button.click(
        fn=load_preset,
        inputs=[preset_selector],
        outputs=[
            preset_status,
            server,
            hf_endpoint,
            endpoint,
            dataset,
            config,
            split,
            offset,
            length,
            query,
            where,
            orderby,
            full,
            auto_resolve,
            insecure_tls,
        ],
    )


if __name__ == "__main__":
    launch_kwargs = {
        "server_name": os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        "server_port": int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    }
    configured_root_path = os.environ.get("GRADIO_ROOT_PATH")
    if configured_root_path:
        try:
            if "root_path" in inspect.signature(demo.launch).parameters:
                launch_kwargs["root_path"] = configured_root_path
        except Exception:
            pass
    demo.launch(**launch_kwargs)
