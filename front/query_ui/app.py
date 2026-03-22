import json
import re
import urllib.parse
from typing import Any

import gradio as gr
import requests

DEFAULT_SERVER = "https://54.198.247.234"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"


def _extract_canonical_dataset_id(final_url: str) -> str | None:
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


def query_splits(
    server: str,
    dataset: str,
    hf_endpoint: str,
    token: str,
    timeout_seconds: float,
    poll_seconds: int,
    auto_resolve: bool,
    insecure_tls: bool,
) -> tuple[str, str, str]:
    verify_tls = not insecure_tls
    headers = {}
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"

    original_dataset = dataset.strip()
    canonical_dataset = original_dataset
    resolve_msg = "Auto-resolve disabled."

    if auto_resolve:
        _, canonical_dataset, resolve_msg = resolve_dataset_id(
            dataset=original_dataset,
            hf_endpoint=hf_endpoint,
            timeout_seconds=timeout_seconds,
        )

    endpoint = f"{server.rstrip('/')}/splits"
    params = {"dataset": canonical_dataset}

    status_lines: list[str] = [
        f"Server: {server}",
        f"Original dataset: {original_dataset}",
        f"Dataset used for query: {canonical_dataset}",
        resolve_msg,
    ]

    response = None
    deadline = poll_seconds
    elapsed = 0
    while True:
        response = requests.get(
            endpoint,
            params=params,
            headers=headers,
            timeout=timeout_seconds,
            verify=verify_tls,
        )

        status_lines.append(f"HTTP {response.status_code} (x-error-code={response.headers.get('x-error-code', 'n/a')})")

        error_code = response.headers.get("x-error-code", "")
        if response.status_code == 200:
            break
        if error_code != "ResponseNotReady" or elapsed >= deadline:
            break

        elapsed += 2

    assert response is not None

    try:
        payload: Any = response.json()
        pretty_json = json.dumps(payload, indent=2, ensure_ascii=True)
    except Exception:
        pretty_json = response.text

    curl_cmd = (
        f"curl -i {'-k ' if insecure_tls else ''}\"{server.rstrip('/')}/splits?dataset={urllib.parse.quote(canonical_dataset, safe='/')}\""
    )

    status_block = "\n".join(status_lines)
    status_markdown = f"### Query Status\n\n```text\n{status_block}\n```\n\n### Equivalent curl\n\n```bash\n{curl_cmd}\n```"

    return status_markdown, pretty_json, canonical_dataset


with gr.Blocks(title="Dataset Viewer Query UI") as demo:
    gr.Markdown("## Dataset Viewer Query UI")
    gr.Markdown("Resolve renamed datasets and query `/splits` from your deployment.")

    with gr.Row():
        server = gr.Textbox(label="Dataset Viewer Server", value=DEFAULT_SERVER)
        hf_endpoint = gr.Textbox(label="HF Endpoint", value=DEFAULT_HF_ENDPOINT)

    with gr.Row():
        dataset = gr.Textbox(label="Dataset", value="ag_news", placeholder="e.g. ag_news or username/dataset")
        token = gr.Textbox(label="HF Token (optional)", type="password", value="")

    with gr.Row():
        timeout_seconds = gr.Slider(label="Request timeout (seconds)", minimum=1, maximum=60, value=15, step=1)
        poll_seconds = gr.Slider(label="Poll window for ResponseNotReady (seconds)", minimum=0, maximum=180, value=30, step=2)

    with gr.Row():
        auto_resolve = gr.Checkbox(label="Auto-resolve renamed dataset IDs", value=True)
        insecure_tls = gr.Checkbox(label="Skip TLS verification (for self-signed certs)", value=True)

    run_button = gr.Button("Resolve + Query /splits", variant="primary")

    status_md = gr.Markdown()
    resolved_dataset = gr.Textbox(label="Resolved dataset", interactive=False)
    response_json = gr.Code(label="Response", language="json")

    run_button.click(
        fn=query_splits,
        inputs=[server, dataset, hf_endpoint, token, timeout_seconds, poll_seconds, auto_resolve, insecure_tls],
        outputs=[status_md, response_json, resolved_dataset],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
