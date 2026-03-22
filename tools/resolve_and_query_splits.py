#!/usr/bin/env python3
"""Resolve renamed HF dataset IDs and query /splits from a datasets-server endpoint.

Example:
  python tools/resolve_and_query_splits.py --server https://54.198.247.234 --insecure ag_news
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


@dataclass
class HttpResult:
    status: int
    headers: dict[str, str]
    body: bytes
    url: str


def _make_ssl_context(insecure: bool) -> ssl.SSLContext:
    if not insecure:
        return ssl.create_default_context()
    return ssl._create_unverified_context()  # noqa: SLF001


def http_get(url: str, insecure: bool, headers: dict[str, str] | None = None) -> HttpResult:
    req = Request(url=url, headers=headers or {}, method="GET")
    try:
        with urlopen(req, context=_make_ssl_context(insecure), timeout=20) as resp:
            return HttpResult(
                status=resp.status,
                headers={k.lower(): v for k, v in resp.headers.items()},
                body=resp.read(),
                url=resp.geturl(),
            )
    except HTTPError as err:
        return HttpResult(
            status=err.code,
            headers={k.lower(): v for k, v in err.headers.items()} if err.headers else {},
            body=err.read() if hasattr(err, "read") else b"",
            url=url,
        )


def resolve_dataset_id(dataset: str, insecure: bool) -> tuple[str, str]:
    encoded = quote(dataset, safe="/")
    hub_url = f"https://huggingface.co/api/datasets/{encoded}"
    result = http_get(hub_url, insecure=insecure)

    # urlopen follows redirects, so final URL usually contains the canonical id.
    parsed = urlparse(result.url)
    marker = "/api/datasets/"
    if marker in parsed.path:
        canonical = parsed.path.split(marker, 1)[1].strip("/")
        if canonical:
            return dataset, canonical

    # Fallback: try JSON response id
    try:
        payload = json.loads(result.body.decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("id"), str):
            return dataset, payload["id"]
    except Exception:
        pass

    return dataset, dataset


def query_splits(server: str, dataset: str, insecure: bool, token: str | None = None) -> HttpResult:
    qs = urlencode({"dataset": dataset})
    url = f"{server.rstrip('/')}/splits?{qs}"
    headers = {"user-agent": "dataset-viewer-resolver/1.0"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    return http_get(url, insecure=insecure, headers=headers)


def parse_json_or_text(body: bytes) -> Any:
    text = body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except Exception:
        return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve renamed dataset ids and query /splits.")
    parser.add_argument("dataset", help="Dataset id to query, e.g. ag_news")
    parser.add_argument(
        "--server",
        default="https://datasets-server.huggingface.co",
        help="Datasets-server base URL, e.g. https://54.198.247.234",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS certificate verification (useful for self-signed certs).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=0,
        help="If > 0, poll while x-error-code is ResponseNotReady.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token for private/gated datasets (Bearer token).",
    )
    args = parser.parse_args()

    original, canonical = resolve_dataset_id(args.dataset, insecure=args.insecure)

    if canonical != original:
        print(f"dataset renamed: {original} -> {canonical}")
    else:
        print(f"dataset id unchanged: {canonical}")

    # We keep resolver unauthenticated because hub id resolution is public.
    result = query_splits(args.server, canonical, insecure=args.insecure, token=args.token)

    if args.poll_seconds > 0:
        deadline = time.time() + args.poll_seconds
        while result.status in (500, 503) and result.headers.get("x-error-code") == "ResponseNotReady" and time.time() < deadline:
            time.sleep(2)
            result = query_splits(args.server, canonical, insecure=args.insecure, token=args.token)

    print(f"status: {result.status}")
    error_code = result.headers.get("x-error-code")
    if error_code:
        print(f"x-error-code: {error_code}")

    payload = parse_json_or_text(result.body)
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(payload)

    return 0 if result.status == 200 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except URLError as err:
        print(f"network error: {err}", file=sys.stderr)
        raise SystemExit(2)
