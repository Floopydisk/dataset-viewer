# SPDX-License-Identifier: Apache-2.0

from io import BytesIO

from starlette.testclient import TestClient

from api.app import create_app_with_config
from api.config import AppConfig, EndpointConfig


def test_upload_list_info_rows_and_delete_local_dataset(client: TestClient) -> None:
    content = b"text,label\nhello,1\nworld,0\n"
    response = client.post(
        "/local-datasets/upload",
        files={"file": ("tiny.csv", BytesIO(content), "text/csv")},
        data={"name": "tiny-train"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    dataset_id = payload["id"]
    assert payload["name"] == "tiny-train"
    assert payload["size_bytes"] > 0
    assert payload["num_rows"] == 2
    assert payload["num_columns"] == 2

    list_response = client.get("/local-datasets")
    assert list_response.status_code == 200, list_response.text
    datasets = list_response.json()["datasets"]
    assert any(dataset["id"] == dataset_id for dataset in datasets)
    assert any(dataset["id"] == dataset_id and dataset["size_bytes"] > 0 for dataset in datasets)

    info_response = client.get(f"/local-datasets/{dataset_id}/info")
    assert info_response.status_code == 200, info_response.text
    info = info_response.json()
    assert info["id"] == dataset_id
    assert [column["name"] for column in info["columns"]] == ["text", "label"]

    rows_response = client.get(f"/local-datasets/{dataset_id}/rows", params={"offset": 0, "length": 2})
    assert rows_response.status_code == 200, rows_response.text
    rows_payload = rows_response.json()
    assert rows_payload["num_rows_total"] == 2
    assert len(rows_payload["rows"]) == 2
    assert rows_payload["rows"][0]["row"]["text"] == "hello"

    search_response = client.get(
        f"/local-datasets/{dataset_id}/search",
        params={"query": "hell", "offset": 0, "length": 10},
    )
    assert search_response.status_code == 200, search_response.text
    search_payload = search_response.json()
    assert search_payload["num_rows_total"] == 1
    assert search_payload["rows"][0]["row"]["text"] == "hello"

    filter_response = client.get(
        f"/local-datasets/{dataset_id}/filter",
        params={"where": '"label" = 1', "orderby": '"text" asc', "offset": 0, "length": 10},
    )
    assert filter_response.status_code == 200, filter_response.text
    filter_payload = filter_response.json()
    assert filter_payload["num_rows_total"] == 1
    assert filter_payload["rows"][0]["row"]["text"] == "hello"

    delete_response = client.delete(f"/local-datasets/{dataset_id}")
    assert delete_response.status_code == 200, delete_response.text

    missing_info_response = client.get(f"/local-datasets/{dataset_id}/info")
    assert missing_info_response.status_code == 404


def test_upload_rejects_unsupported_format(client: TestClient) -> None:
    response = client.post(
        "/local-datasets/upload",
        files={"file": ("tiny.txt", BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_requires_file_field(client: TestClient) -> None:
    response = client.post("/local-datasets/upload", data={"name": "missing-file"})
    assert response.status_code == 422


def test_local_filter_rejects_invalid_query_parameter(client: TestClient) -> None:
    content = b"text,label\nhello,1\n"
    upload_response = client.post(
        "/local-datasets/upload",
        files={"file": ("tiny.csv", BytesIO(content), "text/csv")},
    )
    assert upload_response.status_code == 200
    dataset_id = upload_response.json()["id"]

    response = client.get(
        f"/local-datasets/{dataset_id}/filter",
        params={"where": '"label" = 1; DROP TABLE data', "offset": 0, "length": 10},
    )
    assert response.status_code == 400


def test_local_datasets_require_bearer_token_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_DATASETS_REQUIRE_BEARER_TOKEN", "true")
    app_config = AppConfig.from_env()
    endpoint_config = EndpointConfig.from_env()
    local_client = TestClient(create_app_with_config(app_config=app_config, endpoint_config=endpoint_config))

    response = local_client.get("/local-datasets")
    assert response.status_code == 401


def test_local_datasets_namespaced_by_bearer_token(client: TestClient) -> None:
    content = b"text,label\nhello,1\n"
    upload_response = client.post(
        "/local-datasets/upload",
        files={"file": ("tiny.csv", BytesIO(content), "text/csv")},
        headers={"Authorization": "Bearer token_a"},
    )
    assert upload_response.status_code == 200, upload_response.text

    list_as_a = client.get("/local-datasets", headers={"Authorization": "Bearer token_a"})
    assert list_as_a.status_code == 200
    assert len(list_as_a.json()["datasets"]) >= 1

    list_as_b = client.get("/local-datasets", headers={"Authorization": "Bearer token_b"})
    assert list_as_b.status_code == 200
    assert list_as_b.json()["datasets"] == []
