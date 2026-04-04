# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import json
import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

import duckdb
import fsspec
import pyarrow as pa
import pyarrow.csv as pyarrow_csv
import pyarrow.json as pyarrow_json
import pyarrow.parquet as pyarrow_parquet
from libapi.request import get_request_parameter, get_request_parameter_length, get_request_parameter_offset
from libapi.utils import Endpoint, get_json_error_response, get_json_ok_response
from libcommon.config import S3Config
from starlette.datastructures import FormData, UploadFile
from starlette.requests import Request
from starlette.responses import Response

from api.config import LocalDatasetsConfig

SupportedDatasetFormat = Literal["csv", "json", "jsonl", "parquet"]
SUPPORTED_FILE_EXTENSIONS: dict[str, SupportedDatasetFormat] = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
}

FILTER_QUERY = """\
    SELECT {columns}
    FROM data
    {where}
    {orderby}
    LIMIT {limit}
    OFFSET {offset}"""

FILTER_COUNT_QUERY = """\
    SELECT COUNT(*)
    FROM data
    {where}"""

SQL_INVALID_SYMBOLS = "|".join([";", "--", r"/\*", r"\*/"])
SQL_INVALID_SYMBOLS_PATTERN = re.compile(rf"(?:{SQL_INVALID_SYMBOLS})", flags=re.IGNORECASE)

SQL_MATCH_NUMBER = r"[0-9][0-9\.]*"
SQL_MATCH_VARCHAR = r"'([^']|'')*'"
SQL_MATCH_KEY = r'"([^"]|"")+"'
SQL_MATCH_COL = rf"{SQL_MATCH_KEY}(\.{SQL_MATCH_KEY})*"
SQL_MATCH_ARGS = f"({SQL_MATCH_NUMBER}|{SQL_MATCH_VARCHAR}|,| )*"
SQL_MATCH_FUNC = rf"[a-z_]+\({SQL_MATCH_ARGS}\)"
SQL_MATCH_TRANSFORMED_COL = rf"\(?{SQL_MATCH_COL}\)?(\.{SQL_MATCH_FUNC})?"
SQL_MATCH_OP = (
    r"(=|<|>|!|~|\*| |(like)|(ilike)|(glob)|(similar to)|(is)|(not)|(LIKE)|(ILIKE)|(GLOB)|(SIMILAR TO)|(IS)|(NOT))+"
)
SQL_MATCH_BOOL_OR_NULL = r"(true|false|null|NULL)"
SQL_MATCH_VAL = f"({SQL_MATCH_NUMBER}|{SQL_MATCH_VARCHAR}|{SQL_MATCH_BOOL_OR_NULL})"
SQL_MATCH_COND = r"(and|or|AND|OR)"
SQL_MATCH_EXPR = rf"\(?{SQL_MATCH_TRANSFORMED_COL} ?{SQL_MATCH_OP} ?{SQL_MATCH_VAL}\)?"
SQL_MATCH_DIRECTION = r"(asc|desc|ASC|DESC)"

SQL_MATCH_WHERE = f"^{SQL_MATCH_EXPR}( {SQL_MATCH_COND} {SQL_MATCH_EXPR})*$"
SQL_MATCH_ORDERBY = f"^{SQL_MATCH_TRANSFORMED_COL}( {SQL_MATCH_DIRECTION})?$"

SQL_PARAMETER_PATTERNS = {
    "where": re.compile(SQL_MATCH_WHERE),
    "orderby": re.compile(SQL_MATCH_ORDERBY),
}


@dataclass(frozen=True)
class LocalDatasetMetadata:
    id: str
    namespace: str
    name: str
    format: SupportedDatasetFormat
    original_filename: str
    file_path: str
    size_bytes: int
    num_rows: int
    num_columns: int
    columns: list[dict[str, str]]
    uploaded_at: str


@dataclass(frozen=True)
class LocalDatasetsStore:
    fs: Any
    root: str


def _create_store(config: LocalDatasetsConfig, s3_config: S3Config) -> LocalDatasetsStore:
    if config.storage_protocol == "s3":
        client_kwargs: dict[str, Any] = {"region_name": s3_config.region_name}
        if s3_config.endpoint_url:
            client_kwargs["endpoint_url"] = s3_config.endpoint_url
        config_kwargs = {"s3": {"addressing_style": s3_config.addressing_style}} if s3_config.addressing_style else None
        fs = fsspec.filesystem(
            "s3",
            key=s3_config.access_key_id,
            secret=s3_config.secret_access_key,
            client_kwargs=client_kwargs,
            config_kwargs=config_kwargs,
            max_paths=100,
        )
    else:
        fs = fsspec.filesystem("file", auto_mkdir=True)
    return LocalDatasetsStore(fs=fs, root=config.storage_root.rstrip("/"))


def _store_path(store: LocalDatasetsStore, suffix: str) -> str:
    return f"{store.root}/{suffix.lstrip('/')}"


def _metadata_path(store: LocalDatasetsStore, namespace: str, dataset_id: str) -> str:
    return _store_path(store=store, suffix=f"metadata/{namespace}/{dataset_id}.json")


def _extract_bearer_token(request: Request) -> Optional[str]:
    authorization = request.headers.get("Authorization", "").strip()
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    token = authorization[7:].strip()
    return token or None


def _get_access_error_response(request: Request, config: LocalDatasetsConfig) -> Optional[Response]:
    if not config.require_bearer_token:
        return None
    if _extract_bearer_token(request) is None:
        return get_json_error_response(
            content={"error": "Authorization header with Bearer token is required for local dataset operations."},
            status_code=HTTPStatus.UNAUTHORIZED,
            max_age=0,
        )
    return None


def _get_request_namespace(request: Request) -> str:
    token = _extract_bearer_token(request)
    if token is None:
        return "anonymous"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _slugify_filename(filename: str) -> str:
    basename = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", basename)


def _serialize_arrow_type(type_: pa.DataType) -> str:
    return str(type_)


def _detect_dataset_format(filename: str) -> Optional[SupportedDatasetFormat]:
    suffix = Path(filename).suffix.lower()
    return SUPPORTED_FILE_EXTENSIONS.get(suffix)


def _get_file_size(store: LocalDatasetsStore, path: str) -> int:
    try:
        return int(store.fs.size(path))
    except Exception:
        return -1


def _load_table_from_store(store: LocalDatasetsStore, file_path: str, dataset_format: SupportedDatasetFormat) -> pa.Table:
    if dataset_format == "csv":
        with store.fs.open(file_path, "rb") as handle:
            return pyarrow_csv.read_csv(handle)
    if dataset_format == "parquet":
        with store.fs.open(file_path, "rb") as handle:
            return pyarrow_parquet.read_table(handle)
    if dataset_format == "jsonl":
        with store.fs.open(file_path, "rb") as handle:
            return pyarrow_json.read_json(handle)

    with store.fs.open(file_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        if payload and not all(isinstance(row, dict) for row in payload):
            raise ValueError("JSON arrays must only contain objects")
        return pa.Table.from_pylist(payload)
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            rows = payload["data"]
            if rows and not all(isinstance(row, dict) for row in rows):
                raise ValueError("JSON field 'data' must contain a list of objects")
            return pa.Table.from_pylist(rows)
        return pa.Table.from_pylist([payload])
    raise ValueError("JSON files must contain an object or an array of objects")


def _read_metadata(store: LocalDatasetsStore, namespace: str, dataset_id: str) -> Optional[LocalDatasetMetadata]:
    path = _metadata_path(store=store, namespace=namespace, dataset_id=dataset_id)
    if not store.fs.exists(path):
        return None
    with store.fs.open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return LocalDatasetMetadata(**payload)


def _list_metadata(store: LocalDatasetsStore, namespace: str) -> list[LocalDatasetMetadata]:
    pattern = _store_path(store=store, suffix=f"metadata/{namespace}/*.json")
    results: list[LocalDatasetMetadata] = []
    for metadata_file in sorted(store.fs.glob(pattern)):
        with store.fs.open(metadata_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        results.append(LocalDatasetMetadata(**payload))
    return results


async def _save_upload_file(store: LocalDatasetsStore, file: UploadFile, destination_path: str, max_size: int) -> int:
    total_size = 0
    with store.fs.open(destination_path, "wb") as output:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > max_size:
                raise ValueError("Upload exceeds the maximum allowed size")
            output.write(chunk)
    await file.close()
    return total_size


def _build_rows_payload(table: pa.Table, offset: int, length: int) -> dict[str, Any]:
    page = table.slice(offset, length)
    rows = [{"row_idx": offset + index, "row": row} for index, row in enumerate(page.to_pylist())]
    return {
        "offset": offset,
        "length": length,
        "num_rows_total": table.num_rows,
        "num_rows_per_page": len(rows),
        "rows": rows,
    }


def _get_size_limit_error_response(metadata: LocalDatasetMetadata, config: LocalDatasetsConfig) -> Optional[Response]:
    if metadata.size_bytes > config.max_in_memory_processing_bytes:
        return get_json_error_response(
            content={
                "error": (
                    "Dataset is too large for in-memory processing on this endpoint. "
                    f"Size={metadata.size_bytes} bytes, limit={config.max_in_memory_processing_bytes} bytes."
                )
            },
            status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            max_age=0,
        )
    return None


def _validate_query_parameter(parameter_value: str, parameter_name: Literal["where", "orderby"]) -> None:
    if SQL_INVALID_SYMBOLS_PATTERN.search(parameter_value) or (
        parameter_value and not SQL_PARAMETER_PATTERNS[parameter_name].match(parameter_value)
    ):
        raise ValueError(f"Parameter '{parameter_name}' contains errors or invalid symbols")


def _execute_filter_query(
    table: pa.Table,
    columns: list[str],
    where: str,
    orderby: str,
    limit: int,
    offset: int,
) -> tuple[int, pa.Table]:
    con = duckdb.connect(database=":memory:")
    try:
        con.register("data", table)
        filter_query = FILTER_QUERY.format(
            columns=",".join([f'"{column}"' for column in columns]),
            where=f"WHERE {where}" if where else "",
            orderby=f"ORDER BY {orderby}" if orderby else "",
            limit=limit,
            offset=offset,
        )
        filter_count_query = FILTER_COUNT_QUERY.format(where=f"WHERE {where}" if where else "")
        page = con.sql(filter_query).arrow().read_all()
        total_rows = con.sql(filter_count_query).fetchall()[0][0]
    finally:
        con.close()
    return total_rows, page


def _execute_search_query(table: pa.Table, query: str, limit: int, offset: int) -> tuple[int, pa.Table]:
    con = duckdb.connect(database=":memory:")
    try:
        con.register("data", table)
        escaped_query = query.replace("%", "\\%").replace("_", "\\_").lower()
        columns = [row[0] for row in con.execute("DESCRIBE data").fetchall()]
        if not columns:
            return 0, table.slice(0, 0)
        predicates = [f"LOWER(CAST(\"{column}\" AS VARCHAR)) LIKE ? ESCAPE '\\\\'" for column in columns]
        where_clause = " OR ".join(predicates)
        count_sql = f"SELECT COUNT(*) FROM data WHERE {where_clause}"
        rows_sql = f"SELECT * FROM data WHERE {where_clause} LIMIT {limit} OFFSET {offset}"
        params = [f"%{escaped_query}%"] * len(columns)
        total_rows = con.execute(count_sql, params).fetchall()[0][0]
        page = con.execute(rows_sql, params).arrow().read_all()
    finally:
        con.close()
    return total_rows, page


def create_upload_local_dataset_endpoint(config: LocalDatasetsConfig, s3_config: S3Config) -> Endpoint:
    async def upload_local_dataset_endpoint(request: Request) -> Response:
        access_error = _get_access_error_response(request=request, config=config)
        if access_error is not None:
            return access_error
        namespace = _get_request_namespace(request)
        try:
            store = _create_store(config=config, s3_config=s3_config)
        except Exception as error:
            return get_json_error_response(
                content={"error": f"Failed to initialize local dataset storage: {type(error).__name__}"},
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                max_age=0,
            )

        try:
            form: FormData = await request.form()
        except Exception as error:
            return get_json_error_response(
                content={"error": f"Invalid multipart upload request: {type(error).__name__}"},
                status_code=HTTPStatus.BAD_REQUEST,
                max_age=0,
            )

        file = form.get("file")
        if not isinstance(file, UploadFile):
            return get_json_error_response(
                content={"error": "The multipart field 'file' is required."},
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                max_age=0,
            )

        dataset_format = _detect_dataset_format(file.filename or "")
        if dataset_format is None:
            return get_json_error_response(
                content={
                    "error": "Unsupported file format. Supported extensions: .csv, .json, .jsonl, .ndjson, .parquet"
                },
                status_code=HTTPStatus.BAD_REQUEST,
                max_age=0,
            )

        dataset_id = uuid4().hex
        safe_filename = _slugify_filename(file.filename or f"dataset.{dataset_format}")
        stored_filename = f"{dataset_id}_{safe_filename}"
        stored_file_path = _store_path(store=store, suffix=f"files/{namespace}/{stored_filename}")

        try:
            uploaded_size = await _save_upload_file(
                store=store,
                file=file,
                destination_path=stored_file_path,
                max_size=config.max_upload_size_bytes,
            )
            table = _load_table_from_store(store=store, file_path=stored_file_path, dataset_format=dataset_format)
        except ValueError as error:
            if store.fs.exists(stored_file_path):
                store.fs.rm(stored_file_path, recursive=False)
            status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE if "maximum allowed size" in str(error) else HTTPStatus.BAD_REQUEST
            return get_json_error_response(content={"error": str(error)}, status_code=status_code, max_age=0)
        except Exception as error:
            if store.fs.exists(stored_file_path):
                store.fs.rm(stored_file_path, recursive=False)
            return get_json_error_response(
                content={"error": f"Failed to parse uploaded file: {type(error).__name__}"},
                status_code=HTTPStatus.BAD_REQUEST,
                max_age=0,
            )

        dataset_name = str(form.get("name") or Path(file.filename or "dataset").stem)
        metadata = LocalDatasetMetadata(
            id=dataset_id,
            namespace=namespace,
            name=dataset_name,
            format=dataset_format,
            original_filename=file.filename or safe_filename,
            file_path=stored_file_path,
            size_bytes=uploaded_size,
            num_rows=table.num_rows,
            num_columns=table.num_columns,
            columns=[{"name": field.name, "type": _serialize_arrow_type(field.type)} for field in table.schema],
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )

        metadata_path = _metadata_path(store=store, namespace=namespace, dataset_id=dataset_id)
        with store.fs.open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(asdict(metadata), handle)

        return get_json_ok_response(
            {
                "id": metadata.id,
                "name": metadata.name,
                "format": metadata.format,
                "size_bytes": metadata.size_bytes,
                "num_rows": metadata.num_rows,
                "num_columns": metadata.num_columns,
                "columns": metadata.columns,
                "uploaded_at": metadata.uploaded_at,
            },
            max_age=0,
        )

    return upload_local_dataset_endpoint


def create_list_local_datasets_endpoint(config: LocalDatasetsConfig, s3_config: S3Config) -> Endpoint:
    async def list_local_datasets_endpoint(request: Request) -> Response:
        access_error = _get_access_error_response(request=request, config=config)
        if access_error is not None:
            return access_error
        namespace = _get_request_namespace(request)
        store = _create_store(config=config, s3_config=s3_config)
        datasets = _list_metadata(store=store, namespace=namespace)
        return get_json_ok_response(
            {
                "datasets": [
                    {
                        "id": dataset.id,
                        "name": dataset.name,
                        "format": dataset.format,
                        "size_bytes": dataset.size_bytes,
                        "num_rows": dataset.num_rows,
                        "num_columns": dataset.num_columns,
                        "uploaded_at": dataset.uploaded_at,
                    }
                    for dataset in datasets
                ]
            },
            max_age=0,
        )

    return list_local_datasets_endpoint


def create_local_dataset_info_endpoint(config: LocalDatasetsConfig, s3_config: S3Config) -> Endpoint:
    async def local_dataset_info_endpoint(request: Request) -> Response:
        access_error = _get_access_error_response(request=request, config=config)
        if access_error is not None:
            return access_error
        namespace = _get_request_namespace(request)
        store = _create_store(config=config, s3_config=s3_config)
        dataset_id = request.path_params["dataset_id"]
        metadata = _read_metadata(store=store, namespace=namespace, dataset_id=dataset_id)
        if metadata is None:
            return get_json_error_response(
                content={"error": "Local dataset not found."}, status_code=HTTPStatus.NOT_FOUND, max_age=0
            )

        size_error = _get_size_limit_error_response(metadata=metadata, config=config)
        if size_error is not None:
            return size_error
        return get_json_ok_response(asdict(metadata), max_age=0)

    return local_dataset_info_endpoint


def create_local_dataset_rows_endpoint(config: LocalDatasetsConfig, s3_config: S3Config) -> Endpoint:
    async def local_dataset_rows_endpoint(request: Request) -> Response:
        access_error = _get_access_error_response(request=request, config=config)
        if access_error is not None:
            return access_error
        namespace = _get_request_namespace(request)
        store = _create_store(config=config, s3_config=s3_config)
        dataset_id = request.path_params["dataset_id"]
        metadata = _read_metadata(store=store, namespace=namespace, dataset_id=dataset_id)
        if metadata is None:
            return get_json_error_response(
                content={"error": "Local dataset not found."}, status_code=HTTPStatus.NOT_FOUND, max_age=0
            )

        size_error = _get_size_limit_error_response(metadata=metadata, config=config)
        if size_error is not None:
            return size_error

        try:
            table = _load_table_from_store(store=store, file_path=metadata.file_path, dataset_format=metadata.format)
            offset = get_request_parameter_offset(request)
            length = get_request_parameter_length(request)
        except Exception as error:
            return get_json_error_response(
                content={"error": str(error)}, status_code=HTTPStatus.BAD_REQUEST, max_age=0
            )

        return get_json_ok_response(
            {
                "id": metadata.id,
                "name": metadata.name,
                "columns": metadata.columns,
                **_build_rows_payload(table=table, offset=offset, length=length),
            },
            max_age=0,
        )

    return local_dataset_rows_endpoint


def create_local_dataset_search_endpoint(config: LocalDatasetsConfig, s3_config: S3Config) -> Endpoint:
    async def local_dataset_search_endpoint(request: Request) -> Response:
        access_error = _get_access_error_response(request=request, config=config)
        if access_error is not None:
            return access_error
        namespace = _get_request_namespace(request)
        store = _create_store(config=config, s3_config=s3_config)
        dataset_id = request.path_params["dataset_id"]
        metadata = _read_metadata(store=store, namespace=namespace, dataset_id=dataset_id)
        if metadata is None:
            return get_json_error_response(
                content={"error": "Local dataset not found."}, status_code=HTTPStatus.NOT_FOUND, max_age=0
            )

        size_error = _get_size_limit_error_response(metadata=metadata, config=config)
        if size_error is not None:
            return size_error

        try:
            query = get_request_parameter(request, "query", required=True)
            offset = get_request_parameter_offset(request)
            length = get_request_parameter_length(request)
            table = _load_table_from_store(store=store, file_path=metadata.file_path, dataset_format=metadata.format)
            total_rows, page = _execute_search_query(table=table, query=query, limit=length, offset=offset)
        except Exception as error:
            return get_json_error_response(
                content={"error": str(error)}, status_code=HTTPStatus.BAD_REQUEST, max_age=0
            )

        return get_json_ok_response(
            {
                "id": metadata.id,
                "name": metadata.name,
                "query": query,
                "columns": metadata.columns,
                **_build_rows_payload(table=page, offset=offset, length=length),
                "num_rows_total": total_rows,
            },
            max_age=0,
        )

    return local_dataset_search_endpoint


def create_local_dataset_filter_endpoint(config: LocalDatasetsConfig, s3_config: S3Config) -> Endpoint:
    async def local_dataset_filter_endpoint(request: Request) -> Response:
        access_error = _get_access_error_response(request=request, config=config)
        if access_error is not None:
            return access_error
        namespace = _get_request_namespace(request)
        store = _create_store(config=config, s3_config=s3_config)
        dataset_id = request.path_params["dataset_id"]
        metadata = _read_metadata(store=store, namespace=namespace, dataset_id=dataset_id)
        if metadata is None:
            return get_json_error_response(
                content={"error": "Local dataset not found."}, status_code=HTTPStatus.NOT_FOUND, max_age=0
            )

        try:
            where = get_request_parameter(request, "where")
            orderby = get_request_parameter(request, "orderby")
            _validate_query_parameter(where, "where")
            _validate_query_parameter(orderby, "orderby")
            offset = get_request_parameter_offset(request)
            length = get_request_parameter_length(request)
            table = _load_table_from_store(store=store, file_path=metadata.file_path, dataset_format=metadata.format)
            total_rows, page = _execute_filter_query(
                table=table,
                columns=table.schema.names,
                where=where,
                orderby=orderby,
                limit=length,
                offset=offset,
            )
        except Exception as error:
            return get_json_error_response(
                content={"error": str(error)}, status_code=HTTPStatus.BAD_REQUEST, max_age=0
            )

        return get_json_ok_response(
            {
                "id": metadata.id,
                "name": metadata.name,
                "where": where,
                "orderby": orderby,
                "columns": metadata.columns,
                **_build_rows_payload(table=page, offset=offset, length=length),
                "num_rows_total": total_rows,
            },
            max_age=0,
        )

    return local_dataset_filter_endpoint


def create_delete_local_dataset_endpoint(config: LocalDatasetsConfig, s3_config: S3Config) -> Endpoint:
    async def delete_local_dataset_endpoint(request: Request) -> Response:
        access_error = _get_access_error_response(request=request, config=config)
        if access_error is not None:
            return access_error
        namespace = _get_request_namespace(request)
        store = _create_store(config=config, s3_config=s3_config)
        dataset_id = request.path_params["dataset_id"]
        metadata = _read_metadata(store=store, namespace=namespace, dataset_id=dataset_id)
        if metadata is None:
            return get_json_error_response(
                content={"error": "Local dataset not found."}, status_code=HTTPStatus.NOT_FOUND, max_age=0
            )

        if store.fs.exists(metadata.file_path):
            store.fs.rm(metadata.file_path, recursive=True)
        metadata_path = _metadata_path(store=store, namespace=namespace, dataset_id=dataset_id)
        if store.fs.exists(metadata_path):
            store.fs.rm(metadata_path, recursive=False)
        return get_json_ok_response({"status": "ok", "id": dataset_id}, max_age=0)

    return delete_local_dataset_endpoint
