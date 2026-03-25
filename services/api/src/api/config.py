# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Mapping
from dataclasses import dataclass, field
from environs import Env

from libapi.config import ApiConfig
from libcommon.config import (
    AssetsConfig,
    CacheConfig,
    CachedAssetsConfig,
    CloudFrontConfig,
    CommonConfig,
    LogConfig,
    QueueConfig,
    S3Config,
    STORAGE_PROTOCOL_VALUES,
    StorageProtocol,
    is_storage_protocol,
)
from marshmallow.validate import OneOf
from libcommon.processing_graph import InputType


@dataclass(frozen=True)
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    cached_assets: CachedAssetsConfig = field(default_factory=CachedAssetsConfig)
    cloudfront: CloudFrontConfig = field(default_factory=CloudFrontConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    log: LogConfig = field(default_factory=LogConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    s3: S3Config = field(default_factory=S3Config)
    local_datasets: "LocalDatasetsConfig" = field(default_factory=lambda: LocalDatasetsConfig())

    @classmethod
    def from_env(cls) -> "AppConfig":
        common_config = CommonConfig.from_env()
        return cls(
            common=common_config,
            assets=AssetsConfig.from_env(),
            cache=CacheConfig.from_env(),
            cached_assets=CachedAssetsConfig.from_env(),
            cloudfront=CloudFrontConfig.from_env(),
            log=LogConfig.from_env(),
            queue=QueueConfig.from_env(),
            api=ApiConfig.from_env(hf_endpoint=common_config.hf_endpoint),
            s3=S3Config.from_env(),
            local_datasets=LocalDatasetsConfig.from_env(),
        )


LOCAL_DATASETS_STORAGE_PROTOCOL: StorageProtocol = "file"
LOCAL_DATASETS_STORAGE_ROOT = "/storage/local-datasets"
LOCAL_DATASETS_MAX_UPLOAD_SIZE_BYTES = 200_000_000
LOCAL_DATASETS_REQUIRE_BEARER_TOKEN = True
LOCAL_DATASETS_MAX_IN_MEMORY_PROCESSING_BYTES = 100_000_000


@dataclass(frozen=True)
class LocalDatasetsConfig:
    storage_protocol: StorageProtocol = LOCAL_DATASETS_STORAGE_PROTOCOL
    storage_root: str = LOCAL_DATASETS_STORAGE_ROOT
    max_upload_size_bytes: int = LOCAL_DATASETS_MAX_UPLOAD_SIZE_BYTES
    require_bearer_token: bool = LOCAL_DATASETS_REQUIRE_BEARER_TOKEN
    max_in_memory_processing_bytes: int = LOCAL_DATASETS_MAX_IN_MEMORY_PROCESSING_BYTES

    @classmethod
    def from_env(cls) -> "LocalDatasetsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("LOCAL_DATASETS_"):
            storage_protocol_str = env.str(
                name="STORAGE_PROTOCOL",
                default=LOCAL_DATASETS_STORAGE_PROTOCOL,
                validate=OneOf(
                    STORAGE_PROTOCOL_VALUES,
                    error="LOCAL_DATASETS_STORAGE_PROTOCOL must be one of: {choices}",
                ),
            )
            if is_storage_protocol(storage_protocol_str):
                storage_protocol = storage_protocol_str
            else:
                raise ValueError(f"Invalid storage protocol: {storage_protocol_str}")
            return cls(
                storage_protocol=storage_protocol,
                storage_root=env.str(name="STORAGE_ROOT", default=LOCAL_DATASETS_STORAGE_ROOT),
                max_upload_size_bytes=env.int(
                    name="MAX_UPLOAD_SIZE_BYTES", default=LOCAL_DATASETS_MAX_UPLOAD_SIZE_BYTES
                ),
                require_bearer_token=env.bool(
                    name="REQUIRE_BEARER_TOKEN", default=LOCAL_DATASETS_REQUIRE_BEARER_TOKEN
                ),
                max_in_memory_processing_bytes=env.int(
                    name="MAX_IN_MEMORY_PROCESSING_BYTES",
                    default=LOCAL_DATASETS_MAX_IN_MEMORY_PROCESSING_BYTES,
                ),
            )


ProcessingStepNameByInputType = Mapping[InputType, str]

ProcessingStepNameByInputTypeAndEndpoint = Mapping[str, ProcessingStepNameByInputType]


@dataclass(frozen=True)
class EndpointConfig:
    """Contains the endpoint config specification to relate with step names.
    The list of processing steps corresponds to the priority in which the response
    has to be reached. The cache from the first step in the list will be used first
    then, if it's an error or missing, the second one, etc.
    The related steps depend on the query parameters passed in the request
    (dataset, config, split)
    """

    processing_step_name_by_input_type_and_endpoint: ProcessingStepNameByInputTypeAndEndpoint = field(
        default_factory=lambda: {
            "/splits": {
                "dataset": "dataset-split-names",
                "config": "config-split-names",
            },
            "/first-rows": {"split": "split-first-rows"},
            "/parquet": {
                "dataset": "dataset-parquet",
                "config": "config-parquet",
            },
            "/info": {"dataset": "dataset-info", "config": "config-info"},
            "/size": {
                "dataset": "dataset-size",
                "config": "config-size",
            },
            "/opt-in-out-urls": {
                "dataset": "dataset-opt-in-out-urls-count",
                "config": "config-opt-in-out-urls-count",
                "split": "split-opt-in-out-urls-count",
            },
            "/presidio-entities": {
                "dataset": "dataset-presidio-entities-count",
            },
            "/is-valid": {
                "dataset": "dataset-is-valid",
                "config": "config-is-valid",
                "split": "split-is-valid",
            },
            "/statistics": {"split": "split-descriptive-statistics"},
            "/compatible-libraries": {"dataset": "dataset-compatible-libraries"},
            "/croissant-crumbs": {"dataset": "dataset-croissant-crumbs"},
            "/hub-cache": {"dataset": "dataset-hub-cache"},
        }
    )

    @classmethod
    def from_env(cls) -> "EndpointConfig":
        # TODO: allow passing the mapping between endpoint and processing steps via env vars
        return cls()
