#!/usr/bin/env python3
"""Load WAM RPC message classes from an installed Proto descriptor set."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Type

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


PACKAGE = "wam.rpc.v1"
MESSAGE_NAMES = (
    "GetModelInfoRequest",
    "GetModelInfoResponse",
    "CreateSessionRequest",
    "CreateSessionResponse",
    "PredictRequest",
    "PredictResponse",
    "ResetSessionRequest",
    "ResetSessionResponse",
    "CloseSessionRequest",
    "CloseSessionResponse",
    "Error",
)


def _populate_pool(pool: descriptor_pool.DescriptorPool, files) -> None:
    """Add descriptor files while tolerating non-topological set ordering."""
    pending = list(files)
    while pending:
        deferred = []
        last_error = None
        for item in pending:
            try:
                pool.Add(item)
            except TypeError as error:
                deferred.append(item)
                last_error = error
        if len(deferred) == len(pending):
            names = ", ".join(item.name for item in deferred)
            raise ValueError(
                f"descriptor dependencies cannot be resolved for: {names}"
            ) from last_error
        pending = deferred


def load_types(
    descriptor: Path,
    names: Iterable[str] = MESSAGE_NAMES,
) -> Dict[str, Type]:
    """Construct dynamic WAM message classes without invoking protoc."""
    descriptor = Path(descriptor)
    if not descriptor.is_file():
        raise FileNotFoundError(f"WAM Proto descriptor not found: {descriptor}")
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    try:
        descriptor_set.ParseFromString(descriptor.read_bytes())
    except Exception as error:
        raise ValueError(f"invalid WAM Proto descriptor: {descriptor}") from error
    if not descriptor_set.file:
        raise ValueError(f"WAM Proto descriptor contains no files: {descriptor}")

    pool = descriptor_pool.DescriptorPool()
    _populate_pool(pool, descriptor_set.file)
    result = {}
    for name in names:
        qualified_name = f"{PACKAGE}.{name}"
        try:
            message = pool.FindMessageTypeByName(qualified_name)
        except KeyError as error:
            raise ValueError(
                f"WAM Proto descriptor is missing message {qualified_name}"
            ) from error
        result[name] = message_factory.GetMessageClass(message)
    return result
