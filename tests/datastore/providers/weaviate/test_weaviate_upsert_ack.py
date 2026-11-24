import os
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("BEARER_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from datastore.providers.weaviate_datastore import (  # noqa: E402
    WeaviateBatchWriteError,
    WeaviateDataStore,
)
from datastore.providers import weaviate_datastore  # noqa: E402
from models.models import DocumentChunk, DocumentChunkMetadata  # noqa: E402
from server import main as server_main  # noqa: E402


class CallbackBatch:
    """Protocol stub for the v3 synchronous manual batch response."""

    def __init__(self, response_factory):
        self.response_factory = response_factory
        self.objects: List[dict] = []

    def configure(self, **kwargs):
        return self

    def add_data_object(self, **kwargs):
        self.objects.append(kwargs)

    def create_objects(self):
        objects = self.objects
        self.objects = []
        return self.response_factory(objects)


def chunks() -> Dict[str, List[DocumentChunk]]:
    return {
        "document-one": [
            DocumentChunk(
                id="document-one_0",
                text="first",
                metadata=DocumentChunkMetadata(document_id="document-one"),
                embedding=[0.1, 0.2],
            )
        ],
        "document-two": [
            DocumentChunk(
                id="document-two_0",
                text="second",
                metadata=DocumentChunkMetadata(document_id="document-two"),
                embedding=[0.3, 0.4],
            )
        ],
    }


def datastore(response_factory) -> WeaviateDataStore:
    store = object.__new__(WeaviateDataStore)
    store._new_manual_batch = lambda: CallbackBatch(response_factory)
    return store


def successful_response(objects):
    return [{"id": object_["uuid"], "result": {}} for object_ in objects]


def test_date_only_created_at_is_normalized_to_utc_midnight():
    assert (
        WeaviateDataStore._normalize_created_at("2026-11-24")
        == "2026-11-24T00:00:00Z"
    )
    assert (
        WeaviateDataStore._normalize_created_at("2026-11-24T12:34:56Z")
        == "2026-11-24T12:34:56Z"
    )
    assert WeaviateDataStore._normalize_created_at("2026-02-30") == "2026-02-30"


def test_weaviate_query_provider_imports_asyncio_for_concurrent_queries():
    assert weaviate_datastore.asyncio.gather is not None


@pytest.mark.asyncio
async def test_upsert_returns_ids_only_after_every_object_is_acknowledged():
    store = datastore(successful_response)

    assert await store._upsert(chunks()) == ["document-one", "document-two"]


@pytest.mark.asyncio
async def test_upsert_reports_partial_readonly_failure_without_claiming_failed_document():
    def partial_readonly_response(objects):
        return [
            {"id": objects[0]["uuid"], "result": {}},
            {
                "id": objects[1]["uuid"],
                "result": {
                    "errors": {
                        "error": [{"message": "disk is read-only (READONLY)"}]
                    }
                },
            },
        ]

    store = datastore(partial_readonly_response)

    with pytest.raises(WeaviateBatchWriteError) as error:
        await store._upsert(chunks())

    assert error.value.succeeded_ids == ["document-one"]
    assert error.value.failed_ids == ["document-two"]


@pytest.mark.asyncio
async def test_upsert_reports_zero_successes_when_every_object_is_rejected():
    def readonly_response(objects):
        return [
            {
                "id": object_["uuid"],
                "result": {
                    "errors": {
                        "error": [{"message": "disk is read-only (READONLY)"}]
                    }
                },
            }
            for object_ in objects
        ]

    store = datastore(readonly_response)

    with pytest.raises(WeaviateBatchWriteError) as error:
        await store._upsert(chunks())

    assert error.value.succeeded_ids == []
    assert error.value.failed_ids == ["document-one", "document-two"]


def test_upsert_http_returns_partial_acknowledgement_details(monkeypatch):
    class RejectingStore:
        async def upsert(self, documents):
            raise WeaviateBatchWriteError(
                succeeded_ids=["document-one"],
                failed_ids=["document-two"],
            )

    monkeypatch.setattr(server_main, "datastore", RejectingStore(), raising=False)
    client = TestClient(server_main.app)

    response = client.post(
        "/upsert",
        headers={"Authorization": "Bearer test-token"},
        json={
            "documents": [
                {"id": "document-one", "text": "first"},
                {"id": "document-two", "text": "second"},
            ]
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": {
            "message": "One or more Weaviate writes were not confirmed",
            "succeeded_ids": ["document-one"],
            "failed_ids": ["document-two"],
        }
    }


def test_upsert_http_keeps_the_existing_success_response(monkeypatch):
    class ConfirmingStore:
        async def upsert(self, documents):
            return [document.id for document in documents]

    monkeypatch.setattr(server_main, "datastore", ConfirmingStore(), raising=False)
    client = TestClient(server_main.app)

    response = client.post(
        "/upsert",
        headers={"Authorization": "Bearer test-token"},
        json={"documents": [{"id": "document-one", "text": "first"}]},
    )

    assert response.status_code == 200
    assert response.json() == {"ids": ["document-one"]}
