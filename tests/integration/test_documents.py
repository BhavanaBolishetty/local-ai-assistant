"""Integration tests for /documents: upload, list, and delete against a
real (temp-dir) VectorStore and a fake embedding client."""


async def test_upload_list_and_delete_document(client) -> None:
    upload = await client.post(
        "/documents",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert upload.status_code == 201
    body = upload.json()
    assert body["filename"] == "notes.txt"
    assert body["chunk_count"] == 1
    document_id = body["id"]

    listing = await client.get("/documents")
    assert listing.status_code == 200
    assert [d["id"] for d in listing.json()] == [document_id]

    delete = await client.delete(f"/documents/{document_id}")
    assert delete.status_code == 204

    listing_after = await client.get("/documents")
    assert listing_after.json() == []


async def test_unsupported_extension_is_rejected(client) -> None:
    response = await client.post(
        "/documents",
        files={"file": ("archive.zip", b"whatever", "application/zip")},
    )

    assert response.status_code == 400
