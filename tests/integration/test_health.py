"""Sanity check that the app itself boots and responds."""


async def test_health_check(client) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
