from app import __version__


def test_health_needs_no_api_key(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_version_is_contract_version():
    assert __version__ == "0.1.0"
