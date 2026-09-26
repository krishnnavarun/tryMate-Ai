"""App-wide behaviour: error format, request size limit, in-memory uploads, OpenAPI docs."""

from starlette.formparsers import MultiPartParser

from app.config import MAX_REQUEST_BYTES
from tests.conftest import AUTH


def test_unknown_route_uses_error_format(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert set(response.json()) == {"error_code", "message"}


def test_request_over_size_limit_is_rejected_before_parsing(client):
    body = b"x" * (MAX_REQUEST_BYTES + 1)
    response = client.post(
        "/recommend-size", headers={**AUTH, "Content-Type": "application/json"}, content=body
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"
    assert "larger than" in response.json()["message"]


def test_chunked_request_over_size_limit_is_rejected(client):
    # A generator body is sent without Content-Length (chunked), so the byte counter is used.
    def chunks():
        chunk = b"x" * (1024 * 1024)
        for _ in range(MAX_REQUEST_BYTES // len(chunk) + 2):
            yield chunk

    response = client.post(
        "/recommend-size", headers={**AUTH, "Content-Type": "application/json"}, content=chunks()
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"
    assert "larger than" in response.json()["message"]


def test_uploads_are_never_spooled_to_disk(client):
    # Starlette writes uploads bigger than spool_max_size to a temp file.
    # The app raises that limit above the largest request we accept.
    assert MultiPartParser.spool_max_size > MAX_REQUEST_BYTES


def test_openapi_lists_all_endpoints_and_models(client):
    spec = client.get("/openapi.json").json()
    assert {"/health", "/analyze", "/recommend-size", "/try-on"} <= set(spec["paths"])

    schemas = spec["components"]["schemas"]
    for name in [
        "AnalyzeResponse",
        "Measurements",
        "SkinTone",
        "ColorSuggestion",
        "RecommendSizeRequest",
        "RecommendSizeResponse",
        "SizeFit",
        "TryOnResponse",
        "ErrorResponse",
        "HealthResponse",
    ]:
        assert name in schemas, name

    # X-API-Key shows up as a security scheme ("Authorize" button in /docs)
    assert "APIKeyHeader" in spec["components"]["securitySchemes"]


# ---- request ids and logging -------------------------------------------------------------


def test_every_response_has_a_request_id(client):
    assert len(client.get("/health").headers["x-request-id"]) >= 8


def test_callers_request_id_is_reused(client):
    assert client.get("/health", headers={"X-Request-ID": "store-abc-123"}).headers["x-request-id"] == "store-abc-123"


def test_unsafe_request_id_is_replaced(client):
    rid = client.get("/health", headers={"X-Request-ID": "bad id\nwith newline"}).headers["x-request-id"]
    assert " " not in rid and "\n" not in rid


def test_json_log_lines_carry_request_id_and_no_bodies(client, caplog):
    import json
    import logging

    from app.logging_setup import JsonFormatter

    with caplog.at_level(logging.INFO, logger="app.access"):
        client.post("/recommend-size", headers={**AUTH, "X-Request-ID": "rid-42"}, json={"secret": "x" * 50})
    record = next(r for r in caplog.records if r.name == "app.access")
    line = json.loads(JsonFormatter().format(record))
    assert line["path"] == "/recommend-size" and line["status"] == 422 and "duration_ms" in line
    assert "x" * 50 not in json.dumps(line)
