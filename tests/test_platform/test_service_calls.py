"""核对预测调用原文、访问范围与保存期限。"""

import base64
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from hscredit.platform import create_app
from hscredit.platform.service_calls import DAY

ADMIN = {"Authorization": "Bearer platform-management-token"}


def publish(client):
    response = client.post("/api/services", headers=ADMIN, json={"name": "规则", "rules": [{"name": "低分", "expression": "评分 < 500"}]})
    assert response.status_code == 201, response.text
    service = response.json()
    return service["endpoint"], {"Authorization": "Bearer " + service["token"]}, "/api/services/" + service["id"]


def test_wire_bodies_errors_and_access_scope(tmp_path):
    with TestClient(create_app(tmp_path, token="platform-management-token"), raise_server_exceptions=False) as client:
        endpoint, token, admin = publish(client)
        _, _, other = publish(client)
        raw = ('{ "records": [{"评分":499,"流水":9876543210987654321},{"评分":500}] }\n').encode()
        headers = {**token, "Content-Type": "application/json", "Origin": "https://client.example", "Cookie": "a=secret", "X-API-Key": "secret"}
        cases = [(raw, headers, 200), (raw, {"Content-Type": "application/json"}, 401), (b'{"records":', headers, 422), (b'{"records":[{}]}', headers, 422), (b"\xff", headers, 400)]
        ids = []
        for body, auth, expected in cases:
            response = client.post(endpoint + "?channel=web&access_token=secret", headers=auth, content=body)
            assert response.status_code == expected, response.text
            call_id = response.headers["x-request-id"]
            ids.append(call_id)
            detail = client.get(admin + "/calls/" + call_id, headers=ADMIN).json()
            captured = detail["request_body"]
            encoded = base64.b64decode(captured["text"]) if captured["encoding"] == "base64" else captured["text"].encode()
            assert encoded == body
            assert detail["response_body"]["text"].encode() == response.content
            assert detail["status"] == expected and detail["duration_ms"] >= 0
            assert detail["target"].startswith(endpoint + "?channel=web")
            assert "secret" not in str(detail)
            assert token["Authorization"] not in str(detail)
        assert len(set(ids)) == len(ids)
        assert client.get(admin + "/calls", headers=token).status_code == 401
        assert client.get(other + "/calls/" + ids[0], headers=ADMIN).status_code == 404
        page = client.get(admin + "/calls", headers=ADMIN, params={"page": 2, "page_size": 2, "failed_only": True}).json()
        assert page["retention_days"] is None and page["total"] == 4
        assert [row["id"] for row in page["items"]] == [ids[2], ids[1]]
        assert all("request_body" not in row and "response_body" not in row for row in page["items"])
        assert client.get(admin + "/calls", headers=ADMIN, params={"call_id": ids[0]}).json()["total"] == 1
        schema = client.get(endpoint.replace("/predict", "/openapi.json"), headers=token)
        assert schema.status_code == 200 and "x-request-id" not in schema.headers

        client.patch(admin, headers=ADMIN, json={"enabled": False})
        offline = client.post(endpoint, content=raw, headers=headers)
        record = client.get(admin + "/calls/" + offline.headers["x-request-id"], headers=ADMIN).json()
        assert record["status"] == offline.status_code == 503
        assert record["response_body"]["text"] == offline.text


def test_retention_prunes_old_rows_per_service_and_on_restart(tmp_path):
    path = tmp_path / "services" / "calls.sqlite3"
    with TestClient(create_app(tmp_path)) as client:
        endpoint, token, admin = publish(client)
        other_endpoint, other_token, other_admin = publish(client)

        def predict(i):
            response = client.post(endpoint, headers=token, json={"records": [{"评分": i}]})
            assert response.status_code == 200
            return response.headers["x-request-id"]

        with ThreadPoolExecutor(max_workers=5) as executor:
            ids = list(executor.map(predict, range(10)))
        assert len(set(ids)) == client.get(admin + "/calls").json()["total"] == 10
        other_id = client.post(other_endpoint, headers=other_token, json={"records": [{"评分": 100}]}).headers["x-request-id"]
        with sqlite3.connect(path) as db:
            db.execute("UPDATE calls SET started_at=? WHERE id IN (?,?)", (time.time() - 4 * DAY, ids[0], other_id))
        for invalid in (0, -1, True, 0.5):
            assert client.put(admin + "/retention", json={"retention_days": invalid}).status_code == 422
        assert client.put(admin + "/retention", json={}).status_code == 422
        assert client.put(admin + "/retention", json={"retention_days": 2}).status_code == 200
        assert client.get(admin + "/calls/" + ids[0]).status_code == 404
        assert client.get(other_admin + "/calls/" + other_id).status_code == 200
        with sqlite3.connect(path) as db:
            assert not db.execute("SELECT 1 FROM calls WHERE id=?", (ids[0],)).fetchone()
            db.execute("UPDATE calls SET started_at=? WHERE id=?", (time.time() - 3 * DAY, ids[1]))

    with TestClient(create_app(tmp_path)) as client:
        page = client.get(admin + "/calls").json()
        assert page["retention_days"] == 2 and page["total"] == 8
        with sqlite3.connect(path) as db:
            assert not db.execute("SELECT 1 FROM calls WHERE id=?", (ids[1],)).fetchone()
            db.execute("UPDATE calls SET started_at=? WHERE id=?", (time.time() - 3 * DAY, ids[2]))
        assert client.get(admin + "/calls/" + ids[2]).status_code == 404
        assert client.put(admin + "/retention", json={"retention_days": None}).status_code == 200
        assert client.get(admin + "/calls/" + ids[2]).status_code == 404
        assert client.get(admin + "/calls").json()["retention_days"] is None
        assert client.get(other_admin + "/calls/" + other_id).status_code == 200


def test_internal_error_keeps_actual_http_response(tmp_path, monkeypatch):
    from hscredit.platform import serving

    with TestClient(create_app(tmp_path), raise_server_exceptions=False) as client:
        endpoint, token, admin = publish(client)

        def unavailable(_path):
            raise RuntimeError("制品读取失败")

        monkeypatch.setattr(serving.joblib, "load", unavailable)
        response = client.post(endpoint, headers=token, json={"records": [{"评分": 501}]})
        assert response.status_code == 500
        record = client.get(admin + "/calls/" + response.headers["x-request-id"]).json()
        assert record["status"] == 500
        assert record["response_body"]["text"].encode() == response.content
