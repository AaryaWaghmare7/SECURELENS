from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
import pytest
from PIL import Image
from sqlalchemy import select

from app.models import User, Analysis


def upload(name="test.png", color="teal"):
    buffer = BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, format="PNG")
    return (name, buffer.getvalue(), "image/png")


def analyze(client, headers, **options):
    return client.post("/api/analyze", files={"image": upload()}, headers=headers,
                       data={"save": "true", **options})


def test_registration_hash_and_session(client, signed_in):
    assert client.get("/api/auth/me").json()["user"]["name"] == "Researcher"
    with client.app.state.session_factory() as db:
        user = db.scalar(select(User))
        assert user.password_hash.startswith("$argon2")
        assert user.password_hash != "a-test-password-123"
    response = client.post("/api/auth/register", json={"name": "Other", "email": "researcher@example.com", "password": "a-test-password-123"})
    assert response.status_code == 409


def test_login_wrong_password_and_explicit_logout(client, signed_in):
    cookie = client.cookies.get("securelens_session")
    assert client.post("/api/auth/login", json={"email": "researcher@example.com", "password": "wrong"}).status_code == 401
    response = client.post("/api/auth/login", json={"email": "researcher@example.com", "password": "a-test-password-123"})
    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=lax" in response.headers["set-cookie"]
    assert client.post("/api/auth/logout", headers={"X-CSRF-Token": response.json()["csrf_token"]}).status_code == 200
    client.cookies.set("securelens_session", cookie)
    assert client.get("/api/auth/me").status_code == 401


def test_protected_routes_and_csrf(client, signed_in):
    assert client.post("/api/analyze", files={"image": upload()}).status_code == 403
    assert client.post("/api/analyze", files={"image": upload()}, headers={**signed_in, "Origin": "https://untrusted.example"}).status_code == 403
    client.cookies.clear()
    assert client.get("/api/history").status_code == 401
    assert analyze(client, signed_in).status_code == 401


def test_single_analysis_and_reports(client, signed_in):
    response = analyze(client, signed_in)
    assert response.status_code == 200
    record = response.json()
    item = record["result"]["items"][0]
    assert item["metrics"]["width"] == 64
    assert item["visualizations"]["ela"].startswith("data:image/png;base64,")
    assert item["visualizations"]["fft"].startswith("data:image/png;base64,")
    assert item["classification"]["probability"] is None
    assert item["classification"]["manipulation_indicators"] == "NOT_ESTABLISHED"
    report = client.get(f'/api/reports/{record["id"]}')
    assert report.status_code == 200
    assert report.json()["title"] == "SecureLens Analysis Report"
    assert "visualizations" not in report.json()["result"]["items"][0]
    assert client.get(f'/api/reports/{record["id"]}?format=pdf').content.startswith(b"%PDF")
    assert client.get(f'/api/reports/{record["id"]}?format=csv').text.startswith("filename,")
    history = client.get("/api/history").json()
    assert len(history) == 1
    saved = client.get(f'/api/history/{record["id"]}').json()
    assert saved["result"]["items"][0]["visualizations"] == {}


@pytest.mark.parametrize("file,status", [(('test.txt', b'hello', 'text/plain'), 415),
    (('test.png', b'corrupt', 'image/png'), 422), (('test.jpg', upload()[1], 'image/jpeg'), 415),
    (('empty.png', b'', 'image/png'), 422)])
def test_upload_validation(client, signed_in, file, status):
    assert client.post("/api/analyze", files={"image": file}, headers=signed_in).status_code == status


def test_size_limits(client, signed_in):
    client.app.state.settings.max_upload_bytes = 10
    assert client.post("/api/analyze", files={"image": upload()}, headers=signed_in).status_code == 413
    client.app.state.settings.max_request_bytes = 10
    assert client.post("/api/analyze", files={"image": upload()}, headers=signed_in).status_code == 413


def test_streamed_body_limit_without_content_length(client):
    client.app.state.settings.max_request_bytes = 20
    response = client.post("/api/auth/login", content=iter([b'{"email":"', b'x' * 100, b'"}']),
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_signin_rate_limit(client):
    for _ in range(client.app.state.settings.auth_rate_limit):
        assert client.post("/api/auth/login", json={"email": "absent@example.com", "password": "bad"}).status_code == 401
    response = client.post("/api/auth/login", json={"email": "absent@example.com", "password": "bad"})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"


def test_expired_or_tampered_session(client, signed_in):
    import jwt
    claims = jwt.decode(client.cookies.get("securelens_session"), client.app.state.settings.jwt_secret, algorithms=["HS256"])
    claims["exp"] = int((datetime.now(timezone.utc) - timedelta(minutes=1)).timestamp())
    client.cookies.set("securelens_session", jwt.encode(claims, client.app.state.settings.jwt_secret, algorithm="HS256"))
    assert client.get("/api/auth/me").status_code == 401
    client.cookies.set("securelens_session", "tampered.cookie.value")
    assert client.get("/api/history").status_code == 401
def test_live_is_ephemeral_and_uses_the_same_rules(client, signed_in):
    normal = client.post("/api/analyze", files={"image": upload()}, headers=signed_in).json()
    live = client.post("/api/analyze", files={"image": upload()}, data={"capture_source": "webcam"}, headers=signed_in).json()
    assert live["analysis_type"] == "live"
    assert live["saved"] is False and live["id"] is None
    assert normal["result"]["items"][0]["classification"] == live["result"]["items"][0]["classification"]
    assert client.get("/api/history").json() == []
    assert not list(client.app.state.settings.storage_path.rglob("*.png"))


def test_batch_partial_failures_and_stream_progress(client, signed_in):
    files = [("images", upload()), ("images", ("broken.png", b"bad", "image/png"))]
    response = client.post("/api/analyze/batch?stream=true", files=files, data={"save": "true"}, headers=signed_in)
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert [event["completed"] for event in events[:-1]] == [1, 2]
    record = events[-1]["record"]
    assert record["status"] == "partial"
    assert record["result"]["items"][1]["filename"] == "broken.png"
    assert "failed" in client.get(f'/api/reports/{record["id"]}?format=csv').text
    assert client.post("/api/analyze/batch", files=[("images", upload())] * 11, headers=signed_in).status_code == 422
    failed = client.post("/api/analyze/batch", headers=signed_in, data={"save": "true"},
                         files=[("images", ("bad.png", b"not-an-image", "image/png"))]).json()
    assert failed["status"] == "failed"
    assert client.get("/api/history").json()[0]["label"] == "0 / 1 images analyzed"


def test_compare_same_image(client, signed_in):
    response = client.post("/api/analyze/compare", files={"image_a": upload(), "image_b": upload()}, data={"save": "true"}, headers=signed_in)
    assert response.status_code == 200
    assert response.json()["result"]["comparison"]["similarity"] == 100
    assert len(response.json()["result"]["items"]) == 2
    assert client.get("/api/dashboard").json()["compare"] == 1


def test_private_history_and_images(client, signed_in):
    record = analyze(client, signed_in, retain_images="true").json()
    own = client.get(f'/api/history/{record["id"]}').json()
    url = own["result"]["items"][0]["visualizations"]["original"]
    assert client.get(url).status_code == 200
    other = client.post("/api/auth/register", json={"name": "Other", "email": "other@example.com", "password": "another-test-pass"}).json()
    headers = {"X-CSRF-Token": other["csrf_token"]}
    assert client.get("/api/history").json() == []
    for path in (f'/api/history/{record["id"]}', f'/api/reports/{record["id"]}', url):
        assert client.get(path).status_code == 404
    assert client.delete(f'/api/history/{record["id"]}', headers=headers).status_code == 404


def test_retention_and_deletion_remove_files(client, signed_in):
    record = analyze(client, signed_in, retain_images="true").json()
    with client.app.state.session_factory() as db:
        row = db.get(Analysis, record["id"])
        row.images_expire_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
    assert client.get(f'/api/history/{record["id"]}').json()["result"]["items"][0]["visualizations"] == {}
    assert not list(client.app.state.settings.storage_path.rglob("*.png"))
    assert client.delete(f'/api/history/{record["id"]}', headers=signed_in).status_code == 204
    assert client.get("/api/history").json() == []
    assert not list(client.app.state.settings.storage_path.rglob("*.json"))


def test_settings_consent_and_filename_safety(client, signed_in):
    assert client.patch("/api/settings", json={"name": "   "}, headers=signed_in).status_code == 422
    assert client.patch("/api/settings", json={"name": "Updated", "retain_images": True}, headers=signed_in).status_code == 200
    assert client.get("/api/settings").json()["retain_images"] is True
    consent = client.put("/api/cookie-consent", json={"essential": False, "authentication": False, "analytics": True}, headers=signed_in)
    assert consent.json() == {"essential": True, "authentication": True, "analytics": False}
    response = client.post("/api/analyze", files={"image": upload("../../unsafe.png")}, data={"save": "true", "retain_images": "true"}, headers=signed_in)
    assert response.json()["filename"] == "unsafe.png"
    with pytest.raises(ValueError):
        client.app.state.storage.path("../private")
