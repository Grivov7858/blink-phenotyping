import io
import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index_redirects_to_analyze(client):
    resp = client.get("/")
    assert resp.status_code == 302 or resp.status_code == 200

def test_analyze_page_loads(client):
    resp = client.get("/analyze")
    assert resp.status_code == 200
    assert b"Analyze" in resp.data

def test_statistics_page_loads(client):
    resp = client.get("/statistics")
    assert resp.status_code == 200
    assert b"Statistics" in resp.data

def test_analyze_missing_fields(client):
    resp = client.post("/analyze", data={})
    assert resp.status_code == 200
    assert b"required" in resp.data.lower() or b"error" in resp.data.lower()

def test_analyze_invalid_age(client):
    data = {"subject_id": "001", "age": "25", "screen_time_h": "3.0", "symptom_score": "20", "nibut_s": "7.5"}
    video = (io.BytesIO(b"fake"), "test.mp4")
    resp = client.post("/analyze", data={**data, "video": video})
    assert b"1" in resp.data and b"18" in resp.data

def test_statistics_no_file(client):
    resp = client.post("/statistics", data={})
    assert resp.status_code == 200
    assert b"error" in resp.data.lower() or b"required" in resp.data.lower()
