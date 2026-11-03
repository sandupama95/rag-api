from fastapi.testclient import TestClient
from app.main import app
from app.llm import get_llm

class FakeLLM:
    def generate(self, prompt: str) -> str:
        return "MOCK ANSWER"

app.dependency_overrides[get_llm] = lambda: FakeLLM()
client = TestClient(app)

def test_ask_returns_answer():
    r = client.post("/ask", json={"question": "hello?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "MOCK ANSWER"

def test_rejects_unsupported_file():
    r = client.post("/documents", files={"file": ("photo.png", b"\x89PNG\x00\x9c", "image/png")})
    assert r.status_code == 400

def test_accepts_text_file():
    r = client.post("/documents", files={"file": ("note.txt", b"Sick leave is 10 days.", "text/plain")})
    assert r.status_code == 200