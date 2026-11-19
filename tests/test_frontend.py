from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_chat_contract():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert 'fetch("/api/chat"' in javascript
    assert "company: companySelect.value" in javascript
    assert 'language: "auto"' in javascript
    assert "english_summary" in javascript
    assert "target=\"_blank\"" not in html
    assert "rel = \"noopener noreferrer\"" in javascript
    assert 'id="company"' in html
    assert 'id="clear-chat"' in html
    assert 'estateiq-logo.svg' in html
    assert "DarGlobal official public sources are currently indexed" in html
    assert "EstateIQ will not invent answers" in html
