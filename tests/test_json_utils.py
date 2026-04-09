from app.core.json_utils import extract_json_object


def test_extract_plain_json():
    text = '{"passed": true, "reason": "ok", "missing_evidence": [], "rewrite_suggestion": ""}'
    data = extract_json_object(text)
    assert data['passed'] is True


def test_extract_markdown_json_block():
    text = """```json
{"passed": false, "reason": "bad", "missing_evidence": ["x"], "rewrite_suggestion": "retry"}
```"""
    data = extract_json_object(text)
    assert data['passed'] is False
    assert data['missing_evidence'] == ['x']
