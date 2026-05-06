from project_name.rules import basic_filter
from project_name.validate import validate_sft_record


def test_basic_filter_drop_short_text() -> None:
    assert not basic_filter({"text": "hi"}, min_text_length=3, drop_empty=True)


def test_validate_sft_record_ok() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    ok, errors = validate_sft_record(row)
    assert ok is True
    assert errors == []
