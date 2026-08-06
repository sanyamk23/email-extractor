"""Unit tests for raw-email parsing and From-header handling."""
from email_extractor.email_utils import (
    parse_email, parse_raw_email, parse_email_dict, name_from_from_header,
)
from pathlib import Path


def test_parse_dict():
    email = parse_email({"from": "Jane <jane@x.com>",
                         "subject": "Hi", "body": "Hello"})
    assert email.from_header == "Jane <jane@x.com>"
    assert email.subject == "Hi"
    assert email.body == "Hello"


def test_parse_raw_rfc822():
    raw = ("From: Jane Doe <jane@example.com>\n"
           "Subject: Application for Engineer\n"
           "Content-Type: text/plain; charset=utf-8\n"
           "\n"
           "Body text here.")
    email = parse_raw_email(raw)
    assert email.from_header == "Jane Doe <jane@example.com>"
    assert email.subject == "Application for Engineer"
    assert "Body text here" in email.body


def test_parse_raw_html_body():
    raw = ("Content-Type: text/html; charset=utf-8\n"
           "Subject: HTML\n"
           "\n"
           "<p>Hello <b>world</b>.</p><p>Second line.</p>")
    email = parse_raw_email(raw)
    # Subject header is on the second line, but first line "Content-Type:" is
    # still a valid header so RFC-822 parsing kicks in.
    assert email.subject == "HTML"
    assert "Hello" in email.body and "world" in email.body
    assert "<" not in email.body


def test_plain_string_treated_as_body():
    text = "From my experience, I am a great fit."
    email = parse_email(text)
    assert email.body == text
    assert email.subject == ""


def test_name_from_from_header_quoted():
    assert name_from_from_header('"Jane Doe" <jane@x.com>') == "Jane Doe"


def test_name_from_from_header_unquoted():
    assert name_from_from_header("Jane Doe <jane@x.com>") == "Jane Doe"


def test_name_from_from_header_bare():
    assert name_from_from_header("<jane@x.com>") is None


def test_name_from_from_header_empty():
    assert name_from_from_header("") is None


def test_parse_email_rejects_bad_type():
    import pytest
    with pytest.raises(TypeError):
        parse_email(12345)


def test_parse_email_attachments_extracted():
    eml = Path(__file__).parents[1] / "tests" / "fixtures" / "attachment_sample.eml"
    email = parse_email(eml.read_text(encoding="utf-8", errors="replace"))
    assert email.subject == "Application for Data Analyst Position"
    atts = email.attachments
    assert len(atts) == 1
    att = atts[0]
    assert att["filename"] == "resume.pdf"
    assert att["mime_type"] == "application/pdf"
    assert att["disposition"] == "attachment"
    assert att["size"] == 12  # b"hello resume"
    assert att["content_id"] == "<resume.pdf@localhost>"


def test_parse_email_inline_attachments_not_listed_when_none():
    raw = ("From: Jane <jane@example.com>\n"
           "Subject: hi\n"
           "\n"
           "Just a plain note.")
    assert parse_email(raw).attachments == []
