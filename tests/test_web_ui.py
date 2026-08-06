"""Unit tests for the dependency-free web UI (web_ui.py)."""
from __future__ import annotations

import json
from pathlib import Path

from web_ui import (
    build_form_page,
    format_value,
    parse_eml_bytes,
    parse_multipart,
    render_result,
)

REPO = Path(__file__).resolve().parents[1]


# ── Form page ───────────────────────────────────────────────────────────────


def test_form_page_has_upload_form_and_file_input():
    page = build_form_page()
    assert "<form" in page
    assert 'name="eml_file"' in page
    assert 'type="file"' in page
    assert 'enctype="multipart/form-data"' in page


def test_form_page_renders_error_block():
    page = build_form_page("boom")
    assert "boom" in page
    assert "class=\"error\"" in page


# ── Value formatting ────────────────────────────────────────────────────────


def test_format_value_handles_all_field_types():
    assert format_value(None) == "—"
    assert format_value("") == "—"
    assert format_value(True) == "yes"
    assert format_value(False) == "no"
    assert format_value(0.703) == "0.70"
    assert format_value(["SQL", "Python"]) == "SQL, Python"
    assert format_value([]) == "—"
    assert format_value("Alex Kumar") == "Alex Kumar"


def test_format_value_escapes_html():
    assert format_value("<script>") == "&lt;script&gt;"


# ── Multipart parser ────────────────────────────────────────────────────────


def test_parse_multipart_extracts_file_field():
    boundary = "boundary123"
    file_content = b"From: a@example.com\nSubject: hi\n\nHello world\n"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="eml_file"; filename="test.eml"\r\n'
        "Content-Type: message/rfc822\r\n"
        "\r\n".encode()
    ) + file_content + f"\r\n--{boundary}--\r\n".encode()
    fields = parse_multipart(body, boundary)
    assert fields["eml_file"] == file_content


def test_parse_multipart_handles_multiple_fields():
    boundary = "X"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"_trigger\"\r\n\r\nclick\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"eml_file\"; filename=\"f.eml\"\r\n"
        "\r\n"
    ).encode() + b"\nraw eml\n" + f"\r\n--{boundary}--\r\n".encode()
    fields = parse_multipart(body, boundary)
    assert fields["_trigger"] == b"click"
    assert fields["eml_file"] == b"\nraw eml\n"


# ── Parse pipeline ──────────────────────────────────────────────────────────


def test_parse_eml_bytes_does_not_raise_on_garbage():
    result = parse_eml_bytes("not a real email !!!")
    assert isinstance(result, dict)
    assert "candidate" in result
    assert "is_job_application" in result


def test_parse_eml_bytes_runs_pipeline_on_real_eml():
    eml_path = REPO / "mails" / "Application for Data Analyst Position - Alex Kumar.eml"
    if not eml_path.exists():
        return  # real-data test only runs where the corpus is present
    result = parse_eml_bytes(eml_path.read_text(encoding="utf-8", errors="replace"))
    assert result["is_job_application"] is True
    candidate = result["candidate"]
    assert candidate["name"] == "Alex Kumar"
    assert candidate["skills"] == ["Statistics", "SQL", "Python", "Tableau"]


# ── Result rendering ─────────────────────────────────────────────────────────


def test_render_result_contains_candidate_fields():
    result = {
        "is_job_application": True,
        "confidence_score": 0.95,
        "job_role": "Data Analyst",
        "candidate": {
            "name": "Alex Kumar",
            "email": None,
            "phone": [],
            "links": [],
            "years_of_experience": None,
            "salary_expectation": None,
            "notice_period": None,
            "skills": ["Statistics", "SQL", "Python", "Tableau"],
            "education": [],
            "seniority": None,
            "location": None,
            "company": None,
            "start_date": None,
            "work_type": None,
            "languages": [],
            "certifications": [],
        },
        "sender": {"name": "Shakti Singh", "email": "recruiter@acme.com"},
        "attachments": [{"filename": "resume.pdf", "mime_type": "application/pdf",
                         "size": 12, "content_id": "", "disposition": "attachment"}],
        "clean_cover_letter": "Hello,\nRegards,\nAlex Kumar",
    }
    page = render_result(result)
    assert "Data Analyst" in page
    assert "Alex Kumar" in page
    assert "Statistics, SQL, Python, Tableau" in page  # skills formatted inline
    assert "Shakti Singh" in page                       # sender name rendered
    assert "recruiter@acme.com" in page                 # sender email rendered
    assert "Hello,\nRegards,\nAlex Kumar" in page     # cover letter preserved
    assert "application/json" not in page               # no stray raw json tag
    assert "Start date" in page                        # new candidate fields render
    assert "Work type" in page
    assert "Languages" in page
    assert "Certifications" in page
    assert "Attachments" in page
    assert "resume.pdf" in page
    # the raw JSON panel is present (escaped) but hidden until toggled
    assert "Raw JSON" in page
    assert json.dumps(result["candidate"], default=str) is not None


def test_render_result_html_escapes_cover_letter():
    result = {
        "is_job_application": False,
        "confidence_score": 0.0,
        "job_role": None,
        "candidate": {},
        "clean_cover_letter": "<script>alert(1)</script>",
    }
    page = render_result(result)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
