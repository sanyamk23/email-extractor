"""Top-level orchestration: parse a single email into structured JSON."""
from __future__ import annotations

import json

from . import classifier, candidate_extractor, role_extractor, body_cleaner
from .email_utils import parse_email, extract_forwarded_sender, looks_forwarded


def parse_job_application(email_input: str | dict) -> dict:
    """Parse an email (raw RFC-822 string or a dict with from/subject/body).

    Returns a dict matching the schema:
      {
        "is_job_application": bool,
        "confidence_score": float,
        "job_role": str | None,
        "candidate": {name, email, phone, links, years_of_experience,
                      salary_expectation, notice_period, skills, education,
                      seniority, location, company, start_date, work_type,
                      languages, certifications},
        "sender": {name, email},
        "attachments": [{filename, mime_type, size, content_id, disposition}],
        "clean_cover_letter": str,
      }

    For forwarded messages the candidate is attributed to the *forwarded*
    sender (the real applicant) rather than the forwarder who appears in the
    message ``From:`` header.
    """
    email = parse_email(email_input)

    is_app, confidence = classifier.classify(email.subject, email.body)
    clean_body = body_cleaner.clean_cover_letter(email.body)

    result = {
        "is_job_application": is_app,
        "confidence_score": confidence,
        "job_role": None,
        "candidate": {
            "name": None,
            "email": None,
            "phone": [],
            "links": [],
            "years_of_experience": None,
            "salary_expectation": None,
            "notice_period": None,
            "skills": [],
            "education": [],
            "seniority": None,
            "location": None,
            "company": None,
            "start_date": None,
            "work_type": None,
            "languages": [],
            "certifications": [],
        },
        "sender": candidate_extractor.extract_sender(email.from_header),
        "attachments": email.attachments,
        "clean_cover_letter": clean_body,
    }

    if is_app:
        # For a forwarded message, the real sender (applicant) is the envelope
        # "From:" line inside the body, not the forwarder in the header.
        candidate_from = email.from_header
        if looks_forwarded(email.subject, email.body):
            forwarded = extract_forwarded_sender(email.body)
            if forwarded:
                candidate_from = forwarded

        result["job_role"] = role_extractor.extract_job_role(
            email.subject, clean_body)
        result["candidate"] = candidate_extractor.extract_candidate(
            candidate_from, clean_body)

    return result


def to_json(result: dict, indent: int = 2) -> str:
    """Serialise a parse result to pretty JSON (preserves key order)."""
    return json.dumps(result, indent=indent, ensure_ascii=False)
