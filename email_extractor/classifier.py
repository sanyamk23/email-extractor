"""Job-application classification.

A simple, transparent scoring model:

* each keyword/phrase trigger contributes a weighted point value;
* subject-line hits are amplified by ``SUBJECT_BOOST``;
* a handful of structural bonuses reward emails that look genuinely like an
  application (resume mention, sign-off, phone number, experience metric).

The raw score is clamped to ``[0, 1]`` and the email is classified as an
application when it meets ``CLASSIFICATION_THRESHOLD``.
"""
from __future__ import annotations

import re

from . import config


def _score_text(text: str, scope: str) -> float:
    """Return the summed trigger weight for *text* (deduplicated per pattern)."""
    total = 0.0
    for trig in config.COMPILED_TRIGGERS:
        if trig["scope"] in (scope, "both"):
            if trig["regex"].search(text):
                total += trig["weight"]
    return total


def classify(subject: str, body: str) -> tuple[bool, float]:
    """Classify an email as a job application.

    Returns ``(is_job_application, confidence_score)``.
    """
    subject_score = _score_text(subject, "subject") * config.SUBJECT_BOOST
    body_score = _score_text(body, "body")

    confidence = subject_score + body_score

    # Structural bonuses — these raise confidence for legit applications and
    # help push borderline cases over the threshold.
    if re.search(r"\b(?:resume|cv)\b", body, re.IGNORECASE):
        confidence += config.BONUS_RESUME_MENTION
    if re.search(r"(?:Sincerely|Best\s+regards|Thanks|Thank\s+you|Regards|Cheers)"
                 r"\s*,?\s*\n", body, re.IGNORECASE):
        confidence += config.BONUS_SIGNOFF
    if config.PHONE_REGEX.search(body):
        confidence += config.BONUS_PHONE
    if any(p.search(body) for p in config.EXPERIENCE_PATTERNS):
        confidence += config.BONUS_EXPERIENCE
    if any(p.search(body) for p in config.NOTICE_PERIOD_REGEXES):
        confidence += config.BONUS_NOTICE_PERIOD

    confidence = round(min(confidence, 1.0), 2)
    is_app = confidence >= config.CLASSIFICATION_THRESHOLD
    return is_app, confidence
