"""Extract the target job role from subject / body text."""
from __future__ import annotations

import re

from . import config


def _clean_role(raw: str) -> str:
    """Tidy an extracted role string: trim company suffixes and stray words."""
    role = raw.strip().rstrip(".,;:()[]")

    # Cut trailing company markers: " - Acme Corp", " at Acme Corp".
    marker = re.search(r"\s+-\s+|\s+at\s+", role)
    if marker:
        role = role[: marker.start()]

    # Drop a trailing descriptor word left behind by the regex.
    role = re.sub(r"\s+(position|role|opening|vacancy|job)\s*$", "",
                  role, flags=re.IGNORECASE)

    # Strip trailing punctuation left after trimming.
    role = role.strip().rstrip(".,;:()[]")

    if not role:
        return ""

    # Normalise against the canonical role dictionary for consistent casing
    # ("software engineer" -> "Software Engineer").
    canonical = config.ROLE_CANONICAL.get(role.lower())
    if canonical:
        return canonical

    # Ensure the role starts with a capital letter.
    if role[0].islower():
        role = role[0].upper() + role[1:]
    return role


def _match_patterns(text: str) -> str | None:
    """Try the high-priority extraction patterns in order."""
    for pattern in config.COMPILED_ROLE_PATTERNS:
        match = pattern.search(text)
        if match:
            role = _clean_role(match.group(1))
            if role:
                return role
    return None


def _match_dictionary(text: str) -> str | None:
    """Fallback: look for a known role term anywhere in the text.

    Roles are tested longest-first so the most specific match wins.
    """
    lowered = text.lower()
    # Pre-sort by length (longest first) only once — cheap enough per call.
    sorted_roles = sorted(config.ROLE_DICTIONARY, key=len, reverse=True)
    for role in sorted_roles:
        if role.lower() in lowered:
            return role
    return None


def extract_job_role(subject: str, body: str) -> str | None:
    """Return the best-guess job role, or ``None`` when nothing surfaces."""
    combined = f"{subject} {body}".strip()

    # 1. Pattern-based extraction (priority ordered).
    for text in (subject, combined):
        role = _match_patterns(text)
        if role:
            return role

    # 2. Dictionary fallback over the whole message.
    return _match_dictionary(combined)
