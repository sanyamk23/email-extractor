"""Produce a clean cover-letter / pitch body from raw email text.

The cleaner removes:
* the envelope headers of forwarded messages (``---------- Forwarded message
  ---------`` / ``---------- Original Message --------``) while keeping the
  forwarded body that follows;
* quoted-reply threads (``On ... wrote:``, ``From:/Sent:/To:`` quote headers);
* the RFC 3676 signature separator (``--``) and anything below it;
* attachment-notification boilerplate sentences.
"""
from __future__ import annotations

import re

from . import config

# A header line that appears inside a forwarded-message envelope, e.g.
# ``From: Priyanshu <p@example.com>`` or ``Sent: Mon, 1 Jan ...``.
_ENVELOPE_HEADER_RE = re.compile(
    r"^\s*(from|to|cc|bcc|subject|date|sent)\s*:",
    re.IGNORECASE,
)


def _strip_forward_envelope(body: str) -> str:
    """For a forwarded-message separator, drop the separator line and the
    envelope header block (From/Date/Subject/To:/Sent:...), keeping the
    forwarded message body that follows it.

    Unlike a quoted-reply (which is discarded), the forwarded body is the
    actual content of interest — e.g. an applicant's cover letter — so it is
    preserved.
    """
    for marker in config.FORWARD_MARKERS:
        match = marker.search(body)
        if not match:
            continue
        rest = body[match.end():]
        lines = rest.splitlines(keepends=True)
        idx = 0
        # Skip leading blank lines left by the separator.
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        # Skip the envelope header lines.
        while idx < len(lines) and _ENVELOPE_HEADER_RE.match(lines[idx]):
            idx += 1
        # Skip blank line(s) separating the envelope from the body.
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        kept = "".join(lines[idx:])
        return body[: match.start()] + kept
    return body


def _cut_on_thread_markers(body: str) -> str:
    """Truncate quoted-reply threads; preserve forwarded-message bodies."""
    # Forwarded messages: keep the body, strip only the envelope.
    body = _strip_forward_envelope(body)
    # Quoted replies: discard everything from the first quote marker.
    for marker in config.QUOTE_MARKERS:
        match = marker.search(body)
        if match:
            return body[: match.start()]
    return body


def _strip_signature(body: str) -> str:
    """Remove the signature block below an RFC 3676 ``--`` separator."""
    lines = body.splitlines(keepends=True)
    signature_start = None
    for idx in range(len(lines) - 1, -1, -1):
        if config.SIGNATURE_DASH.match(lines[idx]):
            signature_start = idx
            break
    if signature_start is not None:
        return "".join(lines[:signature_start])
    return body


def _strip_boilerplate(body: str) -> str:
    """Remove common attachment/thank-you boilerplate sentences."""
    for pattern in config.ATTACHMENT_BOILERPLATE:
        body = pattern.sub("", body)
    return body


def _tidy(body: str) -> str:
    """Collapse excessive blank lines, trim edges, drop stray quote remnants."""
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n[ \t]*>", "", body)      # drop lone ">" reply-quote lines
    body = re.sub(r"^>*[ \t]+", "", body, flags=re.MULTILINE)
    return body.strip()


def clean_cover_letter(body: str) -> str:
    """Return the cleaned pitch body of the email."""
    body = _cut_on_thread_markers(body)
    body = _strip_signature(body)
    body = _strip_boilerplate(body)
    return _tidy(body)
