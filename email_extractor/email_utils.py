"""Helpers to normalise raw email input into a uniform internal structure."""
from __future__ import annotations

import email as _email
import email.utils
import re
import html
from dataclasses import dataclass, field

from . import config as _config


@dataclass
class Email:
    """Normalised representation of an email ready for extraction."""
    from_header: str = ""
    subject: str = ""
    body: str = ""
    # extra raw metadata retained for debugging / future use
    extra_headers: dict = field(default_factory=dict)
    attachments: list = field(default_factory=list)


# ── Pre-compiled regexes for HTML-to-text and email parsing ──────────────────
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_ANCHOR_RE = re.compile(
    r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE)
_HTML_SCRIPT_RE = re.compile(r"<script.*?</script>", re.DOTALL | re.IGNORECASE)
_HTML_STYLE_RE = re.compile(r"<style.*?</style>", re.DOTALL | re.IGNORECASE)
_HTML_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_WS_RE = re.compile(r"[ \t]+")
_HTML_BLANK_RE = re.compile(r"\n{3,}")
_FROM_ENVELOPE_RE = re.compile(r'[<\[]([^@\]]+@\S+)[>\]\s]*$')
_BRACKET_TAG_RE = re.compile(r'<[^>]*>')
_AT_NONSPACE_RE = re.compile(r"@\S")


def _html_to_text(html_text: str) -> str:
    """Strip HTML tags crudely; spaCy/Markdown not required for this build."""
    text = _HTML_COMMENT_RE.sub("", html_text)
    # Preserve anchor link text + URL before stripping tags so links survive
    # as "text (URL)" in the plaintext output.
    text = _HTML_ANCHOR_RE.sub(r"\2 (\1)", text)
    text = _HTML_SCRIPT_RE.sub("", text)
    text = _HTML_STYLE_RE.sub("", text)
    text = _HTML_BR_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _HTML_WS_RE.sub(" ", text)
    text = _HTML_BLANK_RE.sub("\n\n", text)
    return text.strip()


def _extract_body(message) -> str:
    """Return the best plain-text body from a parsed message object.

    Walks the MIME tree **once**, collecting text parts by content-type
    priority (plain > html > other text/*), then returns the best candidate.
    Falls back to the first decodable part if no suitable text part exists.
    """
    if not message.is_multipart():
        payload = message.get_payload(decode=True)
        if not payload:
            return str(message.get_payload() or "")
        text = payload.decode(message.get_content_charset() or "utf-8",
                              errors="replace")
        if message.get_content_type() == "text/html":
            text = _html_to_text(text)
        return text

    best_plain = None
    best_html = None
    best_other = None
    fallback_part = None
    for part in message.walk():
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        if fallback_part is None:
            fallback_part = part
        if part.get_filename():
            continue
        text = payload.decode(part.get_content_charset() or "utf-8",
                              errors="replace")
        ctype = part.get_content_type()
        if ctype == "text/plain" and best_plain is None:
            best_plain = text
        elif ctype == "text/html" and best_html is None:
            best_html = text
        elif ctype.startswith("text/") and best_other is None:
            best_other = text
        if best_plain is not None:
            break  # highest priority — no need to keep walking

    if best_plain is not None:
        return best_plain
    if best_html is not None:
        return _html_to_text(best_html)
    if best_other is not None:
        return best_other

    # Fallback: first part with a payload (any content type).
    if fallback_part is not None:
        payload = fallback_part.get_payload(decode=True)
        if payload:
            text = payload.decode(fallback_part.get_content_charset() or "utf-8",
                                  errors="replace")
            if fallback_part.get_content_type() == "text/html":
                text = _html_to_text(text)
            return text
    return ""


def _extract_attachments(message) -> list[dict]:
    """Return metadata for each MIME part that carries a file attachment.

    Only lightweight metadata is returned (never the raw content bytes) so the
    result stays small and JSON-serialisable.  A part is treated as an
    attachment when it has a filename or an explicit ``attachment``
    Content-Disposition.
    """
    attachments: list[dict] = []
    for part in message.walk():
        filename = part.get_filename()
        disposition = part.get_content_disposition() or ""
        if not (filename or disposition == "attachment"):
            continue
        payload = part.get_payload(decode=True) or b""
        attachments.append({
            "filename": filename or "",
            "mime_type": part.get_content_type(),
            "size": len(payload),
            "content_id": (part.get("Content-ID") or "").strip(),
            "disposition": disposition or "inline",
        })
    return attachments


# Matches a leading ``Header: value`` line that RFC-822 messages always start
# with; a plain body that happens to begin with "From" (e.g. "From my notes")
# has no colon immediately after the word and is therefore treated as a body.
_RFC822_HEAD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*\s*:")


def parse_raw_email(raw: str) -> Email:
    """Parse an RFC-822 raw email string into an :class:`Email`.

    A leading line that looks like ``Header: value`` triggers RFC-822 parsing;
    anything else is treated as a plain body string.
    """
    if not raw:
        return Email(body=raw)
    first_line = next((ln for ln in raw.splitlines() if ln.strip()), "")
    if not _RFC822_HEAD_RE.match(first_line):
        # Not an RFC-822 message — treat the whole string as a body.
        return Email(body=raw)
    message = _email.message_from_string(raw)
    from_header = message.get("From", "")
    subject = message.get("Subject", "")
    body = _extract_body(message)
    extra = {h: v for h, v in message.items() if h.lower() not in
             {"from", "subject", "to", "cc", "bcc", "date", "message-id"}}
    return Email(from_header=from_header, subject=subject, body=body,
                 extra_headers=extra, attachments=_extract_attachments(message))


def parse_email_dict(data: dict) -> Email:
    """Build an :class:`Email` from a mapping with from/subject/body keys."""
    return Email(
        from_header=str(data.get("from", "") or ""),
        subject=str(data.get("subject", "") or ""),
        body=str(data.get("body", "") or ""),
        extra_headers={k: v for k, v in data.items()
                       if k.lower() not in {"from", "subject", "body"}},
    )


def parse_email(email_input) -> Email:
    """Accept a raw RFC-822 string or a dict and normalise to :class:`Email`."""
    if isinstance(email_input, str):
        return parse_raw_email(email_input)
    if isinstance(email_input, dict):
        return parse_email_dict(email_input)
    raise TypeError(
        "email_input must be a raw email string or a dict with "
        "'from', 'subject', 'body' keys.")


def name_from_from_header(from_header: str) -> str | None:
    """Extract the human name from a ``From:`` header value.

    Handles ``"Jane Doe" <jane@example.com>`` and ``Jane Doe <...>`` and
    bare ``<jane@example.com>``.
    """
    if not from_header:
        return None
    # Strip a leading display-name comment if present.
    name, _addr = email.utils.parseaddr(from_header)
    name = name.strip().strip('"').strip()
    if name:
        return name
    # Fall back to manual extraction.
    m = _FROM_ENVELOPE_RE.search(from_header)
    if m:
        from_header = from_header.replace(m.group(0), "")
    cleaned = _BRACKET_TAG_RE.sub('', from_header).strip().strip('"').strip()
    return cleaned or None


# A "From:" header-style line, e.g. "From: Priyanshu <p@example.com>".
_FORWARDED_FROM_RE = re.compile(r"^\s*from:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
# Subject prefixes that indicate a forwarded message.
_FWD_SUBJECT_RE = re.compile(r"^(?:re:\s*)?fwd?\b", re.IGNORECASE)


def extract_forwarded_sender(body: str) -> str | None:
    """Return the ``From: Name <email>`` value of a forwarded-message envelope.

    When an email is a forwarded message, the *envelope* ``From:`` line appears
    inside the body and identifies the real sender (e.g. the applicant), whereas
    the message ``From:`` header names the forwarder.  Returns that envelope
    value, or ``None`` when no forwarded ``From:`` line is present.
    """
    if not body:
        return None
    for match in _FORWARDED_FROM_RE.finditer(body):
        value = match.group(1).strip()
        # Treat it as an envelope only if it carries an email address.
        if _AT_NONSPACE_RE.search(value):
            return value
    return None


def looks_forwarded(subject: str, body: str) -> bool:
    """Heuristic: True when the message looks like a forwarded message."""
    if body and any(marker.search(body) for marker in _config.FORWARD_MARKERS):
        return True
    return bool(_FWD_SUBJECT_RE.match(subject or ""))
