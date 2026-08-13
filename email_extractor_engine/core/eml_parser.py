"""EML loader: parse RFC-822 email files/strings into a uniform structure.

Zero required runtime dependencies — uses the stdlib ``email`` module.
``beautifulsoup4`` is auto-detected for richer HTML-to-text conversion; when
absent a regex-based fallback is used.
"""
from __future__ import annotations

import email as _email
import email.utils
import html as _html
import os
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Optional BeautifulSoup import (graceful fallback) ──────────────────────────
try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False


# ── Pre-compiled regexes ────────────────────────────────────────────────────────
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_ANCHOR_RE = re.compile(
    r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_HTML_SCRIPT_RE = re.compile(r"<script.*?</script>", re.DOTALL | re.IGNORECASE)
_HTML_STYLE_RE = re.compile(r"<style.*?</style>", re.DOTALL | re.IGNORECASE)
_HTML_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_WS_RE = re.compile(r"[ \t]+")
_HTML_BLANK_RE = re.compile(r"\n{3,}")

# Detects whether a raw text blob is an RFC-822 email (starts with standard
# headers) vs a plain body.  We require at least one well-known header keyword
# so that body-only text like "Name: Alice. Role: Engineer" is not mistaken
# for an RFC-822 message.
_RFC822_HEAD_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9-]*\s*:",
    re.IGNORECASE,
)
# Well-known RFC-822 / MIME header keywords that signal a real email header.
_KNOWN_HEADERS = frozenset({
    "from", "to", "cc", "bcc", "subject", "date", "message-id", "message_id",
    "mime-version", "mime_version", "content-type", "content_type",
    "content-transfer-encoding", "content_transfer_encoding",
    "content-disposition", "content_disposition", "content-id", "content_id",
    "reply-to", "reply_to", "sender", "return-path", "received",
    "delivered-to",
    "dkim-signature", "dkim_signature",
    # Authentication / routing headers (common in Gmail/GSuite emails)
    "received-spf", "received_spf", "authentication-results",
    "arc-seal", "arc-message-signature", "arc-authentication-results",
    "x-",  # X- headers are common in real emails
})

# ── Forwarded-message / quoted-reply markers ────────────────────────────────────

FORWARD_MARKER_PATTERNS = [
    re.compile(r"-{5,}\s*[Ff]orwarded?\s*[Mm]essage\s*-{5,}"),
    re.compile(r"={3,}\s*[Ff]orwarded?\s*[Mm]essage\s*={3,}"),
    re.compile(r"(?i)begin\s+forwarded\s+message\s*:"),
]

QUOTE_MARKER_PATTERNS = [
    re.compile(r"-{5,}\s*[Oo]riginal\s+[Mm]essage\s*-{5,}"),
    re.compile(r"={3,}\s*[Oo]riginal\s+[Mm]essage\s*={3,}"),
    re.compile(r"(?m)^On\s+.*?\s+wrote:\s*$"),
    re.compile(r"(?m)^From:\s+.+\n.*Date:\s+.+\n.*Subject:\s+.+(\n.*To:\s+.+)?", re.DOTALL),
]

_ENVELOPE_RE = re.compile(
    r"^From:\s*(.+?)\nDate:\s*(.+?)\n(?:Subject:\s*(.+?)\n)?To:\s*(.+?)$",
    re.MULTILINE | re.DOTALL,
)


# ── Data structures ─────────────────────────────────────────────────────────────

@dataclass
class AttachmentInfo:
    """Metadata for a single MIME attachment (content never embedded)."""
    filename: str
    mime_type: str
    size: int
    content_id: str
    disposition: str


@dataclass
class ParsedEmail:
    """Uniform representation of a parsed email."""
    from_header: str = ""
    to_header: str = ""
    cc_header: str = ""
    subject: str = ""
    date: str = ""
    message_id: str = ""
    body: str = ""
    html_body: str = ""
    raw_body: str = ""
    attachments: list[AttachmentInfo] = field(default_factory=list)
    extra_headers: dict[str, str] = field(default_factory=dict)


# ── HTML-to-text ────────────────────────────────────────────────────────────────

def html_to_text(html_text: str) -> str:
    """Convert HTML to plain text.

    Preserves anchor link text as ``text (URL)`` so LinkedIn/GitHub links
    survive in plaintext output.  Uses ``beautifulsoup4`` when available,
    otherwise falls back to a regex-based stripper.
    """
    if not html_text:
        return ""
    if _HAS_BS4:
        return _html_to_text_bs4(html_text)
    return _html_to_text_regex(html_text)


def _html_to_text_regex(html_text: str) -> str:
    """Regex-based HTML-to-text fallback."""
    text = _HTML_COMMENT_RE.sub("", html_text)
    text = _HTML_ANCHOR_RE.sub(r"\2 (\1)", text)
    text = _HTML_SCRIPT_RE.sub("", text)
    text = _HTML_STYLE_RE.sub("", text)
    text = _HTML_BR_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _html.unescape(text)
    text = _HTML_WS_RE.sub(" ", text)
    text = _HTML_BLANK_RE.sub("\n\n", text)
    return text.strip()


def _html_to_text_bs4(html_text: str) -> str:
    """BeautifulSoup-based HTML-to-text (preferred when bs4 is available).

    Preserves anchor links as ``text (URL)``.
    """
    # Rewrite <a href="URL">text</a> → text (URL) before stripping.
    def _rewrite_anchor(match: re.Match) -> str:
        url = match.group(1)
        link_text = match.group(2).strip()
        if not link_text or link_text == url:
            return url
        return f"{link_text} ({url})"

    rewritten = _HTML_ANCHOR_RE.sub(_rewrite_anchor, html_text)
    soup = BeautifulSoup(rewritten, "html.parser")
    # Remove scripts and styles.
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = _html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = _HTML_BLANK_RE.sub("\n\n", text)
    return text.strip()


# ── MIME body extraction ───────────────────────────────────────────────────────

def _extract_body(message) -> tuple[str, str]:
    """Return ``(plain_text, raw_html)`` from a parsed message object.

    Walks the MIME tree **once**, collecting text parts by content-type
    priority (plain > html > other text/*), then returns the best candidate.
    """
    if not message.is_multipart():
        payload = message.get_payload(decode=True)
        if not payload:
            return (str(message.get_payload() or ""), "")
        text = payload.decode(
            message.get_content_charset() or "utf-8", errors="replace"
        )
        if message.get_content_subtype() == "html":
            return (html_to_text(text), text)
        return (text, "")

    best_plain: Optional[str] = None
    best_html: Optional[str] = None
    best_other: Optional[str] = None
    fallback_part = None

    for part in message.walk():
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        if fallback_part is None:
            fallback_part = part
        if part.get_filename():
            continue  # skip attachments
        text = payload.decode(
            part.get_content_charset() or "utf-8", errors="replace"
        )
        ctype = part.get_content_type()
        if ctype == "text/plain" and best_plain is None:
            best_plain = text
        elif ctype == "text/html" and best_html is None:
            best_html = text
        elif ctype.startswith("text/") and best_other is None:
            best_other = text
        if best_plain is not None and best_html is not None:
            break

    if best_plain is not None:
        return (best_plain, best_html or "")
    if best_html is not None:
        return (html_to_text(best_html), best_html)
    if best_other is not None:
        return (best_other, "")

    # Fallback: first part with a payload.
    if fallback_part is not None:
        payload = fallback_part.get_payload(decode=True)
        if payload:
            text = payload.decode(
                fallback_part.get_content_charset() or "utf-8",
                errors="replace",
            )
            if fallback_part.get_content_subtype() == "html":
                return (html_to_text(text), text)
            return (text, "")
    return ("", "")


def _extract_attachments(message) -> list[AttachmentInfo]:
    """Return metadata for each MIME attachment part."""
    attachments: list[AttachmentInfo] = []
    for part in message.walk():
        filename = part.get_filename()
        disposition = part.get_content_disposition() or ""
        if not (filename or disposition == "attachment"):
            continue
        payload = part.get_payload(decode=True) or b""
        attachments.append(AttachmentInfo(
            filename=filename or "",
            mime_type=part.get_content_type(),
            size=len(payload),
            content_id=(part.get("Content-ID") or "").strip(),
            disposition=disposition or "inline",
        ))
    return attachments


# ── Public parsing functions ────────────────────────────────────────────────────

def _looks_like_rfc822(raw: str) -> bool:
    """Return True if *raw* starts with a recognisable RFC-822 header line."""
    if not raw:
        return False
    first_line = next((ln for ln in raw.splitlines() if ln.strip()), "")
    if not _RFC822_HEAD_RE.match(first_line):
        return False
    # Extract the header name (before the colon) and check against known keywords.
    header_name = first_line.split(":", 1)[0].strip().lower()
    for known in _KNOWN_HEADERS:
        if header_name.startswith(known):
            return True
    return False


def parse_raw_email(raw: str) -> ParsedEmail:
    """Parse an RFC-822 raw email string into a :class:`ParsedEmail`.

    A leading line that looks like a standard email header (``From:``,
    ``Subject:``, etc.) triggers RFC-822 parsing; otherwise the whole string
    is treated as a plain body.  This avoids mistaking body-only text like
    ``"Name: Alice. Role: Engineer"`` for an RFC-822 message.
    """
    if not raw:
        return ParsedEmail(body=raw)
    if not _looks_like_rfc822(raw):
        return ParsedEmail(body=raw)
    message = _email.message_from_string(raw)

    body, html_body = _extract_body(message)
    return ParsedEmail(
        from_header=message.get("From", ""),
        to_header=message.get("To", ""),
        cc_header=message.get("Cc", ""),
        subject=message.get("Subject", ""),
        date=message.get("Date", ""),
        message_id=message.get("Message-ID", ""),
        body=body,
        html_body=html_body,
        raw_body=body,
        attachments=_extract_attachments(message),
        extra_headers={
            h: v for h, v in message.items()
            if h.lower() not in {"from", "subject", "to", "cc", "bcc",
                                 "date", "message-id", "content-type",
                                 "mime-version", "content-transfer-encoding"}
        },
    )


def parse_email_dict(data: dict) -> ParsedEmail:
    """Build a :class:`ParsedEmail` from a mapping with from/subject/body keys."""
    return ParsedEmail(
        from_header=str(data.get("from", "") or ""),
        to_header=str(data.get("to", "") or ""),
        cc_header=str(data.get("cc", "") or ""),
        subject=str(data.get("subject", "") or ""),
        date=str(data.get("date", "") or ""),
        message_id=str(data.get("message_id", "") or data.get("message-id", "") or ""),
        body=str(data.get("body", "") or ""),
        html_body=str(data.get("html_body", "") or data.get("html", "") or ""),
        raw_body=str(data.get("body", "") or ""),
        extra_headers={
            k: v for k, v in data.items()
            if k.lower() not in {"from", "to", "cc", "subject",
                                 "body", "date", "message_id",
                                 "message-id", "html_body", "html"}
        },
    )


def parse_eml_file(filepath: str) -> ParsedEmail:
    """Load and parse an ``.eml`` / ``.emlx`` file from disk."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    return parse_raw_email(raw)


def parse_eml(eml_source: str) -> ParsedEmail:
    """Parse an EML source — a file path or raw RFC-822 text.

    If *eml_source* is an existing file path, the file is read and parsed.
    Otherwise it is treated as raw RFC-822 text (or a plain body string).
    """
    if eml_source and os.path.isfile(eml_source):
        return parse_eml_file(eml_source)
    if eml_source and eml_source.lower().endswith((".eml", ".emlx")) and not os.path.isfile(eml_source):
        raise FileNotFoundError(f"EML file not found: {eml_source}")
    return parse_raw_email(eml_source)


def detect_forwarded(subject: str, body: str) -> bool:
    """Return True if *subject* has a Fwd:/Fw: prefix or *body* contains
    a forwarded-message marker.
    """
    if subject and re.match(r"(?i)^(?:re:|fwd:|fw:)+", subject):
        return True
    for pat in FORWARD_MARKER_PATTERNS:
        if pat.search(body):
            return True
    return False


def extract_forwarded_sender(body: str) -> dict[str, Optional[str]]:
    """Extract the *forwarded* sender from a forwarded-message envelope.

    Looks for ``From: ... Date: ... Subject: ... To: ...`` blocks that appear
    inside the body (after a forward marker).  Returns the sender parsed from
    the forwarded ``From:`` line.
    """
    for pat in FORWARD_MARKER_PATTERNS:
        m = pat.search(body)
        if m:
            envelope_start = m.end()
            envelope_block = body[envelope_start:]
            env = _ENVELOPE_RE.search(envelope_block)
            if env:
                return extract_sender(env.group(1))
    return {"name": None, "email": None}


def strip_forward_envelope(body: str) -> str:
    """Remove the forwarded-message envelope (From/Date/Subject/To lines)
    from *body*, preserving the actual message text that follows.
    """
    for pat in FORWARD_MARKER_PATTERNS:
        m = pat.search(body)
        if m:
            after_marker = body[m.end():]
            # Strip the From/Date/Subject/To envelope block.
            env = _ENVELOPE_RE.search(after_marker)
            if env:
                return after_marker[env.end():].lstrip()
            # Fallback: strip until the first blank line after the marker.
            blank = re.search(r"\n\s*\n", after_marker)
            if blank:
                return after_marker[blank.end():].lstrip()
            return after_marker.lstrip()
    return body


def strip_quoted_reply(body: str) -> str:
    """Cut everything from the first quoted-reply marker onward.

    Quoted replies (``----- Original Message -----``, ``On ... wrote:``,
    embedded ``From:/Date:/Subject:/To:`` envelopes) are trailing context
    that does not belong to the current message.
    """
    best_pos = len(body)
    for pat in QUOTE_MARKER_PATTERNS:
        m = pat.search(body)
        if m and m.start() < best_pos:
            best_pos = m.start()
    if best_pos < len(body):
        return body[:best_pos].rstrip()
    return body


def strip_trailing_signature(body: str) -> str:
    """Strip a trailing email signature block.

    Finds the last standalone ``--`` separator line and removes everything
    from that point onward.  This removes the *forwarder's* signature
    (which appears at the bottom of a forwarded message) without affecting
    the forwarded sender's content above it.
    """
    if not body:
        return body
    lines = body.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "--":
            return "\n".join(lines[:i]).rstrip()
    return body


def extract_sender(from_header: str) -> dict[str, Optional[str]]:
    """Extract sender name and email from a ``From:`` header value.

    Returns ``{"name": str | None, "email": str | None}``.
    """
    if not from_header:
        return {"name": None, "email": None}
    name, addr = email.utils.parseaddr(from_header)
    name = name.strip().strip('"').strip() or None
    email_val = addr.strip() or None
    if not email_val:
        # Fall back to regex if parseaddr didn't find an address.
        m = re.search(
            r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', from_header
        )
        if m:
            email_val = m.group(0)
    return {"name": name, "email": email_val}
