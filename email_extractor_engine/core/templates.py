"""Manual field registry for known email domains.

Every topic in :data:`TOPIC_FIELD_MAP` has:

* ``fields`` — an explicit list of target field names that the engine should
  try to populate.
* ``aliases`` — additional phrases (lower-cased during resolution) that a user
  might type instead of the canonical topic name.

Resolution happens in :func:`resolve_topic` which is called by
:func:`get_fields_for_topic`.  When the user's input matches (exactly or via
token-set containment, see source) a canonical topic, the registry's field list
is returned as the *primary* path.  Topics that do **not** match any entry fall
back to dynamic schema discovery (see :mod:`core.dynamic_schema`).
"""
from __future__ import annotations

import re


def _normalize(text: str) -> str:
    """Normalize a topic/alias for comparison: strip, collapse spaces, lowercase."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _build_alias_index() -> dict[str, list[str]]:
    """Build a lookup: normalized_alias -> list of canonical topics.

    Preserves insertion order of aliases within each topic.
    """
    index: dict[str, list[str]] = {}
    for topic, data in TOPIC_FIELD_MAP.items():
        norm_topic = _normalize(topic)
        aliases: set[str] = {_normalize(a) for a in data["aliases"]}
        aliases.add(norm_topic)
        for alias in aliases:
            index.setdefault(alias, []).append(topic)
    return index


# Registry of pre-defined field lists for common email domains.
# Each entry maps a canonical topic name to its field names and aliases.
TOPIC_FIELD_MAP: dict[str, dict] = {
    "job application": {
        "fields": [
            "candidate_name", "applicant_email", "target_role",
            "years_experience", "education", "expected_salary",
            "notice_period", "skills", "phone_number",
            "current_company", "current_role", "seniority_level",
            "work_type", "start_date", "portfolio_links",
        ],
        "aliases": [
            "job application", "job applications", "job app",
            "career application", "job inquiry", "employment application",
            "position application",
        ],
    },
    "dmarc report": {
        "fields": [
            "target_domain", "reporting_period", "submitter_email",
            "source_ip", "total_messages", "passed_messages",
            "failed_messages", "dmarc_policy", "policy_dkim",
            "policy_spf", "auth_results",
        ],
        "aliases": [
            "dmarc report", "dmarc reports", "dmarc aggregate report",
            "dmarc forensic report", "domain report",
            "email authentication report", "spf/dmarc report",
        ],
    },
    "invoice": {
        "fields": [
            "invoice_number", "invoice_date", "due_date", "total_amount",
            "subtotal", "tax_amount", "tax_rate", "discount_amount",
            "shipping_fee", "vendor_name", "customer_name",
            "customer_email", "payment_method", "payment_status",
            "purchase_order", "vat_number", "billing_address", "line_items",
        ],
        "aliases": [
            "invoice", "invoices", "billing", "bill", "receipt",
            "receipts", "financial report", "financial reports",
            "fee", "charge",
        ],
    },
    "e-commerce order": {
        "fields": [
            "order_number", "order_date", "customer_name", "customer_email",
            "items", "quantities", "prices", "subtotal", "tax_amount",
            "shipping_fee", "discount_amount", "total_amount",
            "payment_method", "payment_status", "shipping_address",
            "billing_address", "shipping_carrier", "tracking_number",
            "estimated_delivery",
        ],
        "aliases": [
            "e-commerce order", "ecommerce order", "order confirmation",
            "order", "purchase confirmation", "purchase",
            "shipping notice", "delivery notice", "shop order",
        ],
    },
    "event invitation": {
        "fields": [
            "event_title", "event_date", "start_time", "end_time",
            "location", "organizer", "organizer_email", "meeting_link",
            "agenda", "rsvp_email",
        ],
        "aliases": [
            "event invitation", "event invite", "meeting invitation",
            "calendar invite", "conference invitation",
            "webinar invitation",
        ],
    },
    "travel itinerary": {
        "fields": [
            "booking_reference", "passenger_name", "flight_number",
            "origin", "destination", "departure_datetime", "arrival_datetime",
            "seat_number", "airline", "booking_status", "ticket_number",
            "travel_class", "departure_terminal", "arrival_terminal",
        ],
        "aliases": [
            "travel itinerary", "travel booking", "flight confirmation",
            "trip confirmation", "itinerary", "travel confirmation",
        ],
    },
    "support ticket": {
        "fields": [
            "ticket_id", "ticket_subject", "ticket_status",
            "ticket_priority", "customer_name", "customer_email",
            "issue_description", "assigned_agent", "created_date",
            "resolved_date", "category",
        ],
        "aliases": [
            "support ticket", "support request", "help desk",
            "incident report", "ticket", "customer support",
        ],
    },
    "meeting minutes": {
        "fields": [
            "meeting_title", "meeting_date", "start_time", "end_time",
            "attendees", "agenda_items", "action_items",
            "decisions_made", "meeting_owner", "location", "recording_link",
        ],
        "aliases": [
            "meeting minutes", "meeting notes", "meeting summary",
            "minutes", "board meeting",
        ],
    },
    "purchase order": {
        "fields": [
            "po_number", "po_date", "buyer_name", "buyer_email",
            "supplier_name", "supplier_email", "total_amount", "currency",
            "line_items", "delivery_date", "payment_terms",
            "shipping_address", "billing_address",
        ],
        "aliases": [
            "purchase order", "po", "purchase order form", "procurement",
        ],
    },
    "delivery notice": {
        "fields": [
            "tracking_number", "carrier", "delivery_status",
            "origin", "destination", "estimated_delivery",
            "actual_delivery", "recipient_name", "parcel_weight",
            "number_of_packages",
        ],
        "aliases": [
            "delivery notice", "shipping notification", "package tracking",
            "delivery update", "shipment status",
        ],
    },
    "interview scheduling": {
        "fields": [
            "candidate_name", "candidate_email", "interviewer_name",
            "interviewer_email", "interview_date", "start_time", "end_time",
            "location", "meeting_link", "job_role", "interview_round",
            "interview_stage", "duration", "status",
        ],
        "aliases": [
            "interview scheduling", "interview invitation",
            "interview request", "interview confirmation",
            "interview",
        ],
    },
    "contract": {
        "fields": [
            "contract_id", "contract_date", "parties", "effective_date",
            "expiration_date", "contract_value", "currency", "contract_type",
            "governing_law", "termination_clause", "signatures",
        ],
        "aliases": [
            "contract", "agreement", "nda", "non-disclosure agreement",
            "service agreement", "employment contract",
        ],
    },
    "newsletter": {
        "fields": [
            "newsletter_title", "newsletter_date", "author_name",
            "author_email", "section_titles", "unsubscribe_link",
            "issue_number",
        ],
        "aliases": [
            "newsletter", "email newsletter", "announcement",
            "weekly digest", "monthly digest", "bulletin",
        ],
    },
}

# Pre-computed at import time – maps every normalized alias (including the
# canonical topic name itself) to the list of canonical topics that share it.
_ALIAS_INDEX: dict[str, list[str]] = _build_alias_index()

# Aliases shorter than this many characters are skipped during token-set
# containment matching (but can still be matched exactly).
_MIN_ALIAS_LEN = 3


def _tokens(text: str) -> set[str]:
    """Split *text* into whitespace-delimited tokens (lowercase, stripped)."""
    return {t for t in text.split() if t}


def resolve_topic(topic: str) -> str | None:
    """Resolve a user-supplied *topic* string to a canonical topic name.

    Resolution tiers (in order):

    1. **Exact normalized match** against the topic keys in
       :data:`TOPIC_FIELD_MAP`.
    2. **Exact alias match** — the normalized topic equals a known alias.
    3. **Token-set containment** — every token of an alias appears in the
       topic tokens (alias ⊆ topic), or every topic token appears in an alias
       (topic ⊆ alias).  Only aliases with ``>= _MIN_ALIAS_LEN`` characters
       participate, so tiny fragments like ``"po"`` don't match unrelated
       topics via raw substring.

    Returns the canonical topic string, or ``None`` if no match is found.
    """
    if not topic:
        return None

    norm = _normalize(topic)
    topic_tokens = _tokens(norm)

    # 1. Exact normalized match against topic keys
    for canonical_topic in TOPIC_FIELD_MAP:
        if _normalize(canonical_topic) == norm:
            return canonical_topic

    # 2. Exact alias match
    if norm in _ALIAS_INDEX:
        return _ALIAS_INDEX[norm][0]

    # 3. Token-set containment (fuzzy matching)
    best: tuple[int, str] | None = None
    for alias, topics in _ALIAS_INDEX.items():
        if len(alias) < _MIN_ALIAS_LEN:
            continue
        alias_tokens = _tokens(alias)
        if not alias_tokens:
            continue

        # Case A: alias tokens are a subset of topic tokens
        #         (alias ⊆ topic — e.g. alias="report" in topic="quarterly report")
        #         Only allow multi-token aliases to match via subset to avoid
        #         single tokens like "report" matching unrelated topics like "dmarc report"
        if len(alias_tokens) > 1 and alias_tokens <= topic_tokens:
            score = len(alias_tokens)
            if best is None or score > best[0]:
                best = (score, topics[0])

        # Case B: single-token alias that exactly equals the topic tokens
        if len(alias_tokens) == 1 and alias_tokens == topic_tokens:
            score = len(alias_tokens)
            if best is None or score > best[0]:
                best = (score, topics[0])

        # Case C: topic tokens are a subset of alias tokens
        #         (topic ⊆ alias — e.g. topic="job app" where alias="job application")
        #         Only consider if alias has multiple tokens AND topic has multiple tokens
        #         (to avoid single tokens like "report" matching "dmarc report")
        if topic_tokens and len(topic_tokens) > 1 and len(alias_tokens) > 1 and topic_tokens <= alias_tokens:
            score = len(topic_tokens)
            if best is None or score > best[0]:
                best = (score, topics[0])

    if best is not None:
        return best[1]

    return None


def get_fields_for_topic(topic: str) -> list[str] | None:
    """Return the list of target field names for *topic*, or ``None``.

    The *topic* is resolved via :func:`resolve_topic`.  A ``None`` return
    signals the caller to fall back to dynamic schema discovery.
    """
    canonical = resolve_topic(topic)
    if canonical is None:
        return None
    entry = TOPIC_FIELD_MAP.get(canonical)
    if entry is None:
        return None
    return list(entry["fields"])
