"""Zero-Shot Field Extraction via GLiNER - LLM-like extraction using semantic field registry.

Wraps a local GLiNER model to extract exact text spans for arbitrary field
labels (e.g. ``"candidate name"``, ``"target domain"``).  GLiNER is a
BERT-based token-classification model — it is **not** an LLM and produces no
LLM-style token generation, keeping runtime token cost at zero.

``gliner`` and ``torch`` are **optional** — when absent, a comprehensive
regex-based fallback extractor uses the semantic field registry to provide
LLM-like extraction for ANY field type (emails, phones, dates, money, IPs,
domains, names, organizations, addresses, roles, skills, and 50+ other types).

The extractor now uses a data-driven approach where field types from the
semantic registry determine which extraction patterns to apply, enabling
accurate extraction for arbitrary topics without manual per-field configuration.
"""
from __future__ import annotations

import re
import logging
from typing import Optional
from dataclasses import dataclass, asdict

from .field_registry import (
    get_field_type,
    get_kv_keys_for_field,
    get_regex_pattern_for_field,
    get_gliner_label,
    FIELD_TYPE_PATTERNS,
)

logger = logging.getLogger(__name__)


@dataclass
class Extraction:
    """Provenance metadata for a single extracted field value.

    Every extracted field carries its raw value alongside a confidence
    score and the extraction method that produced it, so downstream
    consumers can filter or weight by reliability — matching the
    transparency LLMs provide.
    """
    value: Optional[str]
    confidence: float           # 0.0–1.0, model score or regex heuristic
    method: str                 # e.g. "gliner", "regex.email", "regex.signoff"
    raw: Optional[str] = None   # the original span before cleaning (for audit)

    def to_dict(self) -> dict:
        return asdict(self)

# ── Optional GLiNER import (lazy, auto-detected) ────────────────────────────────
try:
    from gliner import GLiNER
    _HAS_GLINER = True
except ImportError:
    _HAS_GLINER = False

# Lazy-loaded model cache.
_GLINER_MODEL: Optional[object] = None
_GLINER_MODEL_NAME = "gliner-community/gliner_medium-v2.1"

# ── Confidence thresholds for GLiNER predictions ───────────────────────────────
# Two-tier system (the key to LLM-like quality without LLMs):
#   1. GLINER_THRESHOLD  — extraction threshold (0.25): GLiNER runs at wide
#      recall so it doesn't miss weak-but-real matches ("four years of experience").
#   2. ACCEPT_THRESHOLD   — acceptance threshold (0.40): only results above THIS
#      survive; lower-confidence picks are rejected and handed to the regex
#      fallback, which applies stricter structural validation.
#   3. Validation filters — reject matches that match known false-positive patterns
#      (e.g. "Dear Recruiter" as a company name).
GLINER_THRESHOLD: float = 0.20          # extraction recall threshold
GLINER_ACCEPT_THRESHOLD: float = 0.30   # acceptance precision threshold

# Words/phrases that should NEVER be accepted as entity extractions — they are
# salutations, greetings, or common function words that GLiNER confuses with
# actual entities (PERSON, ORG, GPE, etc.) at low confidence.
_GREETING_OR_SALUTATION_WORDS: frozenset[str] = frozenset({
    "dear", "hello", "hi", "hey", "greetings", "regards", "best",
    "sincerely", "cheers", "thanks", "thank you", "kind regards",
    "warm regards", "many thanks", "yours truly", "yours faithfully",
    "respected", "esteemed", "ladies", "gentlemen", "friends",
    # Common greeting titles that aren't real company/person names
    "recruiter", "sir", "madam", "team", "department", "office",
    "manager", "director", "supervisor", "colleague", "associate",
})


def _is_implausible_entity(label: str, value: str, score: float) -> bool:
    """Reject GLiNER results that are likely false positives.

    Returns ``True`` when the result should be discarded so the regex
    fallback can take over (or the field stays ``None``).
    """
    value_lower = value.strip().lower()
    if not value_lower:
        return True
    # Confidence is below the acceptance threshold → reject.
    if score < GLINER_ACCEPT_THRESHOLD:
        return True
    # Salutation/greeting words detected as entities → reject.
    label_lower = label.lower()
    if any(token in value_lower for token in _GREETING_OR_SALUTATION_WORDS):
        # Allow "Dear" in "Dear Sir" only when label is literally "greeting"
        # (not for person/org/company labels).
        if any(kw in label_lower for kw in ("company", "person", "organization", "name")):
            return True
    return False


# ── Pre-compiled regex patterns (fallback extraction) ─────────────────────────
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)
_PHONE_RE = re.compile(
    r"(?<![\w])"                          # not preceded by word char
    r"(?:"                                  # country code group (optional)
    r"\+[1-9]\d{0,2}[-.\s]?"               # +1, +91, +44 etc.
    r"|(?:\(\+\d+\)\s*)?"                  # (+91) style
    r"(?:00|\-\d+\s)"                      # 00xx or --xx style
    r")?"                                   # end country code group
    r"(?:"                                  # national number
    r"\(?\d{3}\)?[-.\s]?"                  # area code: (123) or 123
    r"\d{3}[-.\s]?"                         # exchange: 123
    r"\d{4}"                                # subscriber: 1234
    r"|[1-9]\d{8,14}"                       # compact 9-15 digit international numbers
    r")"
    r"(?![\w])",                            # not followed by word char
)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CRYPTO_ADDR_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
_DURATION_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?|yrs?)")
_DOMAIN_RE = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2,}\b"
)
_MONEY_RE = re.compile(r"[\$\£\€]\s?\d[\d,]*\.?\d*(?:\s*[kKmMbBtT])?")
_SSN_DATE_RE = re.compile(r"\d{3}-\d{2}-\d{4}")
_DATE_RE = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}"
)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b")

# Key-value pair: "Key: value" or "Key = value" (case-insensitive key).
# Value extends to the next sentence boundary or end.
# Sentence boundary: period followed by whitespace + capital letter/digit,
# or period at end of text, or newline + capital letter.
_SENT_BOUNDARY = r"(?=\s*\.\s+[A-Z\d]|\s*\.\s*$|\s*\n\s*[A-Z\d]|\s*$|\s*\.\s*\n)"

_KV_LINE_RE = re.compile(
    r"(?im)^(?P<key>[A-Za-z][A-Za-z0-9 _\-&./]{1,50}?)\s*[:=]\s*(?P<val>.+?)\s*$"
)
# Inline KV: key anywhere in text, value until sentence boundary.
_KV_INLINE_RE = re.compile(
    r"(?i)\b(?P<key>[A-Z][A-Za-z0-9][A-Za-z0-9 _\-&]{0,40}?)\s*[:=]\s*"
    r"(?P<val>.+?)" + _SENT_BOUNDARY
)

# Generic/common words that are poor discriminators in multi-word field names.
# When a multi-part field name contains both a generic and a specific part
# (e.g. ``policy_dkim`` → generic "policy", specific "dkim"), the specific
# part is tried first so it wins over KV pairs that belong to a *different*
# field (e.g. ``dmarc_policy`` → ``Policy: reject``).
_GENERIC_PARTS: frozenset[str] = frozenset({
    "policy", "total", "count", "number", "messages", "status",
    "type", "value", "name", "date", "time", "address", "location",
    "amount", "price", "cost", "role", "position", "data", "info",
    "period", "rate", "method", "source", "result",
})

# Type-specific patterns for fallback extraction.
_EXPERIENCE_RE = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)\b"
)

# Word-form numbers (e.g. "four years of experience").
_WORD_TO_DIGIT: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100",
}
_WORD_NUM_EXPERIENCE_RE = re.compile(
    r"(?i)\b("
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred"
    r")\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)\b"
)
_SALARY_RE = re.compile(
    r"(?i)(?:salary|expected\s*salary|compensation|rate|per\s*-\s*year|annually)"
    r"[^$0-9]{0,30}(\$[\d,]+\.?\d*(?:\s*[kK]|thousand)?)"
)
_EDUCATION_RE = re.compile(
    r"(?i)\b(B\.S\.?|B\.A\.?|M\.S\.?|M\.A\.?|M\.B\.A\.?|Ph\.D\.?|M\.Tech|B\.Tech|A\.B\.?)"
    r"[^\n.]+?(?=\s*\.\s|\s*\.\s*$|\n|$)"
)

# ── Additional pattern families for broad topic coverage ────────────────────────
# These let the rule-based extractor catch information that appears in natural
# prose (not just structured KV pairs), giving near-LLM-level coverage without
# any external model downloads.

# Portfolio / link extraction – catches URLs anywhere in text
_LINKEDIN_RE = re.compile(
    r"https?://(?:www\.)?linkedin\.com/[^\s,)<>]+"
)
_GITHUB_RE = re.compile(
    r"https?://(?:www\.)?github\.com/[^\s,)<>]+"
)
_URL_RE = re.compile(
    r"https?://[^\s,)>]+\.(?:com|org|net|io|dev|co|me|in)(?:[^\s,)<>]*)?"
)

# Location detection – country / city phrases near location keywords
_LOCATION_PATTERNS: list[tuple[str, str]] = [
    # "based in X", "located in X", "from X" followed by capitalized words
    (r"(?i)(?:based\s+in|located\s+in|from)\s+([A-Z][A-Za-z]+(?:[\s,\-]+[A-Z][A-Za-z]+){0,3})", None),
]
_COUNTRIES: frozenset[str] = frozenset({
    "United States", "Canada", "United Kingdom", "Germany", "France", "India",
    "Australia", "Japan", "China", "Brazil", "Netherlands", "Singapore",
    "Sweden", "Norway", "Denmark", "Finland", "Switzerland", "Austria",
    "Belgium", "Spain", "Italy", "Portugal", "Ireland", "Poland",
    "Russia", "Ukraine", "Mexico", "Argentina", "South Africa",
    "New Zealand", "Israel", "Saudi Arabia", "Uae", "Dubai",
})

# Seniority level detection
_SENIORITY_LEVELS: frozenset[str] = frozenset({
    "entry", "entry level", "junior", "associate",
    "mid", "mid level", "intermediate",
    "senior", "lead", "principal", "staff", "architect",
})

# Notice period / availability detection
_NOTICE_PERIOD_RE = re.compile(
    r"(?i)(?:notice\s+(?:period\s*)?|available(?:ity)?(?:\s+(?:in|within))?)[^0-9]{0,20}"
    r"(\d+)\s*(weeks?|days?|months?)"
)
_IMMEDIATE_AVAILABILITY_RE = re.compile(
    r"(?i)(?:immediately|asap|as soon as possible|currenty available|can start)",
)

# Work type detection (remote / hybrid / onsite)
_WORK_TYPE_MAP: dict[str, str] = {
    "fully remote": "remote", "work-from-home": "remote", "wfh": "remote",
    "remote": "remote", "hybrid": "hybrid", "flexible": "hybrid",
    "onsite": "onsite", "on-site": "onsite", "in-office": "onsite",
    "office": "onsite",
}

# Sign-off phrases followed by a name on the next line(s).
# Captures the first non-empty line after the sign-off phrase.
_SIGNOFF_NAME_RE = re.compile(
    r"(?im)^[ \t]*("  # line start, optional indent
    r"regards?|best[ \t]*(?:regards)?|sincerely|thanks?|thank[ \t]+you|"
    r"cheers|warm[ \t]*(?:regards)?|many[ \t]+thanks|gratefully"
    ")[,. \t]*\n[ \t]*([^\n]+)"
)

# Role extraction from prose like "application for the Data Analyst position".
_ROLE_PROSE_RE = re.compile(
    r"(?i)(?:applying|application)\s+(?:for|to)\s+(?:the\s+)?"
    r"([A-Z][A-Za-z][A-Za-z\s\-]*?)\s+(?:position|role|job)"
)

# Comprehensive skills extraction — catches skills in explicit lists,
# prose mentions, and implicit "designing/building X" contexts.
_SKILLS_RE = re.compile(
    r"(?i)(?:experience with|with experience in|skills include|proficient in|"
    r"knowledge of|skilled in|background in|skilled at|expert in|proficient with|"
    r"strong in|experienced in|specializing in|specialise in|specializing on|"
    r"familiar with|experienced with)\s+"
    r"([A-Za-z0-9][^.\n]+(?:\n[^.\n]+)*)(?=\.\s|\n\n|$)"
)

# Implicit skills: "designing mobile interfaces", "building React apps",
# "developing machine learning models", "working with Python and PostgreSQL".
# The lookahead stops at sentence boundaries or ", and a/an/the X" (new noun
# phrase), but does NOT stop at bare "and" so multi-word skill phrases like
# "user-centered mobile and web interfaces" are captured in full.
_IMPLICIT_SKILLS_RE = re.compile(
    r"(?i)\b(?:designing|developing|building|working with|working on|"
    r"specializing in|expertise in|skilled at|strong at|proficient at|"
    r"background in|specialized in|experienced with|experienced in)\s+"
    r"([a-zA-Z][a-z][A-Za-z0-9\s\-+./]+)"
    r"(?=\s*(?:\.\s+|\.\s*$|\n\n|$|, and\s+(?:a|an|the)\b))"
)

# General number extraction (for count/quantity fields).
_NUMBER_RE = re.compile(r"\b(\d+(?:[,.]\d+)?)\b")


# ── Dynamic field-name → GLiNER label ────────────────────────────────────────────
# Labels are generated from field names using the semantic field registry.
# This keeps the system fully dynamic — no manual per-field label map.


def field_to_label(field_name: str) -> str:
    """Convert a snake_case field name to a human-readable GLiNER label.
    
    Uses the semantic field registry for consistent label generation.
    """
    return get_gliner_label(field_name)


def _load_gliner_model() -> Optional[object]:
    """Lazy-load the GLiNER model (once, cached globally).

    Fails fast if HF Hub download times out (>5s), falling back to regex-only mode.
    """
    global _GLINER_MODEL
    if isinstance(_GLINER_MODEL, list):
        return _GLINER_MODEL[0]
    if _GLINER_MODEL is not None:
        return _GLINER_MODEL
    if not _HAS_GLINER:
        return None
    try:
        import threading
        import time

        # Load model on a background thread with a timeout
        loaded: list = []
        def _do_load():
            try:
                _GLINER_MODEL[0] = GLiNER.from_pretrained(_GLINER_MODEL_NAME)
                loaded.append(True)
            except Exception:  # noqa: BLE001
                loaded.append(False)

        lock = threading.Lock()
        _GLINER_MODEL = [None]  # mutable container
        t = threading.Thread(target=_do_load, daemon=True)
        t.start()
        t.join(timeout=8)
        if t.is_alive():
            raise TimeoutError("HF Hub download timed out; skipping GLiNER")
        return _GLINER_MODEL[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load GLiNER model '%s': %s", _GLINER_MODEL_NAME, exc)
        _GLINER_MODEL = None
        return None


# ── GLiNER extraction ────────────────────────────────────────────────────────────

def extract_with_gliner(
    text: str, field_names: list[str], threshold: float = GLINER_THRESHOLD
) -> dict[str, Extraction]:
    """Extract values for *field_names* from *text* using GLiNER.

    Returns a dict mapping each field name to an :class:`Extraction` with
    provenance metadata (value, confidence, method, raw).  When multiple
    entities are predicted for the same label, the one with the highest
    confidence score wins.
    """
    _NONE = Extraction(value=None, confidence=0.0, method="none", raw=None)

    if not text or not field_names:
        return {fn: _NONE for fn in field_names}

    model = _load_gliner_model()
    if model is None:
        return _extract_with_regex(text, field_names)

    labels = [field_to_label(fn) for fn in field_names]
    # De-duplicate labels while preserving order.
    seen_labels: set[str] = set()
    unique_labels: list[str] = []
    for label in labels:
        if label not in seen_labels:
            seen_labels.add(label)
            unique_labels.append(label)

    try:
        entities = model.predict_entities(text, unique_labels, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GLiNER prediction failed, falling back to regex: %s", exc)
        return _extract_with_regex(text, field_names)

    # Group results by label, keeping the highest-confidence span per label.
    best_by_label: dict[str, tuple[str, float]] = {}
    for ent in entities:
        label = ent.get("label", "")
        text_val = ent.get("text", "").strip()
        score = ent.get("score", 0.0)
        if not text_val:
            continue
        if label not in best_by_label or score > best_by_label[label][1]:
            best_by_label[label] = (text_val, score)

    # Map labels back to field names.  Validate each result with the
    # plausibility filter: if GLiNER returned a low-confidence or implausible
    # match (e.g. "Recruiter" as a company name), reject it so the regex
    # fallback handles that field instead.
    gliner_result: dict[str, Extraction] = {}
    for fn, label in zip(field_names, labels):
        if label in best_by_label:
            value, score = best_by_label[label]
            if not _is_implausible_entity(label, value, score):
                gliner_result[fn] = Extraction(
                    value=_clean_value(value), confidence=score,
                    method="gliner", raw=value,
                )
            else:
                # Rejected: let regex fallback validate/re-extract.
                gliner_result[fn] = _NONE
        else:
            gliner_result[fn] = _NONE

    # ── Merge with regex fallback ────────────────────────────────────────────
    # Only run regex for fields GLiNER missed (avoid redundant work).
    missing = [fn for fn in field_names if not gliner_result[fn].value]
    if missing:
        regex_data = _extract_with_regex(text, missing)
        for fn in missing:
            gliner_result[fn] = regex_data[fn]

    # ── Extend skills fields with regex for completeness ───────────────────────
    # If GLiNER returned a partial skills value (e.g. "user-centered mobile"),
    # the regex fallback may produce a more complete match ("...and web interfaces").
    # Prefer the longer, more descriptive result.
    skills_fields = [fn for fn in field_names if "skill" in fn.lower()]
    if skills_fields:
        regex_data = _extract_with_regex(text, skills_fields)
        for fn in skills_fields:
            gliner_val = gliner_result.get(fn)
            regex_val = regex_data.get(fn)
            if regex_val and regex_val.value and (
                not gliner_val or not gliner_val.value
                or len(regex_val.value) > len(gliner_val.value)
            ):
                gliner_result[fn] = regex_val

    return gliner_result


# ── Regex fallback extraction ────────────────────────────────────────────────────

# Known abbreviations that end with a period and should not be treated as
# sentence boundaries when they appear at the start of a KV value.
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "vs", "etc", "inc", "ltd", "co", "corp",
    "jr", "sr", "prof", "rev", "fr", "st", "dept", "univ", "inst",
    "ltd", "llc", "plc", "gmbh", "ag", "sa", "nv", "ab",
})


def _clean_kv_value(key: str, value: str, body: str) -> Optional[str]:
    """Post-process a KV-extracted value, fixing abbreviation truncation.

    When the inline/line-anchored KV pattern captures a value that ends with
    a known abbreviation (e.g., "Mr" instead of "Mr. John Doe"), this function
    re-extracts the full value using a greedy capture and strips trailing KV
    pairs.

    Also handles the case where the value is truncated *after* an abbreviation
    (e.g., "St" instead of "St. Mary's Cathedral") by using a greedy capture.
    """
    if not value or len(value) <= 1:
        return value

    val_lower = value.lower().strip()

    # Check if the value IS a known abbreviation (e.g., "Dr" from "Dr. Sarah Lee",
    # or "St" from "St. Mary's Cathedral").
    # The sentence boundary fires after the period, truncating the value.
    # Use a greedy capture to get the full value.
    if val_lower in _ABBREVIATIONS:
        greedy_pattern = re.compile(
            r"(?i)\b" + re.escape(key) + r"\b\s*[:=]\s*(.+)"
        )
        greedy_match = greedy_pattern.search(body)
        if greedy_match:
            value = greedy_match.group(1).strip()
            # Strip trailing KV pairs (e.g., "Dr. Sarah Lee. Reason: annual checkup")
            # A KV pair starts with: period + space + Capitalized word(s) + colon/equals
            value = re.sub(r"\s*\.\s+[A-Z][A-Za-z\s]+\s*[:=#].*$", "", value)
            # Also strip trailing sentence fragments after a period
            # (e.g., "St. Mary's Cathedral. Reception to follow")
            # Only strip if the value contains an apostrophe (indicating a name
            # like "Mary's" that should be preserved) and the trailing sentence
            # doesn't contain an apostrophe.
            if "'" in value:
                value = re.sub(r"\s*\.\s+([A-Z][A-Za-z\s]+)\s*$", "", value)
            # Strip trailing sentence-ending punctuation
            value = re.sub(r"\s*\.\s*$", "", value)
        return value

    # Check if the value is very short — likely truncated by a sentence boundary
    # after an abbreviation (e.g., "St" from "St. Mary's Cathedral").
    if len(value) <= 3:
        greedy_pattern = re.compile(
            r"(?i)\b" + re.escape(key) + r"\b\s*[:=]\s*(.+)"
        )
        greedy_match = greedy_pattern.search(body)
        if greedy_match:
            full_value = greedy_match.group(1).strip()
            # Strip trailing KV pairs and sentence-ending punctuation.
            full_value = re.sub(
                r"\s*\.\s+[A-Z][A-Za-z\s]+\s*[:=#].*$", "", full_value
            )
            # Also strip trailing sentence fragments after a period
            # (e.g., "St. Mary's Cathedral. Reception to follow")
            # Only strip if the trailing sentence has no apostrophes (to preserve
            # names like "Mary's" or "O'Brien").
            full_value = re.sub(r"\s*\.\s+([A-Z][A-Za-z\s]+)\s*$", "", full_value)
            full_value = re.sub(r"\s*\.\s*$", "", full_value)
            if len(full_value) > len(value):
                return full_value

    return value


def _clean_value(value: str) -> Optional[str]:
    """Normalise and clean an extracted value string.

    - Strips trailing parentheses containing emails (e.g. ``John (a@b.com)``).
    - Strips trailing punctuation.
    - Returns ``None`` for empty results.
    """
    if value is None:
        return None
    value = value.strip()
    # Strip markdown wrapping (*Name* / _Name_).
    value = re.sub(r"^[*_]+([^*_]+)[*_]+$", r"\1", value).strip()
    # Strip trailing parenthetical content (often an email/note).
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
    # Strip trailing punctuation that's not part of the value.
    value = value.rstrip(".,;:'\"").strip()
    return value if value else None


def _extract_signoff_name(body: str) -> Optional[str]:
    """Extract a person's name from an email sign-off (e.g. ``Regards,\\nJohn Doe``).

    Collects all matches across sign-off phrases and prefers the *fullest*
    name (most tokens, tie-broken by length) so a stray single token
    ("Regards") doesn't win over a two-token name ("Jane Doe").
    Tolerates markdown wrapping (``*Name*`` / ``_Name_``).
    """
    matches: list[str] = []
    for m in _SIGNOFF_NAME_RE.finditer(body):
        name = m.group(2).strip()
        # Strip markdown wrapping (*Name* / _Name_).
        name = re.sub(r"^[*_]+([^*_\n]+)[*_]+$", r"\1", name).strip()
        name = re.sub(r"^[*_]+|[*_]+$", "", name).strip()
        # Validate: name should be 1-4 capitalized words, no colons.
        words = [w for w in name.split() if w]
        if (1 <= len(words) <= 4 and all(w[0].isupper() for w in words)
                and ":" not in name):
            matches.append(name)
    if not matches:
        return None
    # Prefer fullest name: most tokens, then longest string.
    return max(matches, key=lambda n: (len(n.split()), len(n)))


def _extract_with_regex(
    text: str, field_names: list[str]
) -> dict[str, Extraction]:
    """Regex-based fallback extractor for when GLiNER is unavailable.

    Uses a cascading strategy with provenance tracking:
    1. Key-value pair extraction (line-anchored and inline) — method ``"kv"``,
       confidence based on KV pattern specificity.
    2. Type-specific regex patterns (email, phone, date, money, IP, domain) —
       method ``"regex.<type>"``, confidence 0.85.
    3. Keyword-context patterns (experience, salary, education) —
       method ``"regex.<type>"``, confidence 0.80.
    4. Generic fuzzy token matching — method ``"regex.generic"``,
       confidence based on token overlap.
    """
    _NONE = Extraction(value=None, confidence=0.0, method="none", raw=None)
    result: dict[str, Extraction] = {}
    for fn in field_names:
        raw_val: Optional[str] = _extract_from_keyvalue(fn, text)
        if raw_val is not None:
            cleaned = _clean_value(raw_val)
            result[fn] = Extraction(
                value=cleaned, confidence=0.80, method="kv", raw=raw_val,
            )
            continue
        by_type_val, by_type_method = _extract_by_type(fn, text)
        if by_type_val is not None:
            cleaned = _clean_value(by_type_val)
            result[fn] = Extraction(
                value=cleaned, confidence=0.85, method=by_type_method,
                raw=by_type_val,
            )
            continue
        generic_val = _extract_generic_value(fn, text)
        if generic_val is not None:
            cleaned = _clean_value(generic_val)
            result[fn] = Extraction(
                value=cleaned, confidence=0.60, method="regex.generic",
                raw=generic_val,
            )
            continue
        result[fn] = _NONE
    return result


def _field_to_search_keys(field_name: str) -> list[str]:
    """Generate possible human-readable key variations for KV extraction.
    
    Uses the semantic field registry for comprehensive KV key generation.
    E.g. ``candidate_name`` → ["candidate name", "Candidate Name", "Name"]
    """
    # Get KV keys from the semantic field registry
    keys = get_kv_keys_for_field(field_name)
    
    # Add compound key (full field name with spaces)
    parts = field_name.split("_")
    if len(parts) > 1:
        keys.append(" ".join(parts))
        # Try specific parts first (least generic)
        specific = [p for p in parts if p.lower() not in _GENERIC_PARTS]
        for part in reversed(specific):
            keys.append(part)
    
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for k in keys:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            unique.append(k)
    return unique


def _extract_from_keyvalue(field_name: str, body: str) -> Optional[str]:
    """Extract a value using key-value pair patterns in the body.

    Tries both line-anchored (``Key: value`` on its own line) and inline
    (``Key: value`` embedded in a sentence) patterns.  Single-word keys
    also match as prefixes (e.g. ``Total`` matches ``Total messages:``).
    """
    search_keys = _field_to_search_keys(field_name)

    for key in search_keys:
        key_lower = key.lower()
        is_single_word = " " not in key_lower

        # 1. Line-anchored (multi-line emails): "^Key: value$" or "^Key = value"
        #    Requires at least one separator character to avoid matching
        #    sentence-initial words (e.g. "DMARC Aggregate Report" for key "dmarc").
        #    Uses _SENT_BOUNDARY to limit value capture on single-line bodies.
        line_pattern = re.compile(
            r"(?im)^" + re.escape(key) + r"\b\s*[:=]\s*(.+?)" + _SENT_BOUNDARY
        )
        match = line_pattern.search(body)
        if match:
            value = match.group(1).strip()
            if value and len(value) > 1:
                value = _clean_kv_value(key, value, body)
                if value:
                    return value

        # 2. Inline KV: "key: value" anywhere, value to sentence boundary.
        if is_single_word:
            # Word-boundary match plus common inflection suffixes (s, es,
            # 's, ed, ing, ion) so that "fail" matches "Failures" / "failure"
            # but NOT "Reviewer" (a different word).  This avoids the false
            # positive where a 4-char stem like "rev" matches "Reviewer" via
            # rev\w* — "review" + "er" is a distinct word from "review".
            # No extra words allowed between key and separator — "Proposal
            # Date:" should NOT match key "proposal" and capture "2024-09-01".
            suffix_alt = r"(?:s|es|'s|ed|ing|ion|ions)"
            inline_pattern = re.compile(
                r"(?i)\b" + re.escape(key) + r"(?:" + suffix_alt + r")?\b\s*[:=]\s*"
                r"(.+?)" + _SENT_BOUNDARY
            )
        else:
            inline_pattern = re.compile(
                r"(?i)\b" + re.escape(key) + r"\b\s*[:=]\s*"
                r"(.+?)" + _SENT_BOUNDARY
            )
        match = inline_pattern.search(body)
        if match:
            value = match.group(1).strip()
            if value and len(value) > 1:
                value = _clean_kv_value(key, value, body)
                if value:
                    return value

    return None


def _extract_by_type(field_name: str, body: str) -> tuple[Optional[str], str]:
    """Extract a value based on the field's semantic type from the field registry.

    Returns a ``(value, method)`` tuple where *method* describes which
    extraction pattern produced the value (e.g. ``"regex.email"``).
    This is the core LLM-like extraction — uses the field registry to
    determine the field type, then applies the appropriate patterns.
    """
    field_type = get_field_type(field_name)
    fl = field_name.lower()

    # Get the regex pattern for this field type from the registry
    pattern_str = get_regex_pattern_for_field(field_name)

    # Type-specific extraction using registry patterns
    if field_type == "email":
        m = _EMAIL_RE.search(body)
        if m:
            return m.group(0), "regex.email"

    elif field_type == "phone":
        m = _PHONE_RE.search(body)
        if m:
            return m.group(0).strip(), "regex.phone"

    elif field_type == "ip":
        m = _IP_RE.search(body)
        if m:
            return m.group(0).strip(), "regex.ip"

    elif field_type == "domain":
        # Prefer domain that appears after "for" (e.g., "Report for enterprise.com")
        m = re.search(r"(?i)for\s+([a-zA-Z0-9][a-zA-Z0-9\-]*\.[a-zA-Z]{2,})", body)
        if not m:
            m = _DOMAIN_RE.search(body)
        if m:
            return (m.group(1) if m.lastindex and m.group(1) else m.group(0)), "regex.domain"

    elif field_type == "url":
        # LinkedIn
        m = _LINKEDIN_RE.search(body)
        if m:
            return m.group(0), "regex.url.linkedin"
        # GitHub
        m = _GITHUB_RE.search(body)
        if m:
            return m.group(0), "regex.url.github"
        # Generic URLs
        m = _URL_RE.search(body)
        if m:
            return m.group(0), "regex.url.generic"
    
    elif field_type == "date":
        # Avoid matching SSNs (123-45-6789) as dates
        for m in _DATE_RE.finditer(body):
            candidate = m.group(0).strip()
            if _SSN_DATE_RE.fullmatch(candidate):
                continue
            ssn_match = _SSN_DATE_RE.search(body)
            if ssn_match and candidate in ssn_match.group(0):
                continue
            return candidate, "regex.date"

    elif field_type == "time":
        m = _TIME_RE.search(body)
        if m:
            return m.group(0).strip(), "regex.time"

    elif field_type == "datetime":
        # Try to find date + time combination
        m = re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}[T\s]\d{1,2}:\d{2}", body)
        if m:
            return m.group(0), "regex.datetime"
        # Fallback: find date and time separately
        date_match = _DATE_RE.search(body)
        time_match = _TIME_RE.search(body)
        if date_match and time_match:
            return f"{date_match.group(0)} {time_match.group(0)}", "regex.datetime"
        if date_match:
            return date_match.group(0), "regex.datetime"

    elif field_type == "money":
        m = _MONEY_RE.search(body)
        if m:
            return m.group(0), "regex.money"
    
    elif field_type == "percentage":
        m = re.search(r"\b\d+(?:\.\d+)?\s*%", body)
        if m:
            return m.group(0), "regex.percentage"

    elif field_type == "integer":
        # For count/quantity fields, find numbers with context
        m = re.search(r"(?i)\b(\d+(?:,\d{3})*(?:\.\d+)?)\b", body)
        if m:
            return m.group(1), "regex.integer"

    elif field_type == "person":
        # Try sign-off name extraction (e.g. "Regards,\nJohn Doe")
        m = _extract_signoff_name(body)
        if m:
            return _clean_value(m), "regex.signoff"
        # Try to find capitalized names near relevant keywords
        if any(kw in fl for kw in ["candidate", "applicant", "employee", "contact", "author", "sender", "recipient", "interviewer", "reviewer", "assessor", "auditor", "inspector"]):
            # Look for "Name: value" patterns first via KV extraction
            pass  # KV extraction already handles this
    
    elif field_type == "organization":
        # Look for company/organization names near relevant keywords.
        # Requires contextual evidence — a bare Capitalized phrase like
        # "Dear Recruiter" is NOT a company name.
        if any(kw in fl for kw in ["company", "employer", "organization",
                                   "vendor", "supplier", "client",
                                   "customer", "buyer", "seller", "partner"]):
            # Pattern 1: "currently at Company" / "presently with Company"
            m = re.search(
                r"(?i)(?:current(?:ly)?|present(?:ly)?)\s+(?:at|with)\s+"
                r"([A-Z][A-Za-z0-9&.'\-]+(?:\s+[A-Z][A-Za-z0-9&.'\-]+){0,3})",
                body,
            )
            if m and not _is_implausible_entity(field_name, m.group(1), 1.0):
                return m.group(1).strip().rstrip(".,;"), "regex.organization"

            # Pattern 2: "Company: X" / "Employer: X" (KV-style)
            m = re.search(
                r"(?i)\b(?:company|employer|organization|org)\s*[:=]\s*"
                r"([A-Z][A-Za-z0-9&.'\-]+(?:\s+[A-Z][A-Za-z0-9&.'\-]+){0,3})",
                body,
            )
            if m and not _is_implausible_entity(field_name, m.group(1), 1.0):
                return m.group(1).strip().rstrip(".,;"), "regex.organization"

            # Pattern 3: "works at X" / "employed by X" / "from X"
            m = re.search(
                r"(?i)(?:works?\s+at|employed\s+by|from)\s+"
                r"([A-Z][A-Za-z0-9&.'\-]+(?:\s+[A-Z][A-Za-z0-9&.'\-]+){0,3})",
                body,
            )
            if m and not _is_implausible_entity(field_name, m.group(1), 1.0):
                return m.group(1).strip().rstrip(".,;"), "regex.organization"
        # Explicit return: for organization fields, only accept matches from
        # the targeted context patterns above. The generic Capitalized-words
        # regex is too noisy (matches "Dear Recruiter") and must NOT run here.
        return None, "none"

    elif field_type == "address":
        # Address patterns
        m = re.search(r"\d+\s+[A-Z][A-Za-z\s]+(?:,\s*[A-Z][A-Za-z\s]+){1,3}", body)
        if m:
            return m.group(0).strip(), "regex.address"

    elif field_type == "role":
        # Try prose patterns first (e.g. "application for the Data Analyst position")
        m = _ROLE_PROSE_RE.search(body)
        if m:
            return _clean_value(m.group(1)), "regex.role"
        m = re.search(r"(?i)applying\s+(?:for|to)\s+(?:the\s+)?(.+?)\s+role", body)
        if m:
            return _clean_value(m.group(1)), "regex.role"
        m = re.search(r"(?i)(?:role|position)\s*[=:]\s*(.+?)(?=\s*\.\s|\s*$)", body)
        if m:
            return _clean_value(m.group(1)), "regex.role"

    elif field_type == "experience":
        # 1. Digit-based: "5 years of experience", "4 yrs exp"
        m = _EXPERIENCE_RE.search(body)
        if m:
            return f"{m.group(1)} years", "regex.experience"
        # 2. Word-form numbers: "four years of experience", "three yrs"
        m = _WORD_NUM_EXPERIENCE_RE.search(body)
        if m:
            word_num = m.group(1).lower()
            digit = _WORD_TO_DIGIT.get(word_num, word_num)
            return f"{digit} years", "regex.experience"
        # 3. Bare digit: "5 years" (without "of experience")
        m = re.search(r"(?i)(\d+)\s*years?\s", body)
        if m:
            return f"{m.group(1)} years", "regex.experience"

    elif field_type == "education":
        m = _EDUCATION_RE.search(body)
        if m:
            return _clean_value(m.group(0)), "regex.education"
    
    elif field_type == "skills":
        # 1. Explicit skills list after trigger phrases ("skills include...", "proficient in...")
        m = _SKILLS_RE.search(body)
        if m:
            text = m.group(1).strip().rstrip(",").strip()
            terms = re.findall(r"\b([A-Z][A-Za-z]+(?:\.(?:NET|C\+\+|JS|ML|AI|NLP))?[A-Za-z]*)", text)
            if len(terms) >= 2:
                return ", ".join(terms[:10]), "regex.skills"
        # 2. Implicit skills from context phrases ("designing mobile interfaces")
        m = _IMPLICIT_SKILLS_RE.search(body)
        if m:
            text = " ".join(m.group(1).split()).rstrip(",.").strip()
            terms = re.findall(r"\b([A-Z][A-Za-z]+(?:\.(?:NET|C\+\+|JS|ML|AI|NLP))?[A-Za-z]*)", text)
            if len(terms) >= 2:
                return ", ".join(terms[:10]), "regex.skills"
            elif terms:
                return ", ".join(terms[:10]), "regex.skills"
            # Fallback: return the phrase context (better than nothing
            # for downstream consumers that need to know skills exist).
            if len(text) > 3:
                return text, "regex.skills"

    elif field_type == "seniority":
        for level in sorted(_SENIORITY_LEVELS, key=len, reverse=True):
            if re.search(r"\b" + re.escape(level) + r"\b", body, re.IGNORECASE):
                return level, "regex.seniority"

    elif field_type == "notice_period":
        m = _NOTICE_PERIOD_RE.search(body)
        if m:
            return f"{m.group(1)} {m.group(2)}", "regex.notice_period"
        if _IMMEDIATE_AVAILABILITY_RE.search(body):
            return "immediately", "regex.notice_period"

    elif field_type == "work_type":
        for keyword, value in sorted(_WORK_TYPE_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r"\b" + re.escape(keyword) + r"\b", body, re.IGNORECASE):
                return value, "regex.work_type"

    elif field_type == "location":
        # Location keywords (based in X, located in X, from X...)
        for pattern, _group in _LOCATION_PATTERNS:
            pat = re.compile(pattern)
            m = pat.search(body)
            if m:
                val = m.group(1).strip().rstrip(".,;")
                if val:
                    return val, "regex.location"
        # Country fallback
        for country in _COUNTRIES:
            if re.search(r"\b" + re.escape(country) + r"\b", body, re.IGNORECASE):
                return country, "regex.location"

    elif field_type == "enum":
        # Status/priority/state - look for explicit status patterns
        if "status" in fl:
            m = re.search(r"(?i)\bstatus\s*[:=]\s*(.+?)(?=\s*\.\s|\s*$)", body)
            if m:
                return m.group(1).strip().rstrip(".,;"), "regex.enum"
        if "priority" in fl:
            m = re.search(r"(?i)\bpriority\s*[:=]\s*(.+?)(?=\s*\.\s|\s*$)", body)
            if m:
                return m.group(1).strip().rstrip(".,;"), "regex.enum"
        # General enum pattern - look for known enum values
        enum_values = ["pending", "confirmed", "approved", "rejected", "completed", "cancelled",
                       "in progress", "open", "closed", "resolved", "shipped", "delivered", "processing",
                       "high", "medium", "low", "critical", "urgent", "normal"]
        for val in enum_values:
            if re.search(r"\b" + re.escape(val) + r"\b", body, re.IGNORECASE):
                return val, "regex.enum"

    elif field_type == "boolean":
        if re.search(r"\b(?:yes|true|y)\b", body, re.IGNORECASE):
            return "yes", "regex.boolean"
        if re.search(r"\b(?:no|false|n)\b", body, re.IGNORECASE):
            return "no", "regex.boolean"

    elif field_type == "identifier":
        # IDs, reference numbers, codes - look for alphanumeric codes
        m = re.search(r"\b[A-Z0-9][A-Z0-9\-_]{4,}\b", body)
        if m:
            return m.group(0), "regex.identifier"

    elif field_type == "currency":
        m = re.search(r"\b(?:USD|EUR|GBP|INR|CAD|AUD|JPY|CNY|CHF)\b", body, re.IGNORECASE)
        if m:
            return m.group(0).upper(), "regex.currency"

    elif field_type == "duration":
        m = _DURATION_RE.search(body)
        if m:
            return m.group(0).strip(), "regex.duration"

    elif field_type in ("text_list", "email_list", "organization_list", "person_list", "identifier_list", "integer_list", "money_list", "line_items"):
        # For list types, try to extract multiple values
        # This is a simplified version - in practice would need more sophisticated parsing
        pass

    # If we have a custom pattern from the registry, try it
    if pattern_str and pattern_str not in (
        _EMAIL_RE.pattern, _PHONE_RE.pattern, _IP_RE.pattern,
        _DOMAIN_RE.pattern, _URL_RE.pattern, _DATE_RE.pattern,
        _TIME_RE.pattern, _MONEY_RE.pattern,
    ):
        try:
            custom_pat = re.compile(pattern_str)
            for _m in custom_pat.finditer(body):
                _val = _m.group(0)
                # Reject false positives (e.g. "Dear Recruiter" matched as a
                # company name by the broad Capitalized-words regex).
                if _val and not _is_implausible_entity(field_name, _val, 1.0):
                    return _val, "regex.custom"
        except re.error:
            pass

    return None, "none"


def _extract_generic_value(field_name: str, body: str) -> Optional[str]:
    """Last-resort extraction: find any inline KV pair whose key shares
    tokens with the field name."""
    fl = field_name.lower()
    field_tokens = set(fl.split("_"))
    # Remove common suffixes that aren't part of the semantic key.
    field_tokens.discard("name")
    field_tokens.discard("email")
    field_tokens.discard("amount")
    field_tokens.discard("date")
    field_tokens.discard("number")
    if not field_tokens:
        return None

    # First pass: prefer exact KV-key matches (key equals one of the
    # search keys, or key contains a search key as a whole-word token).
    search_keys = _field_to_search_keys(field_name)
    search_key_tokens = {k.lower() for k in search_keys}
    best_value = None
    best_score = 0
    for match in _KV_INLINE_RE.finditer(body):
        key = match.group("key").strip()
        key_lower = key.lower()
        key_tokens = set(key_lower.split())
        # Exact key match (case-insensitive).
        if key_lower in search_key_tokens:
            value = match.group("val").strip()
            if value and len(value) > 1:
                return value
        # Token overlap: prefer higher overlap, then earlier position.
        # Require at least 2 overlapping tokens or 50% of field tokens
        # to avoid weak matches like "tax" in "tax_return_summary" matching
        # "Federal tax withheld".
        overlap = len(field_tokens & key_tokens)
        if overlap >= 2 or (len(field_tokens) > 0 and overlap / len(field_tokens) >= 0.5):
            if overlap > best_score:
                best_score = overlap
                best_value = match.group("val").strip()

    if best_value and len(best_value) > 1:
        return best_value

    # Last resort: try search-key patterns with word-boundary + inflection
    # suffixes (same logic as _extract_from_keyvalue, no extra words before
    # the separator so "Proposal Date:" does not match key "proposal").
    # Requires at least one separator character to avoid matching words
    # that appear in prose (e.g., "email" in "this email as my application").
    _suffix_alt = r"(?:s|es|'s|ed|ing|ion|ions)"
    for key in search_keys:
        if " " in key.lower():
            pat = re.compile(
                r"(?i)\b" + re.escape(key) + r"\b\s*[:=]\s*(.+?)" + _SENT_BOUNDARY
            )
        else:
            pat = re.compile(
                r"(?i)\b" + re.escape(key) + r"(?:" + _suffix_alt + r")?\b\s*[:=]\s*(.+?)" + _SENT_BOUNDARY
            )
        m = pat.search(body)
        if m:
            value = m.group(1).strip()
            if value and len(value) > 1:
                return value
    return None
