
"""Dynamic Field Discovery - LLM-like field generation for ANY topic.

This module provides LLM-like field generation capabilities:
1. Semantic understanding of the topic/query to generate comprehensive field lists
2. Domain concept matching (job application, invoice, audit, incident, etc.)
3. Token-based field generation for arbitrary topics
4. Body-driven KV pair and noun phrase extraction as supplementary signals
5. spaCy NER entity detection for additional field discovery
6. Sentence-transformers semantic similarity ranking (optional)

All ML backends are optional - works with zero runtime dependencies using
comprehensive semantic field registry.
"""
from __future__ import annotations

import re
import logging

from .field_registry import (
    generate_fields_from_topic,
    get_field_type,
    get_kv_keys_for_field,
    get_regex_pattern_for_field,
    get_gliner_label,
    UNIVERSAL_FIELDS,
    DOMAIN_CONCEPTS,
    resolve_topic_to_concept,
    get_field_names_for_topic,
    get_fields_for_concept,
)

logger = logging.getLogger(__name__)

# ── Optional dependencies (lazy / auto-detected) ────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# Lazy-loaded model cache.
_ST_MODEL: SentenceTransformer | None = None
_ST_MODEL_NAME = "all-MiniLM-L6-v2"


# ── Tunable thresholds ─────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = 0.35
MAX_FIELDS: int = 50  # Increased from 20 for comprehensive extraction
MIN_LABEL_LEN: int = 2
MAX_LABEL_LEN: int = 60

# ── Pre-compiled regexes ────────────────────────────────────────────────────────

# Sentence-boundary lookahead used by the inline KV regex: matches a period
# (optionally followed by whitespace) followed by a capital letter or digit
# (next sentence), end-of-string, or a newline.  Including digits in the
# lookahead prevents inline KV values from bleeding into the next sentence
# when the next sentence starts with a number (e.g. "Name: John. 6 years...").
_SENT_BOUNDARY_LKP = r"(?=\s*\.\s+[A-Z\d]|\s*\.\s*$|\s*\n\s*[A-Z\d]|\s*$)"

# Inline key-value pair: "Key: value" where the key can appear anywhere in the
# text (not just at line start).  Value extends to the next sentence boundary.
_KV_RE = re.compile(
    r"(?i)\b(?P<key>[A-Z][A-Za-z0-9][A-Za-z0-9 _\-&]{0,40}?)\s*[=:]\s*"
    r"(?P<val>.+?)" + _SENT_BOUNDARY_LKP
)

# Noun-phrase candidates: sequences of 1–5 Capitalized-word tokens.
# Each token starts with an uppercase letter; multi-word phrases are captured
# greedily so "Senior Python Engineer" is kept as one phrase, not three.
_NOUN_PHRASE_RE = re.compile(
    r"\b(?:(?:[A-Z][a-z]+|[A-Z]{2,}))(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|\d+))*\b"
)

# Common stop-words that are not meaningful field labels.
_STOPWORDS: set[str] = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "at",
    "by", "with", "from", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "as", "into",
    "Dear", "Hi", "Hello", "Thanks", "Thank", "Regards", "Best", "Sincerely",
}

def to_snake_case(text: str) -> str:
    """Convert a human-readable label to snake_case.

    Examples:
        "Auditor"           -> "auditor"
        "Defects Found"     -> "defects_found"
        "Source IP"         -> "source_ip"
        "Target Domain"     -> "target_domain"
    """
    # Insert underscore between camelCase boundaries.
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", text)
    # Replace spaces, hyphens, and other separators with underscores.
    text = re.sub(r"[\s\-]+", "_", text)
    # Collapse multiple underscores.
    text = re.sub(r"_+", "_", text)
    return text.lower().strip("_")


def _extract_kv_labels(body: str) -> list[str]:
    """Extract key labels from key-value pairs in the body.

    Only labels that look like field names (Capitalized, reasonable length,
    not common prose) are returned, in order of first appearance.
    """
    seen: set[str] = set()
    labels: list[str] = []
    for match in _KV_RE.finditer(body):
        key = match.group("key").strip()
        if not key or key.lower() in _STOPWORDS:
            continue
        if len(key) < MIN_LABEL_LEN or len(key) > MAX_LABEL_LEN:
            continue
        normalized = key.strip()
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        labels.append(normalized)
    return labels


def _extract_noun_phrase_labels(body: str) -> list[str]:
    """Extract capitalized noun-phrase candidates from the body text."""
    seen: set[str] = set()
    labels: list[str] = []
    for match in _NOUN_PHRASE_RE.finditer(body):
        phrase = match.group(0).strip()
        if not phrase or phrase.lower() in _STOPWORDS:
            continue
        if len(phrase) < MIN_LABEL_LEN:
            continue
        # Skip if the phrase is purely numeric or starts with a digit.
        if phrase[0].isdigit():
            continue
        # Skip single words that are common English function words.
        if " " not in phrase and phrase.lower() in _STOPWORDS:
            continue
        if phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        labels.append(phrase)
    return labels


def _collect_candidate_labels(body: str) -> tuple[list[str], list[str]]:
    """Collect candidate field labels from the body.

    Returns ``(kv_labels, np_labels)`` — KV-pair labels are separated from
    noun-phrase labels so the ranking logic can treat them differently:
    KV labels are always included (structurally determined), while noun
    phrases are filtered by semantic similarity to the topic.
    """
    kv_labels = _extract_kv_labels(body)
    np_labels = _extract_noun_phrase_labels(body)

    # Remove noun phrases that are already captured as KV labels.
    kv_lower = {l.lower() for l in kv_labels}
    np_unique = [l for l in np_labels if l.lower() not in kv_lower]

    return kv_labels[:MAX_FIELDS * 2], np_unique[:MAX_FIELDS * 3]


def _load_st_model() -> SentenceTransformer | None:
    """Lazy-load the sentence-transformers model (once)."""
    global _ST_MODEL
    if _ST_MODEL is not None:
        return _ST_MODEL
    if not _HAS_ST:
        return None
    try:
        _ST_MODEL = SentenceTransformer(_ST_MODEL_NAME)
        return _ST_MODEL
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load sentence-transformers model: %s", exc)
        _ST_MODEL = None
        return None


def _cosine_similarity(vec_a, vec_b) -> float:
    """Compute cosine similarity between two vectors."""
    if _HAS_NUMPY:
        a = np.array(vec_a, dtype=float)
        b = np.array(vec_b, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    # Manual fallback.
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = sum(x * x for x in vec_a) ** 0.5
    norm_b = sum(y * y for y in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_overlap_score(topic: str, label: str) -> float:
    """Fallback similarity score based on keyword overlap (no ML).

    Returns a score in [0, 1].
    """
    topic_words = set(re.findall(r"\w+", topic.lower()))
    label_words = set(re.findall(r"\w+", label.lower()))

    # Direct word overlap.
    overlap = topic_words & label_words
    if not topic_words:
        return 0.0
    score = len(overlap) / len(topic_words)

    # Bonus for semantic keyword overlap.
    if topic_words:
        overlap_ratio = len(overlap) / len(topic_words | label_words)
        score = max(score, overlap_ratio * 1.5)

    return min(score, 1.0)


# ── Public API ─────────────────────────────────────────────────────────────────

# Generic extraction field names used when the body yields no discoverable
# structure (no KV pairs, no matching noun phrases).  These names are chosen
# so the regex-based extractor's type-aware patterns recognise them
# (e.g. ``name`` → sign-off extraction, ``email`` → email regex,
# ``job_role`` → role-prose pattern).  The list is deliberately broad so
# that even completely free-form prose emails return typed results instead
# of opaque placeholder names.
_DEFAULT_FIELDS: tuple[str, ...] = (
    # Identity & contact
    "personal_name",
    "full_name",
    "email_address",
    "phone_number",
    # Role & position
    "job_role",
    "position_title",
    "current_company",
    "current_role",
    "seniority_level",
    "work_type",
    # Experience & metrics
    "experience_years",
    "education",
    "skills",
    "salary_expectation",
    "notice_period",
    "portfolio_url",
    # Common business fields
    "status",
    "company",
    "location",
    "date",
    "amount",
    "total",
    # Additional universal fields for broader coverage
    "message_text",
    "attachment_count",
    "cc_recipients",
    "bcc_recipients",
    "reference_id",
    "priority_level",
)


def _minimal_fields() -> list[str]:
    """Return generic extraction field names as a last-resort fallback.

    Used when no field labels can be discovered from the email body at all
    (e.g. empty body, no KV pairs, no noun phrases).  These field names are
    recognised by the regex-based extractor's type patterns so that even
    without discoverable structure the engine still produces meaningful,
    typed output rather than opaque placeholders.
    """
    return list(_DEFAULT_FIELDS)


# ── Topic → universal concept mapping ───────────────────────────────────────────
# When the body has no discoverable structure (no KV pairs, noun phrases fail),
# we fall back to PROBE FIELDS — a set of generic entity categories that the
# regex-based extractor knows how to extract.  These are deliberately broad so
# free-form prose emails return typed results instead of opaque placeholders.
_PROBE_FIELDS: tuple[str, ...] = _DEFAULT_FIELDS


# ── SpaCy NER support (optional) ────────────────────────────────────────────────
try:
    import spacy
    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False

_NLP_CACHED = None


def _get_spacy_model():
    """Lazy-load the spaCy model (once)."""
    global _NLP_CACHED
    if not _HAS_SPACY:
        return None
    if _NLP_CACHED is not None:
        return _NLP_CACHED
    try:
        _NLP_CACHED = spacy.load('en_core_web_sm')
        return _NLP_CACHED
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load spaCy model: %s", exc)
        _NLP_CACHED = None
        return None


# SpaCy entity label → field name mapping.
# When spaCy detects a PERSON in an email body, we know that span likely
# corresponds to one of these field names — no keyword guessing involved.
_SPACY_TO_FIELD: frozenset[tuple[str, str]] = frozenset({
    # Identity & contact
    ("PERSON", "personal_name"), ("PERSON", "full_name"),
    # Organization / company
    ("ORG", "company"), ("ORG", "current_company"), ("ORG", "organization"),
    # Geography / location
    ("GPE", "location"), ("GPE", "city"), ("GPE", "country"), ("GPE", "address"),
    # Time / duration
    ("DATE", "date"), ("DATE", "deadline"), ("DATE", "start_date"),
    ("TIME", "time"),
    # Monetary
    ("MONEY", "amount"), ("MONEY", "total"), ("MONEY", "salary_expectation"), ("MONEY", "price"),
    # Quantity / count
    ("CARDINAL", "quantity"), ("CARDINAL", "count"), ("CARDINAL", "experience_years"),
    # Work of art / product
    ("WORK_OF_ART", "job_role"), ("PRODUCT", "skills"),
})


def _entity_to_fields(entities: list[object]) -> list[str]:
    """Map spaCy detected entities to field names."""
    seen: set[str] = set()
    fields: list[str] = []
    for ent in entities:
        for entity_label, field_name in _SPACY_TO_FIELD:
            if ent.label_ == entity_label and field_name not in seen:
                seen.add(field_name)
                fields.append(field_name)
    return fields


def _resolve_concept_semantically(topic: str) -> str | None:
    """Find the best matching domain concept by neural semantic similarity.

    Uses *sentence-transformers* embeddings to match topics to concepts by **meaning**,
    not keyword tokens.  So ``"career opportunity"`` matches the ``job_application``
    concept because the two phrases are semantically equivalent, even though they
    share no tokens.

    Falls back to :func:`resolve_topic_to_concept` (token-based) when the model
    is unavailable, so the system never degrades to broken behavior.
    """
    # Fast path: exact/keyword resolution (handles "job application", "invoice", etc.)
    concept = resolve_topic_to_concept(topic)
    if concept:
        return concept

    st_model = _load_st_model()
    if st_model is None:
        return None

    topic_norm = topic.lower().replace("_", " ").strip()
    if not topic_norm:
        return None

    try:
        topic_emb = st_model.encode([topic_norm], convert_to_numpy=_HAS_NUMPY)
        topic_emb = topic_emb[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Topic embedding failed: %s", exc)
        return None

    # Pre-encode each concept's combined description + keywords.
    concept_texts = []
    concept_names = list(DOMAIN_CONCEPTS.keys())
    for cname in concept_names:
        concept = DOMAIN_CONCEPTS[cname]
        text = (concept.description + " " + " ".join(concept.keywords[:5])).lower()
        concept_texts.append(text)

    try:
        concept_embs = st_model.encode(concept_texts, convert_to_numpy=_HAS_NUMPY)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Concept embedding failed: %s", exc)
        return None

    best_concept = None
    best_score = 0.0

    for i, cname in enumerate(concept_names):
        if _HAS_NUMPY:
            sim = float(
                np.dot(topic_emb, concept_embs[i])
                / (np.linalg.norm(topic_emb) * np.linalg.norm(concept_embs[i]) + 1e-10)
            )
        else:
            sim = _cosine_similarity(
                [float(x) for x in topic_emb],
                [float(x) for x in concept_embs[i]],
            )
        if sim > best_score:
            best_score = sim
            best_concept = cname

    # Only promote to a concept when confidence is above the similarity threshold.
    if best_score >= SIMILARITY_THRESHOLD and best_concept:
        logger.debug("Semantic concept match: %r -> %s (%.3f)", topic, best_concept, best_score)
        return best_concept

    return None


def _probe_topic_for_concepts(topic: str) -> list[str]:
    """Derive candidate field names from the topic using neural semantic understanding.

    LLM-like understanding without LLM calls:

    1. **Semantic concept resolution** — uses sentence-transformers embeddings to
       find the best matching domain concept by meaning (e.g. ``"career
       opportunity"`` → ``job_application``).  If found, that concept's rich field
       list is used as the seed.
    2. If no concept matches, fall back to :func:`generate_fields_from_topic`
       (token-aware semantic registry matching).
    3. Rank the candidate field names by **embedding similarity** to the topic —
       replaces keyword-overlap heuristics.
    4. Return the full ranked set (capped at MAX_FIELDS) so extraction can
       attempt every field the topic implies.
    """
    topic = (topic or "").strip()
    if not topic:
        return [f.name for f in UNIVERSAL_FIELDS[:15]]

    # 1. Keyword-based field generation (token-aware registry matching).
    #    Handles SEMANTIC_CONCEPT_FIELDS like "inspection", "reservation", etc.
    keyword_fields = generate_fields_from_topic(topic, max_fields=MAX_FIELDS)

    # 2. Neural concept resolution (meaning-based matching).
    #    Finds semantically similar concepts even when keywords don't match
    #    (e.g. "career opportunity" → job_application).  These fields are
    #    ADDED to the keyword-derived set, not a replacement for it.
    semantic_concept = _resolve_concept_semantically(topic)
    neural_fields: list[str] = []
    if semantic_concept:
        concept_specs = get_fields_for_concept(semantic_concept)
        neural_fields = [f.name for f in sorted(concept_specs, key=lambda s: s.priority)]

    # 3. MERGE keyword fields (priority) + neural concept fields (supplementary).
    candidates: list[str] = list(keyword_fields)
    for fn in neural_fields:
        if fn not in candidates:
            candidates.append(fn)

    # Cap to leave room for body-driven discovery (KV pairs, spaCy NER, noun
    # phrases) in discover_fields().
    candidates = candidates[: MAX_FIELDS - 10]

    if not candidates:
        return _rank_concepts_semantically(topic, [f.name for f in UNIVERSAL_FIELDS])[:MAX_FIELDS]

    # Semantic ranking: rank candidate fields by embedding similarity to the topic.
    ranked = _rank_concepts_semantically(topic, candidates)
    # Return full ranked list (no halving — let extraction validate).
    return ranked[:MAX_FIELDS]


def _rank_concepts_semantically(topic: str, candidates: list[str]) -> list[str]:
    """Rank candidate field names by embedding similarity to the *topic*.

    Uses *sentence-transformers* (``all-MiniLM-L6-v2``) when available for
    true semantic understanding.  Falls back to keyword-overlap scoring when
    the model is not installed — never degrades to broken behavior.
    """
    if not candidates:
        return []

    topic_norm = topic.lower().replace("_", " ")
    st_model = _load_st_model()

    if st_model is not None:
        try:
            topic_emb = st_model.encode([topic_norm], convert_to_numpy=_HAS_NUMPY)[0]
            cand_texts = [c.replace("_", " ") for c in candidates]
            cand_embs = st_model.encode(cand_texts, convert_to_numpy=_HAS_NUMPY)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sentence embedding failed, falling back to overlap: %s", exc)
            return _rank_concepts_overlap(topic_norm, candidates)

        scored: list[tuple[str, float]] = []
        for i, candidate in enumerate(candidates):
            if _HAS_NUMPY:
                sim = float(
                    np.dot(topic_emb, cand_embs[i])
                    / (np.linalg.norm(topic_emb) * np.linalg.norm(cand_embs[i]) + 1e-10)
                )
            else:
                sim = _cosine_similarity(
                    [float(x) for x in topic_emb],
                    [float(x) for x in cand_embs[i]],
                )
            scored.append((candidate, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        # Return ALL ranked candidates (reorder only, don't over-filter).
        # Embedding similarity between short field names and a topic is inherently
        # low, so thresholding here would drop legitimately relevant fields.
        # Relevance is enforced downstream during extraction, not at discovery.
        return [c for c, _ in scored]

    # Fallback: keyword-overlap similarity (no ML dependency).
    return _rank_concepts_overlap(topic_norm, candidates)


def _rank_concepts_overlap(topic: str, candidates: list[str]) -> list[str]:
    """Rank candidate fields by keyword-overlap similarity to the topic."""
    scored: list[tuple[str, float]] = []
    for candidate in candidates:
        score = _keyword_overlap_score(topic, candidate.replace("_", " "))
        scored.append((candidate, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored if _ >= SIMILARITY_THRESHOLD] or [c for c, _ in scored[:5]]


def discover_fields(topic: str, body: str) -> list[str]:
    """Dynamically discover target field names from the email content.

    Discovery uses multiple complementary signals for LLM-like field generation:

    1. **KV-pair labels** from the body (structurally determined, highest
       confidence, always included up to the field cap).
    2. **spaCy NER** entity types mapped to field names (neural entity linking
       from actual body content, detects persons/orgs/dates/money/etc.).
    3. **Noun-phrase labels** from the body, ranked by semantic similarity
       to the topic (sentence-transformers or keyword-overlap fallback).
    4. **Topic-driven semantic field generation** — supplementary fields
       derived from the query/topic to fill remaining slots.

    Body-driven signals are prioritized so the engine works like an LLM —
    observing what is actually present in the content first, then applying
    topic context.  Topic-driven fields only fill remaining capacity,
    ensuring real entities (e.g. personal_name) are never pushed out by
    irrelevant template fields when the topic does not match the content.

    Accepts None or empty strings for topic and body.
    """
    topic = (topic or "").strip()
    topic_norm = topic.lower().replace("_", " ") if topic else ""

    seen: set[str] = set()
    result: list[str] = []

    # -- No body: fall back to topic-driven fields only --
    if not body:
        topic_fields = _probe_topic_for_concepts(topic)
        for fn in topic_fields:
            if fn not in seen:
                seen.add(fn)
                result.append(fn)
            if len(result) >= MAX_FIELDS:
                break
        return result[:MAX_FIELDS]

    # -- 1. KV-pair labels from the body (explicit structure, highest priority) --
    kv_labels, np_labels = _collect_candidate_labels(body)
    for label in kv_labels:
        fn = _field_name_from_label(label)
        if fn and fn not in seen:
            seen.add(fn)
            result.append(fn)
        if len(result) >= MAX_FIELDS:
            return result[:MAX_FIELDS]

    # -- 2. spaCy NER entity discovery (neural, detects actual entities) --
    try:
        _nlp = _get_spacy_model()
        if _nlp is not None:
            _doc = _nlp(body)
            for _fn in _entity_to_fields(_doc.ents):
                if _fn not in seen and len(result) < MAX_FIELDS:
                    seen.add(_fn)
                    result.append(_fn)
    except Exception as _e:
        logger.debug("spaCy entity discovery failed: %s", _e)

    # -- 2.5. Concept/default body-scan fields — universally extractable field
    # names whose regex patterns can match content in any email body.  These
    # ensure the extraction layer can find values for common fields (job_role,
    # experience_years, skills, etc.) regardless of the topic.  When the topic
    # maps to a known semantic concept, that concept's field list is tried first
    # so domain-specific fields (e.g. reserver_name) are not crowded out.
    concept = _resolve_concept_semantically(topic)
    if concept:
        _body_scan = [f.name for f in get_fields_for_concept(concept)] + list(_DEFAULT_FIELDS)
    else:
        _body_scan = list(_DEFAULT_FIELDS)
    for fn in _body_scan:
        if fn not in seen and len(result) < MAX_FIELDS:
            seen.add(fn)
            result.append(fn)
        if len(result) >= MAX_FIELDS:
            return result[:MAX_FIELDS]

    # -- 3. Noun-phrase labels ranked by semantic similarity to the topic --
    np_ranked: list[str] = []
    if np_labels:
        if topic_norm and _HAS_ST:
            model = _load_st_model()
            if model is not None:
                np_ranked = _rank_noun_phrases(model, topic_norm, np_labels)
        if not np_ranked:
            if topic_norm:
                np_ranked = _rank_noun_phrases_overlap(topic_norm, np_labels)
            else:
                np_ranked = sorted(np_labels, key=len, reverse=True)

    for fn in np_ranked:
        if fn and fn not in seen:
            seen.add(fn)
            result.append(fn)
        if len(result) >= MAX_FIELDS:
            break

    # -- 4. Topic-driven fields (supplementary, fills remaining slots) --
    if len(result) < MAX_FIELDS:
        topic_fields = _probe_topic_for_concepts(topic)
        for fn in topic_fields:
            if fn not in seen:
                seen.add(fn)
                result.append(fn)
            if len(result) >= MAX_FIELDS:
                return result[:MAX_FIELDS]

    return result[:MAX_FIELDS]
def _rank_noun_phrases(
    model, topic: str, labels: list[str]
) -> list[str]:
    """Rank noun-phrase labels by semantic similarity to the topic."""
    try:
        topic_emb = model.encode(topic, convert_to_numpy=True)
        label_embs = model.encode(labels, convert_to_numpy=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sentence embedding failed, falling back to overlap: %s", exc)
        return _rank_noun_phrases_overlap(topic, labels)

    scored: list[tuple[str, float]] = []
    for label, emb in zip(labels, label_embs):
        if _HAS_NUMPY:
            sim = float(
                np.dot(topic_emb, emb)
                / (np.linalg.norm(topic_emb) * np.linalg.norm(emb) + 1e-10)
            )
        else:
            sim = _cosine_similarity(
                [float(x) for x in topic_emb],
                [float(x) for x in emb],
            )
        scored.append((label, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    result: list[str] = []
    for label, sim in scored:
        if sim < SIMILARITY_THRESHOLD:
            continue
        # Map noun phrases to semantic field names via the registry so they
        # integrate with the type-aware regex extractor (e.g. "Job Title" →
        # "job_role", "Years Experience" → "years_experience").
        fn = _field_name_from_label(label)
        if fn and fn not in result:
            result.append(fn)
        if len(result) >= MAX_FIELDS:
            break
    return result


def _rank_noun_phrases_overlap(topic: str, labels: list[str]) -> list[str]:
    """Rank noun-phrase labels by keyword-overlap similarity to the topic."""
    scored: list[tuple[str, float]] = []
    for label in labels:
        score = _keyword_overlap_score(topic, label)
        scored.append((label, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    result: list[str] = []
    for label, score in scored:
        if score < SIMILARITY_THRESHOLD:
            continue
        fn = _field_name_from_label(label)
        if fn and fn not in result:
            result.append(fn)
        if len(result) >= MAX_FIELDS:
            break
    return result


def _field_name_from_label(label: str) -> str:
    """Convert a discovered label to a snake_case field name.

    Multi-word labels like "Source IP" become ``source_ip``; single-word labels
    like "Auditor" become ``auditor``.  No semantic suffix is inferred — all
    field names are derived purely from the discovered label text, with no
    manual keyword-to-type mapping.
    """
    return to_snake_case(label)
