"""Email Extraction Engine — a generic, zero-cost engine for extracting
structured data from .eml files or raw RFC-822 email text for any topic.

Two-tier field resolution:
  1. Manual registry (``core.templates``) — pre-defined field lists for
     known domains (job application, DMARC report, invoice, etc.).
  2. Dynamic schema discovery (``core.dynamic_schema``) — derives field
     names from KV pairs, noun phrases, and topic-driven concept probes
     for arbitrary user-supplied topics.

Public API (lazy imports to avoid pulling in GLiNER / torch at import time):
    from email_extractor_engine.core.engine import TopicEmailExtractor, ExtractionResult
    result = TopicEmailExtractor().extract(eml_source="sample.eml", topic="job application")
"""
from __future__ import annotations

__all__: list[str] = [
    "ExtractionResult",
    "TopicEmailExtractor",
    "extract",
]
