"""Core package for the Email Extraction Engine.

Public API:
    from core.engine import TopicEmailExtractor
    extractor = TopicEmailExtractor()
    result = extractor.extract(eml_source="sample.eml", topic="job application")
"""
from .engine import TopicEmailExtractor, ExtractionResult

__all__ = ["TopicEmailExtractor", "ExtractionResult"]
