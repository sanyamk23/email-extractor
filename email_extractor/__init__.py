"""Rule-based job-application email parser.

Public entry point::

    from email_extractor import parse_job_application
    result = parse_job_application(raw_email_text)
"""
from .pipeline import parse_job_application

__all__ = ["parse_job_application"]
__version__ = "1.0.0"
