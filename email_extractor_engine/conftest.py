"""Pytest configuration for email_extractor_engine tests.

Ensures the ``core`` package is importable regardless of the working
directory from which pytest is invoked.
"""
import sys
import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
