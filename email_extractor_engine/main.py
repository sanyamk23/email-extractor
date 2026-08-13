"""CLI entry point for the Email Extraction Engine.

Usage:
    python -m email_extractor_engine.main --eml sample.eml --requirement "job application"
    python -m email_extractor_engine.main --stdin --requirement "vendor inspection" < sample.eml
    python -m email_extractor_engine.main --eml sample.eml --requirement "dmarc report" --pretty
    python -m email_extractor_engine.main --eml sample.eml            # defaults to "job application"
"""
from __future__ import annotations

import sys
import argparse
import os

# Allow running as a script (python main.py) or as a module
# (python -m email_extractor_engine.main).  In both cases we add the package
# directory to sys.path so ``core`` is importable regardless of how the
# entry point is invoked.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from core.engine import TopicEmailExtractor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="email-extractor",
        description="Generic zero-cost Email Extraction Engine. "
                    "Extract structured data from .eml files using a topic string.",
    )
    parser.add_argument(
        "--eml", "--eml-source",
        dest="eml_source",
        help="Path to a .eml/.emlx file, or raw RFC-822 text.",
    )
    parser.add_argument(
        "--requirement", "-r",
        dest="topic",
        default="job application",
        help='Extraction topic / domain directive (e.g. "job application", '
             '"DMARC report", "invoice"). Defaults to "job application".',
    )
    parser.add_argument(
        "--topic",
        dest="topic",
        default=argparse.SUPPRESS,
        help="Alias for --requirement (backward compatibility).",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read raw email text from stdin instead of a file path.",
    )
    parser.add_argument(
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print JSON output with indentation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"GLiNER confidence threshold (default: {TopicEmailExtractor.__init__.__defaults__[0]}).",
    )
    args = parser.parse_args(argv)

    # Gather the email source.
    if args.stdin:
        raw_text = sys.stdin.read()
        eml_source = "<stdin>"
    else:
        raw_text = None
        eml_source = args.eml_source
        if eml_source is None:
            parser.error("--eml is required (or use --stdin)")

    # Build extractor.
    kwargs = {}
    if args.threshold is not None:
        kwargs["gliner_threshold"] = args.threshold
    extractor = TopicEmailExtractor(**kwargs)

    result = extractor.extract(eml_source=eml_source, topic=args.topic, raw_text=raw_text)
    indent = 2 if args.pretty else None
    print(result.to_json(indent=indent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
