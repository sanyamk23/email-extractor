"""Command-line interface for the job-application email parser.

Usage::

    # Parse an RFC-822 .eml file
    email-extractor application.eml

    # Pipe a raw email on stdin
    cat application.eml | email-extractor --stdin

    # Feed a JSON document {"from", "subject", "body"} via stdin
    echo '{"from":"...","subject":"...","body":"..."}' | email-extractor --stdin
"""
from __future__ import annotations

import argparse
import json
import sys

from .pipeline import parse_job_application


def _load_email_data(raw: str):
    """Accept either a JSON dict ({"from","subject","body"}) or raw email text."""
    stripped = raw.lstrip()
    if stripped.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="email-extractor",
        description="Rule-based job-application email parser.",
    )
    parser.add_argument("file", nargs="?",
                        help="Path to an .eml file or a JSON file with "
                             "from/subject/body keys. Omit to read stdin.")
    parser.add_argument("-s", "--stdin", action="store_true",
                        help="Read the email from standard input.")
    parser.add_argument("-i", "--indent", type=int, default=2,
                        help="JSON indentation width (default: 2).")
    args = parser.parse_args(argv)

    if args.stdin or (args.file is None and not sys.stdin.isatty()):
        raw = sys.stdin.read()
    elif args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    else:
        parser.error("provide a file path or use --stdin to read from stdin.")
        return 2

    email_input = _load_email_data(raw)
    result = parse_job_application(email_input)
    print(json.dumps(result, indent=args.indent, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
