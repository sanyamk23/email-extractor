#!/usr/bin/env python3
"""Dependency-free web UI for the job-application email parser.

Run it and open the printed URL in a browser, then upload a ``.eml`` file:

    python web_ui.py            # -> http://localhost:8000
    python web_ui.py 8080       # -> http://localhost:8080

The whole thing is built on the Python standard library only, so it runs with
no third-party installs (``email_extractor`` itself is stdlib + optional).
"""
from __future__ import annotations

import html
import json
import re
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from email_extractor.pipeline import parse_job_application


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested directly).
# --------------------------------------------------------------------------- #
def parse_multipart(body: bytes, boundary: str) -> dict[str, bytes]:
    """Parse a ``multipart/form-data`` body into a {field-name: value} dict.

    A minimal, dependency-free parser sufficient for single-file uploads.
    """
    fields: dict[str, bytes] = {}
    delim = b"--" + boundary.encode("latin-1")
    for part in body.split(delim):
        if part in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if b"\r\n\r\n" not in part:
            continue
        header_blob, content = part.split(b"\r\n\r\n", 1)
        disposition = ""
        for line in header_blob.split(b"\r\n"):
            if line.lower().startswith(b"content-disposition:"):
                disposition = line.decode("latin-1")
                break
        name = None
        for token in disposition.split(";"):
            token = token.strip()
            if token.startswith("name="):
                name = token[len("name="):].strip().strip('"')
        if name:
            fields[name] = content
    return fields


def parse_eml_bytes(raw: str) -> dict:
    """Run the full pipeline on raw ``.eml`` text, never raising to the caller."""
    try:
        return parse_job_application(raw)
    except Exception as exception:  # defensive: surface parse errors in the UI
        return {
            "is_job_application": None,
            "confidence_score": 0.0,
            "error": f"{type(exception).__name__}: {exception}",
            "traceback": traceback.format_exc(),
            "candidate": {},
        }


def format_value(value) -> str:
    """Format any candidate field value into an HTML-safe display string.

    Escaping happens here (single pass) so callers must not re-escape.
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, list):
        if not value:
            return "—"
        return ", ".join(html.escape(str(v)) for v in value)
    if isinstance(value, str):
        return html.escape(value) if value.strip() else "—"
    return html.escape(str(value))


def build_form_page(error: str | None = None) -> str:
    """Return the HTML for the landing / upload page."""
    error_block = ""
    if error:
        error_block = (
            '<div class="error">' + html.escape(error) + "</div>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Email extractor — upload .eml</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
           margin: 0; padding: 2rem; background: #f5f7fa; color: #1f2937; }}
    h1 {{ font-size: 1.4rem; margin: 0 0 .25rem; }}
    .sub {{ color: #6b7280; margin-bottom: 1.5rem; }}
    form {{ max-width: 480px; }}
    .row {{ display: flex; align-items: center; gap: .75rem; }}
    input[type=file] {{ flex: 1; }}
    button {{ background: #2563eb; color: #fff; border: 0; padding: .6rem 1.1rem;
             border-radius: .4rem; font: 600 1rem/1 sans-serif; cursor: pointer; }}
    button:hover {{ background: #1d4ed8; }}
    .error {{ background: #fee2e2; border: 1px solid #fca5a5; color: #991b1b;
             padding: .75rem 1rem; border-radius: .4rem; margin: 1rem 0; }}
    .hint {{ color: #6b7280; font-size: .85rem; margin-top: .5rem; }}
  </style>
</head>
<body>
  <h1>Email extractor</h1>
  <p class="sub">Upload a job-application <code>.eml</code> file and inspect the parsed fields.</p>
  {error_block}
  <form method="post" action="/parse" enctype="multipart/form-data">
    <div class="row">
      <input type="file" name="eml_file" accept=".eml" required>
      <button type="submit">Parse .eml</button>
    </div>
  </form>
  <p class="hint">No third-party dependencies required — open Developer Tools → Network if you want the raw JSON response too.</p>
</body>
</html>"""


def render_result(result: dict) -> str:
    """Return an HTML page rendering the parsed result of a single ``.eml``."""
    candidate = result.get("candidate") or {}
    sender = result.get("sender") or {}
    rows: list[str] = []

    def row(label: str, key: str, value) -> None:
        rows.append(
            f"<tr><th>{html.escape(label)}</th>"
            f"<td>{format_value(value)}</td></tr>"
        )

    row("Is application", "is_job_application", result.get("is_job_application"))
    row("Confidence", "confidence_score", result.get("confidence_score"))
    row("Job role", "job_role", result.get("job_role"))
    row("Name", "name", candidate.get("name"))
    row("Email", "email", candidate.get("email"))
    row("Phone", "phone", candidate.get("phone"))
    row("Links", "links", candidate.get("links"))
    row("Years of experience", "years_of_experience", candidate.get("years_of_experience"))
    row("Salary expectation", "salary_expectation", candidate.get("salary_expectation"))
    row("Notice period", "notice_period", candidate.get("notice_period"))
    row("Skills", "skills", candidate.get("skills"))
    row("Education", "education", candidate.get("education"))
    row("Seniority", "seniority", candidate.get("seniority"))
    row("Location", "location", candidate.get("location"))
    row("Company", "company", candidate.get("company"))
    row("Start date", "start_date", candidate.get("start_date"))
    row("Work type", "work_type", candidate.get("work_type"))
    row("Languages", "languages", candidate.get("languages"))
    row("Certifications", "certifications", candidate.get("certifications"))
    row("Sender name", "sender.name", sender.get("name"))
    row("Sender email", "sender.email", sender.get("email"))
    attachments = result.get("attachments") or []
    row("Attachments", "attachments", [a.get("filename", "") for a in attachments])

    cover = result.get("clean_cover_letter") or ""
    error = result.get("error")
    error_block = ""
    if error:
        error_block = (
            '<section class="panel"><h2>Parse note</h2>'
            '<pre class="code">' + html.escape(error) + "</pre></section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Email extractor — result</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
           margin: 0; padding: 2rem; background: #f5f7fa; color: #1f2937; }}
    h1 {{ font-size: 1.4rem; margin: 0 0 .25rem; }}
    .sub {{ color: #6b7280; margin-bottom: 1.5rem; }}
    a.back {{ color: #2563eb; text-decoration: none; }} a.back:hover {{ text-decoration: underline; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 720px; background: #fff;
            box-shadow: 0 1px 3px rgba(15,23,42,.08); }}
    th, td {{ text-align: left; padding: .55rem .85rem; border-bottom: 1px solid #e5e7eb;
             vertical-align: top; }}
    th {{ background: #f9fafb; width: 34%; color: #374151; font-weight: 600; }}
    .panel {{ background: #fff; margin-top: 1.5rem; padding: 1rem 1.25rem;
             border-radius: .4rem; box-shadow: 0 1px 3px rgba(15,23,42,.08); }}
    .panel h2 {{ margin: 0 0 .5rem; font-size: 1.05rem; }}
    pre.code {{ white-space: pre-wrap; word-break: break-word; max-height: 320px;
                overflow:auto; background: #0f172a; color: #e2e8f0; padding: 1rem;
                border-radius: .4rem; font-size: .8rem; }}
    .json-toggle {{ color: #2563eb; cursor: pointer; font-size: .85rem; }}
  </style>
</head>
<body>
  <h1>Email extractor — result</h1>
  <p class="sub"><a class="back" href="/">&larr; New upload</a></p>
  {error_block}
  <table>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <div class="panel">
    <h2>Clean cover letter</h2>
    <pre class="code">{html.escape(cover)}</pre>
  </div>
  <div class="panel">
    <h2>Raw JSON <span class="json-toggle" id="toggle">show</span></h2>
    <pre class="code" id="raw" style="display:none">{html.escape(json.dumps(result, indent=2, default=str))}</pre>
  </div>
  <script>
    const pre = document.getElementById('raw');
    const btn = document.getElementById('toggle');
    btn.onclick = () => {{ const on = pre.style.display === 'none'; pre.style.display = on ? 'block' : 'none'; btn.textContent = on ? 'hide' : 'show'; }};
  </script>
</body>
</html>"""


# --------------------------------------------------------------------------- #
# HTTP handler.
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep stdout clean; silence default logging
        pass

    def _send_html(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(build_form_page())
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        path = urlparse(self.path).path
        if path != "/parse":
            self.send_error(404)
            return

        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""

        raw = ""
        if content_type.startswith("multipart/form-data"):
            boundary = None
            for part in content_type.split(";"):
                if "boundary=" in part:
                    boundary = part.split("boundary=", 1)[1].strip().strip('"')
                    break
            if not boundary:
                self._send_html(build_form_page("Could not read upload boundary."), 400)
                return
            fields = parse_multipart(body, boundary)
            raw = fields.get("eml_file", b"").decode("utf-8", "replace")
        else:
            raw = body.decode("utf-8", "replace")

        if not raw.strip():
            self._send_html(build_form_page("No .eml file was received."), 400)
            return

        result = parse_eml_bytes(raw)
        self._send_html(render_result(result))


def main(argv: list[str]) -> int:
    port = int(argv[0]) if argv else 8000
    server = ThreadingHTTPServer(("localhost", port), _Handler)
    print(f"Email extractor UI ready: http://localhost:{port}")
    print("Upload a .eml file. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
