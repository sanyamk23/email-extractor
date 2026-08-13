#!/usr/bin/env python3
"""Quick web UI for the Email Extraction Engine.

Run it and open the printed URL in a browser:

    python ui.py            # -> http://localhost:8000
    python ui.py 8080       # -> http://localhost:8080

Built on the Python standard library only — no third-party installs required.
"""
from __future__ import annotations

import html
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# Ensure core is importable when run as a script.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from core.engine import TopicEmailExtractor


MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB


def parse_multipart(body: bytes, boundary: str) -> dict[str, str]:
    """Parse a multipart/form-data body into a {field-name: value} dict."""
    fields: dict[str, str] = {}
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
        name = None
        for line in header_blob.split(b"\r\n"):
            line_str = line.decode("latin-1")
            if "name=" in line_str:
                # Extract name="..."
                import re
                m = re.search(r'name="([^"]+)"', line_str)
                if m:
                    name = m.group(1)
        if name:
            fields[name] = content.decode("utf-8", errors="replace")
    return fields


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Email Extraction Engine</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }
        h1 { color: #333; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: 600; }
        input[type="text"], textarea { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
        textarea { min-height: 150px; font-family: monospace; }
        button { background: #007acc; color: white; border: none; padding: 10px 20px; border-radius: 4px; font-size: 14px; cursor: pointer; }
        button:hover { background: #005a99; }
        .result { background: #f5f5f5; padding: 15px; border-radius: 4px; margin-top: 20px; white-space: pre-wrap; font-family: monospace; font-size: 13px; }
        .error { color: #d32f2f; }
        .info { color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <h1>Email Extraction Engine</h1>
    <p class="info">Extract structured data from .eml files or raw email text for any topic.</p>

    <form method="POST" enctype="multipart/form-data">
        <div class="form-group">
            <label>Topic/Requirement (e.g., "job application", "DMARC report", "pizza order"):</label>
            <input type="text" name="topic" placeholder="Enter any topic..." required>
        </div>
        <div class="form-group">
            <label>Or paste raw email text:</label>
            <textarea name="raw_text" placeholder="Paste raw RFC-822 email text here..."></textarea>
        </div>
        <div class="form-group">
            <label>Or upload a .eml file:</label>
            <input type="file" name="eml_file" accept=".eml,.emlx,.txt">
        </div>
        <button type="submit">Extract</button>
    </form>

    {result}
</body>
</html>"""


class UIHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        page = HTML_TEMPLATE.replace("{result}", "")
        self.wfile.write(page.encode("utf-8"))

    def do_POST(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > MAX_CONTENT_LENGTH:
            self._send_error("File too large (max 10 MB)")
            return

        body = self.rfile.read(content_length)

        if "multipart/form-data" in content_type:
            boundary = content_type.split("boundary=")[1]
            fields = parse_multipart(body, boundary)
            topic = fields.get("topic", "").strip()
            raw_text = fields.get("raw_text", "").strip()
            eml_content = fields.get("eml_file", "").strip()
        elif "application/x-www-form-urlencoded" in content_type:
            from urllib.parse import parse_qs
            parsed = parse_qs(body.decode("utf-8", errors="replace"))
            topic = parsed.get("topic", [""])[0].strip()
            raw_text = parsed.get("raw_text", [""])[0].strip()
            eml_content = parsed.get("eml_content", [""])[0].strip()
        else:
            self._send_error("Unsupported content type")
            return

        if not topic:
            self._send_error("Topic is required")
            return

        # Use raw_text if provided, otherwise use file content
        eml_source = "uploaded.eml"
        if raw_text:
            result = self._extract(topic, raw_text, eml_source)
        elif eml_content:
            result = self._extract(topic, eml_content, eml_source)
        else:
            self._send_error("Please provide email text or upload a file")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        html_result = html.escape(json.dumps(result, indent=2))
        page = HTML_TEMPLATE.replace("{result}", f'<div class="result">{html_result}</div>')
        self.wfile.write(page.encode("utf-8"))

    def _extract(self, topic: str, raw_text: str, eml_source: str) -> dict:
        extractor = TopicEmailExtractor()
        result = extractor.extract(eml_source=eml_source, topic=topic, raw_text=raw_text)
        return result.to_dict()

    def _send_error(self, message: str) -> None:
        self.send_response(400)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        error_html = f'<div class="error">{html.escape(message)}</div>'
        page = HTML_TEMPLATE.replace("{result}", error_html)
        self.wfile.write(page.encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        pass  # Suppress logging


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("localhost", port), UIHandler)
    print(f"Email Extraction Engine UI running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
