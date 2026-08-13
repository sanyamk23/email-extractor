#!/usr/bin/env python3
"""Quick verification script for Gmail email parsing."""
import sys, json
sys.path.insert(0, '.')
from core.engine import TopicEmailExtractor

extractor = TopicEmailExtractor()
result = extractor.extract(
    eml_source='/Users/sanya/Documents/project/email_extractor/mails/Application for UX Designer Role - Priya Patel.eml',
    topic='job application'
)
with open('/tmp/gmail_result.json', 'w') as f:
    json.dump(result.to_dict(), f, indent=2)
print("Result written to /tmp/gmail_result.json")
