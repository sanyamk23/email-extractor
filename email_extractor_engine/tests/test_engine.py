"""Unit tests for the Email Extraction Engine.

Verifies the two-tier field-resolution architecture:

- **Manual registry** (``core.templates``): known topics like "job application",
  "dmarc report", "invoice" resolve to pre-defined field lists
  (``field_source == "manual_registry"``).
- **Dynamic discovery** (``core.dynamic_schema``): arbitrary topics fall back
  to deriving fields from key-value pairs, noun phrases, and topic-token
  inference (``field_source == "dynamic_discovery"``).

Tests also cover the regex-based fallback when GLiNER/sentence-transformers
are unavailable.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap

import pytest

# Ensure the package root is on sys.path so ``core`` is importable.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from core.engine import TopicEmailExtractor, ExtractionResult


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def extractor() -> TopicEmailExtractor:
    return TopicEmailExtractor()


@pytest.fixture
def job_application_email() -> str:
    return textwrap.dedent("""\
        From: Sarah Jenkins <s.jenkins@email.com>
        To: hr@techcompany.com
        Subject: Job Application - Senior Python Engineer

        Hi HR, applying for the Senior Python Engineer role. Name: Sarah Jenkins (s.jenkins@email.com). 6 years experience, B.S. in Computer Science from MIT. Expected salary: $130,000/yr.
    """)


@pytest.fixture
def dmarc_report_email() -> str:
    return textwrap.dedent("""\
        From: dmarc-reports@google.com
        To: security@enterprise.com
        Subject: DMARC Aggregate Report for enterprise.com

        DMARC Aggregate Report for enterprise.com. Period: 2026-08-01 to 2026-08-07. Submitter: dmarc-reports@google.com. Source IP: 192.0.2.1. Total messages: 1420. Passed: 1400. Failures: 20. Policy: reject.
    """)


@pytest.fixture
def vendor_inspection_email() -> str:
    return textwrap.dedent("""\
        From: quality@auditor.com
        To: vendor@mfg.com
        Subject: Vendor Inspection Report

        Vendor Quality Check Report #8841. Auditor: Mark Vance. Facility: Detroit Plant 4. Status: CONDITIONAL PASS. Defects Found: 3 minor paint scratches.
    """)


@pytest.fixture
def invoice_email() -> str:
    return textwrap.dedent("""\
        From: billing@supplier.com
        To: accounting@buyer.com
        Subject: Invoice INV-2026-0042

        Invoice Number: INV-2026-0042. Invoice Date: 2026-08-05. Due Date: 2026-09-05. Total Amount: $5,400.00. Vendor: Supplier Corp. Customer: Buyer Inc. Payment Method: Bank Transfer. Payment Status: Pending.
    """)


@pytest.fixture
def forwarded_job_application_email() -> str:
    return textwrap.dedent("""\
        From: hr@recruiter.com
        To: hiring-manager@company.com
        Subject: Fwd: Job Application - Data Analyst

        ---------- Forwarded Message ----------
        From: Alex Chen <alex.chen@example.com>
        Date: Mon, Aug 10, 2026 at 10:30 AM
        Subject: Application for Data Analyst
        To: hr@recruiter.com

        Dear Hiring Manager,

        I am writing to apply for the Data Analyst position. My email is alex.chen@example.com and I have 3 years of experience in data analysis. I hold a B.A. in Statistics from UC Berkeley. My expected salary is $85,000.

        Best regards,
        Alex Chen
    """)


# ── Two-Tier Resolution Tests ────────────────────────────────────────────────────

class TestTwoTierResolution:
    """Tests for the two-tier field-resolution: registry for known topics,
    dynamic discovery for arbitrary topics."""

    def test_job_application_extraction(self, extractor, job_application_email):
        result = extractor.extract(eml_source="job.eml", topic="job application",
                                   raw_text=job_application_email)
        assert result.field_source == "manual_registry"
        assert result.topic == "job application"
        data = result.extracted_data
        assert data["candidate_name"] == "Sarah Jenkins"
        assert data["expected_salary"] is not None

    def test_job_application_alias_resolves(self, extractor, job_application_email):
        """Topic aliases resolve to the canonical topic and manual registry."""
        result = extractor.extract(eml_source="job.eml", topic="career application",
                                   raw_text=job_application_email)
        assert result.field_source == "manual_registry"
        assert result.topic == "job application"

    def test_dmarc_report_extraction(self, extractor, dmarc_report_email):
        result = extractor.extract(eml_source="dmarc.eml", topic="dmarc report",
                                   raw_text=dmarc_report_email)
        assert result.field_source == "manual_registry"
        data = result.extracted_data
        assert data["submitter_email"] == "dmarc-reports@google.com"
        assert data["source_ip"] == "192.0.2.1"
        assert data["total_messages"] is not None
        assert data["dmarc_policy"] == "reject"

    def test_invoice_extraction(self, extractor, invoice_email):
        result = extractor.extract(eml_source="invoice.eml", topic="invoice",
                                   raw_text=invoice_email)
        assert result.field_source == "manual_registry"
        data = result.extracted_data
        assert data["invoice_number"] == "INV-2026-0042"
        assert data["total_amount"] is not None
        assert data["invoice_date"] is not None

    def test_forwarded_email_sender_fallback(self, extractor, forwarded_job_application_email):
        """Forwarded emails should use the forwarded sender, not the forwarder."""
        result = extractor.extract(eml_source="fwd.eml", topic="job application",
                                   raw_text=forwarded_job_application_email)
        assert result.field_source == "manual_registry"
        # The forwarded sender should be used for sender info
        assert result.sender["email"] == "alex.chen@example.com"

        # Sender fallback: when body extraction fails for candidate fields,
        # the forwarded sender's name/email should be used.
        data = result.extracted_data
        assert data.get("candidate_name") == "Alex Chen"

    def test_vendor_inspection_dynamic(self, extractor, vendor_inspection_email):
        result = extractor.extract(eml_source="inspection.eml", topic="vendor inspection",
                                   raw_text=vendor_inspection_email)
        assert result.field_source == "dynamic_discovery"
        assert result.topic == "vendor inspection"
        data = result.extracted_data
        # KV-pair labels should be discovered
        assert data.get("auditor") == "Mark Vance"
        assert data.get("facility") == "Detroit Plant 4"
        assert data.get("status") == "CONDITIONAL PASS"

    def test_dynamic_with_empty_body(self, extractor):
        """Dynamic discovery should still return fields even with empty body."""
        result = extractor.extract(eml_source="test.eml", topic="quarterly audit report",
                                   raw_text="From: test@test.com\nSubject: Test\n\n")
        assert result.field_source == "dynamic_discovery"
        assert len(result.extracted_data) > 0

    def test_dynamic_field_count_capped(self, extractor):
        """Dynamic discovery should cap the number of fields."""
        body = "Key1: val1. Key2: val2. Key3: val3. Key4: val4. Key5: val5. " * 5
        result = extractor.extract(eml_source="test.eml", topic="random topic xyz",
                                   raw_text=f"From: test@test.com\nSubject: Test\n\n{body}")
        assert result.field_source == "dynamic_discovery"
        assert len(result.extracted_data) <= 20  # MAX_FIELDS cap

    def test_arbitrary_topic_with_kv_pairs(self, extractor):
        """Engine should handle completely arbitrary topics by discovering KV pairs."""
        result = extractor.extract(
            eml_source="test.eml",
            topic="quantum flux capacitor maintenance",
            raw_text="From: doc@timetravel.com\nSubject: Maintenance\n\n"
                     "Technician: Dr. Emmett Brown. Part number: FC-8841. "
                     "Status: operational. Date: 2026-08-15."
        )
        assert result.field_source == "dynamic_discovery"
        data = result.extracted_data
        assert data.get("technician") == "Dr. Emmett Brown"
        assert data.get("part_number") == "FC-8841"
        assert data.get("status") == "operational"
        assert data.get("date") == "2026-08-15"

    def test_empty_topic_discovers_kv_pairs(self, extractor):
        """Empty topic should still discover KV-pair labels from the body."""
        result = extractor.extract(
            eml_source="test.eml",
            topic="",
            raw_text="From: test@test.com\nSubject: Test\n\n"
                     "Name: Bob. Email: bob@test.com. Date: 2026-08-10."
        )
        assert result.field_source == "dynamic_discovery"
        data = result.extracted_data
        assert data.get("name") == "Bob"
        assert data.get("email") == "bob@test.com"
        assert data.get("date") == "2026-08-10"

    def test_single_word_arbitrary_topic(self, extractor):
        """Single arbitrary word topic should still work."""
        result = extractor.extract(
            eml_source="test.eml",
            topic="flibbertigibbet",
            raw_text="From: test@test.com\nSubject: Test\n\n"
                     "Contact: Alice Wonder. Value: 42. Date: 2026-08-10."
        )
        assert result.field_source == "dynamic_discovery"
        data = result.extracted_data
        assert data.get("contact") == "Alice Wonder"
        assert data.get("date") == "2026-08-10"

    def test_topic_with_numbers_and_symbols(self, extractor):
        """Topic with numbers and symbols should not crash."""
        result = extractor.extract(
            eml_source="test.eml", topic="Café résumé review 2026!",
            raw_text="From: test@test.com\nSubject: Test\n\n"
                     "Candidate: Jose Garcia. Score: 85. Date: 2026-08-20."
        )
        assert result.field_source == "dynamic_discovery"

    def test_topic_with_no_kv_pairs(self, extractor):
        """Topic with body containing no KV pairs should still return fields."""
        result = extractor.extract(
            eml_source="test.eml",
            topic="random topic with no keywords",
            raw_text="From: test@test.com\nSubject: Test\n\n"
                     "This is just a plain text body with no structured data."
        )
        assert result.field_source == "dynamic_discovery"
        # Should still return a minimal set of placeholder fields
        assert len(result.extracted_data) > 0

    def test_registered_topics_use_manual_registry(self, extractor):
        """All topics that have a manual registry entry should use manual_registry."""
        topics = ["job application", "dmarc report", "invoice", "e-commerce order",
                   "event invitation", "travel itinerary", "support ticket",
                   "meeting minutes", "purchase order", "delivery notice",
                   "interview scheduling", "contract", "newsletter"]
        for topic in topics:
            result = extractor.extract(
                eml_source="test.eml", topic=topic,
                raw_text="From: test@test.com\nSubject: Test\n\nContact: Alice."
            )
            assert result.field_source == "manual_registry", \
                f"Topic '{topic}' should use manual_registry, got {result.field_source}"

    def test_unregistered_topics_use_dynamic_discovery(self, extractor):
        """Topics without a manual registry entry fall back to dynamic discovery."""
        topics = ["vendor inspection", "quarterly audit report", "random topic xyz",
                  "flibbertigibbet"]
        for topic in topics:
            result = extractor.extract(
                eml_source="test.eml", topic=topic,
                raw_text="From: test@test.com\nSubject: Test\n\nContact: Alice."
            )
            assert result.field_source == "dynamic_discovery", \
                f"Topic '{topic}' should use dynamic_discovery, got {result.field_source}"


# ── Regex Fallback Tests ────────────────────────────────────────────────────────

class TestRegexFallback:
    """Tests for the regex-based fallback when GLiNER is unavailable."""

    def test_email_extraction(self, extractor):
        # Registry field "applicant_email" — "email" alias triggers email extraction
        body = "Email: john.doe@example.com for details."
        result = extractor.extract(eml_source="test.eml", topic="job application",
                                   raw_text=f"From: test@test.com\nSubject: Test\n\n{body}")
        assert result.extracted_data.get("applicant_email") is not None

    def test_phone_extraction(self, extractor):
        # Registry field "phone_number" — "phone" alias triggers phone extraction
        body = "Phone: Call me at (555) 123-4567 tomorrow."
        result = extractor.extract(eml_source="test.eml", topic="job application",
                                   raw_text=f"From: test@test.com\nSubject: Test\n\n{body}")
        assert result.extracted_data.get("phone_number") is not None

    def test_ip_extraction(self, extractor):
        # KV pair "Source IP:" is discovered as field "source_ip"
        body = "Source IP: 192.168.1.1. The report is complete."
        result = extractor.extract(eml_source="test.eml", topic="dmarc report",
                                   raw_text=f"From: test@test.com\nSubject: Test\n\n{body}")
        assert result.extracted_data.get("source_ip") == "192.168.1.1"

    def test_money_extraction(self, extractor):
        # KV pair "Total amount:" is discovered as field "total_amount"
        body = "Total amount: $5,400.00. Subtotal: $4,500."
        result = extractor.extract(eml_source="test.eml", topic="invoice",
                                   raw_text=f"From: test@test.com\nSubject: Test\n\n{body}")
        assert result.extracted_data.get("total_amount") is not None

    def test_date_extraction(self, extractor):
        # KV pair "Invoice Date:" is discovered as field "invoice_date"
        body = "Invoice Date: 2026-08-05. Due Date: 2026-09-05."
        result = extractor.extract(eml_source="test.eml", topic="invoice",
                                   raw_text=f"From: test@test.com\nSubject: Test\n\n{body}")
        assert result.extracted_data.get("invoice_date") is not None

    def test_kv_extraction(self, extractor):
        # Registry field "ticket_status" — "status" alias triggers KV extraction
        body = "Status: APPROVED. Priority: HIGH. Owner: Jane Smith."
        result = extractor.extract(eml_source="test.eml", topic="support ticket",
                                   raw_text=f"From: test@test.com\nSubject: Test\n\n{body}")
        assert result.extracted_data.get("ticket_status") == "APPROVED"

    def test_experience_extraction(self, extractor):
        # Registry field "years_experience" — KV "Years:" → "years" alias
        body = "Years: I have 5 years of experience in software development."
        result = extractor.extract(eml_source="test.eml", topic="job application",
                                   raw_text=f"From: test@test.com\nSubject: Test\n\n{body}")
        assert result.extracted_data.get("years_experience") is not None

    def test_role_prose_extraction(self, extractor):
        # Registry field "target_role" — KV "Role:" → "role" alias
        body = "Role: I am applying for the Senior Python Engineer position."
        result = extractor.extract(eml_source="test.eml", topic="job application",
                                   raw_text=f"From: test@test.com\nSubject: Test\n\n{body}")
        assert result.extracted_data.get("target_role") is not None

    def test_signoff_name_extraction(self, extractor):
        # Registry field "candidate_name" — KV "Name:" + sign-off regex
        body = "Name: John Doe. Thanks,\nJohn Doe\nSenior Developer"
        result = extractor.extract(eml_source="test.eml", topic="job application",
                                   raw_text=f"From: test@test.com\nSubject: Test\n\n{body}")
        assert result.extracted_data.get("candidate_name") == "John Doe"

    def test_education_extraction(self, extractor):
        # KV pair "Education:" is discovered as field "education"
        body = "Education: B.S. in Computer Science from MIT. Also M.B.A. from Harvard."
        result = extractor.extract(eml_source="test.eml", topic="job application",
                                   raw_text=f"From: test@test.com\nSubject: Test\n\n{body}")
        assert result.extracted_data.get("education") is not None


# ── EML File Parsing Tests ──────────────────────────────────────────────────────

class TestEmlParsing:
    """Tests for EML file loading and parsing."""

    def test_parse_eml_file(self, tmp_path):
        eml_content = textwrap.dedent("""\
            From: sender@example.com
            To: recipient@example.com
            Subject: Test Email
            Date: Mon, Aug 10, 2026 at 10:30 AM

            This is the email body. Name: John Doe. Email: john.doe@example.com.
        """)
        eml_file = tmp_path / "test.eml"
        eml_file.write_text(eml_content)
        result = TopicEmailExtractor().extract(eml_source=str(eml_file), topic="job application")
        assert result.topic == "job application"
        assert result.sender["email"] == "sender@example.com"

    def test_parse_raw_email_with_headers(self):
        raw = textwrap.dedent("""\
            From: sender@example.com
            To: recipient@example.com
            Subject: Test

            Body text here.
        """)
        result = TopicEmailExtractor().extract(eml_source="inline.eml", topic="job application",
                                               raw_text=raw)
        assert result.subject == "Test"
        assert result.sender["email"] == "sender@example.com"

    def test_parse_plain_body_no_headers(self):
        raw = "Just a plain body. Name: Alice Smith. Done."
        result = TopicEmailExtractor().extract(eml_source="plain.eml", topic="job application",
                                               raw_text=raw)
        assert result.extracted_data.get("candidate_name") == "Alice Smith"

    def test_gmail_email_with_delivered_to_header(self, tmp_path):
        """Emails starting with Delivered-To (Gmail/GSuite) should parse correctly."""
        eml_content = textwrap.dedent("""\
            Delivered-To: user@example.com
            Received: by 2002:a0c:e9c7:0:b0:8da:e324:b7c0 with SMTP id q7csp123
            From: sender@example.com
            To: user@example.com
            Subject: Gmail-style Email
            Date: Mon, Aug 10, 2026 at 12:00 PM

            This is the email body. Name: Alice Smith.
        """)
        eml_file = tmp_path / "gmail.eml"
        eml_file.write_text(eml_content)
        result = TopicEmailExtractor().extract(
            eml_source=str(eml_file), topic="job application"
        )
        assert result.subject == "Gmail-style Email"
        assert result.sender["email"] == "sender@example.com"
        assert result.extracted_data.get("candidate_name") == "Alice Smith"

    def test_gmail_email_full_extraction(self):
        """Full extraction from a Gmail-sent email starting with Delivered-To.

        Gmail emails begin with ``Delivered-To:`` rather than ``From:``, so
        the parser must recognise ``delivered-to`` as a known RFC-822 header
        to route the content through proper email parsing (not body-only).
        """
        eml_path = '/Users/sanya/Documents/project/email_extractor/mails/Application for UX Designer Role - Priya Patel.eml'
        if not os.path.exists(eml_path):
            pytest.skip("Gmail test email not available")

        from core.eml_parser import parse_eml
        email = parse_eml(eml_path)
        # Headers must be parsed correctly, not treated as body text.
        assert email.subject == "Application for UX Designer Role - Priya Patel"
        assert "developer.shaktisingh" in email.from_header
        assert len(email.body) > 0

        result = TopicEmailExtractor().extract(
            eml_source=eml_path, topic='job application'
        )
        assert result.field_source == "manual_registry"
        assert result.subject == "Application for UX Designer Role - Priya Patel"
        assert result.sender["email"] == "developer.shaktisingh@gmail.com"
        # The body has no KV pairs, but topic-driven field discovery + GLiNER
        # should still surface real fields (candidate_name, target_role, etc.)
        assert len(result.extracted_data) > 0
        assert result.extracted_data.get("candidate_name") is not None
        assert result.extracted_data.get("target_role") is not None

    def test_attachments_extracted(self, tmp_path):
        eml_content = textwrap.dedent("""\
            From: sender@example.com
            Subject: Test
            MIME-Version: 1.0
            Content-Type: multipart/mixed; boundary="boundary"

            --boundary
            Content-Type: text/plain

            Body text.
            --boundary
            Content-Type: application/pdf
            Content-Disposition: attachment; filename="doc.pdf"

            JVBERi0xLjQKJcOkw7zDtsO...
            --boundary--
        """)
        eml_file = tmp_path / "test.eml"
        eml_file.write_text(eml_content)
        result = TopicEmailExtractor().extract(eml_source=str(eml_file), topic="job application")
        assert len(result.attachments) >= 1


# ── Result Structure Tests ─────────────────────────────────────────────────────

class TestResultStructure:
    """Tests for the ExtractionResult dataclass."""

    def test_to_dict(self, extractor, job_application_email):
        result = extractor.extract(eml_source="job.eml", topic="job application",
                                   raw_text=job_application_email)
        d = result.to_dict()
        assert "topic" in d
        assert "field_source" in d
        assert "extracted_data" in d
        assert "sender" in d
        assert "subject" in d
        assert "attachments" in d

    def test_to_json(self, extractor, job_application_email):
        result = extractor.extract(eml_source="job.eml", topic="job application",
                                   raw_text=job_application_email)
        json_str = result.to_json()
        d = json.loads(json_str)
        assert d["topic"] == "job application"
        assert d["field_source"] == "manual_registry"

    def test_callable_interface(self, extractor, job_application_email):
        d = extractor(eml_source="job.eml", topic="job application",
                      raw_text=job_application_email)
        assert isinstance(d, dict)
        assert d["field_source"] == "manual_registry"

    def test_batch_extraction(self, extractor, job_application_email, dmarc_report_email):
        results = extractor.extract_batch(
            eml_sources=["job.eml", "dmarc.eml"],
            topic="job application",
            raw_texts=[job_application_email, dmarc_report_email],
        )
        assert len(results) == 2
        assert all(isinstance(r, ExtractionResult) for r in results)

    def test_batch_with_error(self, extractor):
        """Batch should continue even if one item fails."""
        results = extractor.extract_batch(
            eml_sources=["nonexistent_file.eml"],
            topic="job application",
        )
        assert len(results) == 1
        assert results[0].field_source == "error"


# ── Robustness Tests ─────────────────────────────────────────────────────────────

class TestRobustness:
    """Tests ensuring the engine never crashes on arbitrary inputs."""

    def test_none_topic_with_body(self, extractor):
        """None topic should trigger dynamic discovery, not crash."""
        result = extractor.extract(
            eml_source="test.eml", topic=None,
            raw_text="From: test@test.com\nSubject: Test\n\nName: Bob. Value: 42."
        )
        assert result.field_source == "dynamic_discovery"
        assert len(result.extracted_data) > 0

    def test_empty_topic_with_body(self, extractor):
        """Empty topic should trigger dynamic discovery, not crash."""
        result = extractor.extract(
            eml_source="test.eml", topic="",
            raw_text="From: test@test.com\nSubject: Test\n\nName: Bob."
        )
        assert result.field_source == "dynamic_discovery"
        assert len(result.extracted_data) > 0

    def test_whitespace_topic(self, extractor):
        """Whitespace-only topic should trigger dynamic discovery."""
        result = extractor.extract(
            eml_source="test.eml", topic="   \t  ",
            raw_text="From: test@test.com\nSubject: Test\n\nName: Bob."
        )
        assert result.field_source == "dynamic_discovery"

    def test_none_eml_source_and_raw_text(self, extractor):
        """No input at all should return empty result, not crash."""
        result = extractor.extract(eml_source=None, topic="job application",
                                   raw_text=None)
        assert len(result.extracted_data) > 0

    def test_unicode_topic(self, extractor):
        """Unicode topic should not crash."""
        result = extractor.extract(
            eml_source="test.eml", topic="αβγ δελτα",
            raw_text="From: test@test.com\nSubject: Test\n\nCandidate: Jose. Score: 85."
        )
        assert result.field_source == "dynamic_discovery"

    def test_symbol_topic(self, extractor):
        """Symbol-only topic should not crash."""
        result = extractor.extract(
            eml_source="test.eml", topic="!!!???",
            raw_text="From: test@test.com\nSubject: Test\n\nContact: Alice."
        )
        assert result.field_source == "dynamic_discovery"

    def test_body_starting_with_kv_pair(self, extractor):
        """Body starting with 'Key: value' should be treated as body, not headers."""
        result = extractor.extract(
            eml_source="test.eml", topic="nonsense topic",
            raw_text="Technician: Dr. Emmett Brown. Part: FC-8841. Status: operational."
        )
        assert result.field_source == "dynamic_discovery"
        # The KV labels should be discovered and extracted
        assert "technician" in result.extracted_data
        assert result.extracted_data["technician"] == "Dr. Emmett Brown"

    def test_body_starting_with_kv_not_treated_as_header(self):
        """parse_raw_email should not treat body KV pairs as RFC-822 headers."""
        from core.eml_parser import parse_raw_email
        raw = "Name: Alice Smith. Role: Engineer. Date: 2026-08-15."
        email = parse_raw_email(raw)
        assert email.body == raw
        assert email.extra_headers == {}

    def test_very_long_topic(self, extractor):
        """Extremely long topic should not crash."""
        result = extractor.extract(
            eml_source="test.eml", topic="X" * 500,
            raw_text="From: test@test.com\nSubject: Test\n\nName: Test User."
        )
        assert len(result.extracted_data) > 0

    def test_null_byte_topic(self, extractor):
        """Topic with null bytes should not crash."""
        result = extractor.extract(
            eml_source="test.eml", topic="topic\x00with\x00nulls",
            raw_text="From: test@test.com\nSubject: Test\n\nName: Test User."
        )
        assert len(result.extracted_data) > 0

    def test_arbitrary_topic_with_realistic_body(self, extractor):
        """Any arbitrary topic with realistic KV body should extract KV pairs."""
        result = extractor.extract(
            eml_source="test.eml", topic="supercalifragilisticexpialidocious",
            raw_text="From: test@test.com\nSubject: Test\n\n"
                     "Inspector: Jane Doe. Facility: Site B. Result: Pass. Notes: All clear."
        )
        assert result.field_source == "dynamic_discovery"
        # KV-pair labels should be discovered and their values extracted.
        assert result.extracted_data.get("inspector") == "Jane Doe"
        # "Result" → field "result", value should be "Pass".
        assert any(
            v == "Pass" for v in result.extracted_data.values()
            if v is not None
        )

    def test_batch_continues_after_error(self, extractor):
        """Batch should return error entry for failed items, continue with rest."""
        results = extractor.extract_batch(
            eml_sources=["nonexistent.eml", "also_missing.eml"],
            topic="job application",
        )
        assert len(results) == 2
        assert all(r.field_source == "error" for r in results)
        assert all("error" in r.extracted_data for r in results)
