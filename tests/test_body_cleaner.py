"""Unit tests for clean cover-letter extraction."""
from email_extractor.body_cleaner import clean_cover_letter


def test_strips_forwarded_thread():
    body = (
        "Hi there.\n\n"
        "I am excited to apply for the Software Engineer role.\n\n"
        "----Original Message----\n"
        "From: recruiter@acme.com\n"
        "Sent: Monday, July 1, 2026 9:00 AM\n"
        "Subject: Software Engineer\n"
        "Jane, thanks for applying!\n"
    )
    cleaned = clean_cover_letter(body)
    assert "----Original Message----" not in cleaned
    assert "From: recruiter@acme.com" not in cleaned
    assert "Sent: Monday" not in cleaned
    assert "I am excited to apply" in cleaned


def test_preserves_forwarded_message_body():
    # A forwarded message's envelope (separator + From/Date/Subject/To headers)
    # is stripped, but the forwarded body itself is preserved.
    body = ("Hello.\n\nThanks for considering my application.\n\n"
            "------ Forwarded Message ------\n"
            "From: a@b.com\n"
            "Date: Tue, 1 Jan 2025\n"
            "Subject: Re: role\n"
            "To: me@my.com\n"
            "\n"
            "Forwarded content that should be preserved.")
    cleaned = clean_cover_letter(body)
    assert "Forwarded Message" not in cleaned
    assert "From: a@b.com" not in cleaned
    assert "Subject: Re: role" not in cleaned
    assert "Thanks for considering" in cleaned
    assert "Forwarded content that should be preserved." in cleaned


def test_strips_signature_block():
    body = "Best,\nJane Doe\n--\nJane Doe\nEngineer\nAcme Corp"
    cleaned = clean_cover_letter(body)
    assert "Engineer" not in cleaned
    assert "Acme Corp" not in cleaned
    assert "Best," in cleaned


def test_strips_attachment_boilerplate():
    body = ("Hi.\n\nPlease find my resume attached for your review.\n\n"
            "I look forward to hearing from you.\nSincerely,\nJane")
    cleaned = clean_cover_letter(body)
    assert "Please find my resume attached" not in cleaned


def test_strips_thank_you_closing():
    body = ("I would love to join your team.\n\n"
            "Thank you for your time and consideration.")
    cleaned = clean_cover_letter(body)
    assert "Thank you for your time" not in cleaned
    assert "I would love to join your team." in cleaned


def test_retains_greeting_and_pitch():
    body = "Dear Hiring Manager,\n\nI am writing to express interest.\n\nBest,\nA"
    cleaned = clean_cover_letter(body)
    assert "Dear Hiring Manager" in cleaned
    assert "I am writing to express interest" in cleaned
    assert "Best," in cleaned


def test_cleans_whitespace():
    body = "Hello.\n\n\n\n\nWorld.\n   "
    cleaned = clean_cover_letter(body)
    assert "\n\n\n" not in cleaned


def test_strips_on_wrote_thread():
    body = ("Hi.\n\nMy cover letter.\n\n"
            "On Mon, Jul 1, 2026 John wrote:\n> quoted text")
    cleaned = clean_cover_letter(body)
    assert "On Mon, Jul 1" not in cleaned
    assert "quoted text" not in cleaned
    assert "My cover letter" in cleaned
