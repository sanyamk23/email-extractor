"""Unit tests for classification (is_job_application + confidence)."""
from email_extractor.classifier import classify


def test_clear_application_gets_high_confidence():
    subject = "Application for Software Engineer"
    body = ("Dear Hiring Manager, please find my resume attached. "
            "I am applying for the role. Sincerely, Alex")
    is_app, score = classify(subject, body)
    assert is_app is True
    assert score >= 0.7


def test_subject_boost_matters():
    # A strong subject trigger alone should classify.
    subject = "Job Application"
    body = "Hi."
    is_app, score = classify(subject, body)
    assert is_app is True
    assert score >= 0.3


def test_non_application_low_confidence():
    subject = "Tech Weekly: Issue #42"
    body = "Welcome to this week's roundup of AI news."
    is_app, score = classify(subject, body)
    assert is_app is False
    assert score < 0.3


def test_confidence_capped_at_one():
    subject = "Application for Software Engineer"
    body = ("Applying for the role. Job application. Resume attached. "
            "Please find my resume attached.")
    is_app, score = classify(subject, body)
    assert score <= 1.0
    assert is_app is True


def test_empty_input():
    is_app, score = classify("", "")
    assert is_app is False
    assert score == 0.0


def test_resume_bonus_and_body_trigger():
    # Body-only signal with resume mention should still classify.
    subject = "Hello"
    body = ("I am excited to apply for the Senior Data Engineer position. "
            "My resume is attached.")
    is_app, score = classify(subject, body)
    assert is_app is True
    assert 0.4 <= score <= 1.0


def test_interested_in_the_role_trigger():
    # "Interested in the role" (not "position") is a strong application signal.
    subject = "Data Scientist role"
    body = ("Interested in the role. 4 years of experience. "
            "Notice: 2 weeks.")
    is_app, score = classify(subject, body)
    assert is_app is True
    assert score >= 0.3


def test_applying_for_plus_resume_subject_boost():
    # A strong subject trigger combined with a resume mention is a confident app.
    subject = "Application for Product Manager"
    body = ("Please find my resume attached. Thank you for your time "
            "and consideration.")
    is_app, score = classify(subject, body)
    assert is_app is True
    assert score >= 0.5
