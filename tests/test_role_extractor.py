"""Unit tests for job-role extraction."""
from email_extractor.role_extractor import extract_job_role


def test_direct_application_for_pattern():
    role = extract_job_role(
        "Application for Senior Frontend Developer", "")
    assert role == "Senior Frontend Developer"


def test_pattern_with_trailing_company():
    # Trailing " - Company" / " at Company" must be stripped.
    role = extract_job_role("", "Application for Software Engineer - Acme Corp")
    assert role == "Software Engineer"


def test_role_colon_pattern():
    role = extract_job_role("", "Role: Product Manager at Google.")
    assert role == "Product Manager"


def test_position_colon_pattern():
    role = extract_job_role("", "Position: Data Scientist")
    assert role == "Data Scientist"


def test_generic_the_role_pattern():
    role = extract_job_role("", "the DevOps Engineer position")
    assert role == "DevOps Engineer"


def test_dictionary_fallback():
    # No explicit pattern marker — a known role term is the only signal.
    role = extract_job_role("Interested in the position",
                            "I am a seasoned Data Analyst with 5 years of experience.")
    assert role == "Data Analyst"


def test_dictionary_fallback_returns_known_role():
    role = extract_job_role("Quick question",
                            "Do you have openings for a Data Analyst?")
    assert role == "Data Analyst"


def test_case_insensitive_match():
    role = extract_job_role("", "apply for the product manager role")
    assert role == "Product Manager"


def test_no_role_returns_none():
    role = extract_job_role("Re: meeting notes", "Talk about the weather.")
    assert role is None


def test_no_leading_article_artefact():
    role = extract_job_role("", "the role of Engineering Manager")
    assert role == "Engineering Manager"
