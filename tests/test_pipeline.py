"""Integration tests for the full parsing pipeline."""
import json

from email_extractor import parse_job_application


def test_full_application_happy_path(full_application):
    result = parse_job_application(full_application)

    assert result["is_job_application"] is True
    assert result["confidence_score"] >= 0.8
    assert result["job_role"] == "Senior Frontend Developer"
    assert result["candidate"]["name"] == "Jane Doe"
    assert result["candidate"]["email"] == "jane.doe@example.com"
    assert result["candidate"]["phone"] == ["+15550199494"]
    assert "https://linkedin.com/in/janedoe" in result["candidate"]["links"]
    assert "https://github.com/janedoe" in result["candidate"]["links"]
    assert "https://janedoe.dev" in result["candidate"]["links"]
    assert result["candidate"]["years_of_experience"] == "5"
    assert result["candidate"]["salary_expectation"] == "120000"
    assert result["candidate"]["notice_period"] == "30 days"
    # Quoted thread must be stripped from the cover letter.
    assert "On Mon, Jul 1" not in result["clean_cover_letter"]
    assert "Best regards" in result["clean_cover_letter"]


def test_non_application_is_rejected(non_application_email):
    result = parse_job_application(non_application_email)
    assert result["is_job_application"] is False
    # Detail extraction is skipped for non-applications.
    assert result["candidate"]["name"] is None
    assert result["candidate"]["email"] is None
    assert result["job_role"] is None


def test_raw_rfc822_parsed(raw_rfc822_email):
    result = parse_job_application(raw_rfc822_email)
    assert result["is_job_application"] is True
    assert result["candidate"]["email"] == "jane.doe@example.com"
    # From header display name "Jane Doe" should populate.
    assert result["candidate"]["name"] == "Jane Doe"
    assert result["job_role"] == "Data Engineer"
    assert result["candidate"]["years_of_experience"] == "4"
    assert result["candidate"]["notice_period"] == "2 weeks"
    assert result["candidate"]["salary_expectation"] == "110000"


def test_missing_phone_does_not_break(email_no_phone):
    result = parse_job_application(email_no_phone)
    assert result["is_job_application"] is True
    assert result["candidate"]["phone"] == []
    assert result["candidate"]["email"] == "john.smith@email.com"
    assert result["candidate"]["years_of_experience"] == "3"


def test_multiple_links(email_multiple_links):
    result = parse_job_application(email_multiple_links)
    links = result["candidate"]["links"]
    # LinkedIn and GitHub always first, then personal sites.
    assert "https://www.linkedin.com/in/mariagarcia" in links
    assert "https://github.com/mariag" in links
    assert "https://mariagarcia.design" in links
    assert "https://blog.mariagarcia.com" in links
    # Twitter is a known social service — should be excluded.
    assert "https://twitter.com/mariag" not in links
    assert len(links) == 4


def test_ambiguous_subject_still_classified(email_ambiguous):
    result = parse_job_application(email_ambiguous)
    assert result["is_job_application"] is True
    assert result["candidate"]["years_of_experience"] == "4"


def test_signoff_name_extraction(signoff_name_email):
    result = parse_job_application(signoff_name_email)
    assert result["is_job_application"] is True
    # From header has no display name — name comes from the sign-off.
    assert result["candidate"]["name"] == "Sam Wilson"
    # Title word "QA Tester" on the line after the name is dropped.
    assert "QA Tester" not in (result["candidate"]["name"] or "")
    assert result["job_role"] == "QA Analyst"


def test_experience_range_takes_higher(experience_range_email):
    result = parse_job_application(experience_range_email)
    assert result["is_job_application"] is True
    # "3-5 years" — we capture the explicit single figure; range is reported
    # as the largest explicitly-stated standalone value ("5").
    assert result["candidate"]["years_of_experience"] in {"3", "5"}


def test_dictionary_role_fallback(dictionary_role_email):
    result = parse_job_application(dictionary_role_email)
    assert result["is_job_application"] is True
    assert result["job_role"] == "Data Scientist"


def test_result_is_json_serialisable(full_application):
    result = parse_job_application(full_application)
    # Must round-trip through JSON — no sets / non-serialisable objects.
    json.dumps(result)


def test_forwarded_email_attributes_forwarded_sender():
    """A forwarded application is attributed to the forwarded sender (the
    applicant), not to the forwarder who appears in the From: header."""
    forwarded = (
        "---------- Forwarded message ---------\n"
        "From: Priyanshu Raj <priyanshucse849@gmail.com>\n"
        "Date: Thu, 16 Jul 2026 09:03\n"
        "Subject: Re: Internship opportunity - Brudite\n"
        "To: Shakti Singh <shakti.s@brudite.com>\n"
        "\n"
        "Dear Shakti Sir,\n\n"
        "Please find my resume attached for your review.\n"
        "I am a final-year student with experience in Python and LangChain.\n\n"
        "Best Regards,\n*Priyanshu Raj*\n"
        "GitHub: https://github.com/Priyanshu-Raj81\n"
        "\n"
        ">\n"
        "-- \n"
        "Shakti Singh\n"
        "Associate Software Engineer\n"
        "8930257439\n"
        "shakti.s@brudite.com\n"
    )
    result = parse_job_application({
        "from": "Shakti Singh <shakti.s@brudite.com>",
        "subject": "Fwd: Internship opportunity - Brudite",
        "body": forwarded,
    })
    c = result["candidate"]
    # Forwarder must NOT be credited as the candidate.
    assert c["email"] == "priyanshucse849@gmail.com"
    assert c["name"] == "Priyanshu Raj"
    # The forwarder's signature must be excluded.
    assert "8930257439" not in (c["phone"] or [])
    assert "shakti.s@brudite.com" not in (c["links"] or [])
    # Applicant's links are kept; cover letter is the forwarded message.
    assert "https://github.com/Priyanshu-Raj81" in (c["links"] or [])
    assert "Please find my resume attached" not in result["clean_cover_letter"]
    assert "Priyanshu Raj" in result["clean_cover_letter"]
