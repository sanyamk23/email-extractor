"""Unit tests for candidate-detail extraction."""
from email_extractor.candidate_extractor import (
    extract_name, extract_email, extract_phones, extract_links,
    extract_years_of_experience, extract_salary, extract_notice_period,
    extract_candidate, extract_skills, extract_education,
    extract_seniority, extract_location, extract_company, extract_sender,
    extract_start_date, extract_work_type, extract_languages,
    extract_certifications,
)
from email_extractor import parse_job_application
from pathlib import Path


# ── Name ────────────────────────────────────────────────────────────────────
def test_name_from_from_header():
    assert extract_name('"Jane Doe" <jane@example.com>', "") == "Jane Doe"


def test_name_from_from_header_bare_email():
    assert extract_name("jane.doe@example.com", "") is None


def test_name_from_signoff():
    body = "Hi,\n\nThanks for reviewing.\n\nThanks,\nSam Wilson"
    assert extract_name("", body) == "Sam Wilson"


def test_signoff_name_drops_trailing_title():
    body = "Hello.\n\nBest regards,\nJane Doe\nSenior Engineer\nAcme Corp"
    # The title "Senior Engineer" sits on the next line and must not be joined.
    assert extract_name("", body) == "Jane Doe"


# ── Email ───────────────────────────────────────────────────────────────────
def test_email_from_header():
    assert extract_email('"Jane Doe" <jane.doe@example.com>', "") == \
        "jane.doe@example.com"


def test_email_fallback_to_body():
    assert extract_email("no-email-here", "Reach me at bob@test.org") == \
        "bob@test.org"


def test_email_returns_none_when_absent():
    assert extract_email("nobody<someone.io>", "no email here") is None


# ── Phone ───────────────────────────────────────────────────────────────────
def test_phone_international_dash():
    phones = extract_phones("Call +1-555-019-9494 tomorrow")
    assert phones == ["+15550199494"]


def test_phone_parentheses_us():
    phones = extract_phones("Office (555) 019-9494 or 5550199494")
    # Both forms normalise to the same number and are de-duplicated.
    assert phones.count("+15550199494") == 1


def test_phone_no_false_positive_on_years():
    # "2026" and "5 years" must not become phone numbers.
    phones = extract_phones("The year is 2026 and I have 5 years of experience")
    assert phones == []


def test_phone_dedup_normalised():
    phones = extract_phones("+1 555-019-9494 and +1-555-019-9494")
    assert phones == ["+15550199494"]


def test_phone_none_when_absent():
    assert extract_phones("No contact details here at all") == []


# ── Links ───────────────────────────────────────────────────────────────────
def test_links_linkedin_github_portfolio():
    body = ("LinkedIn https://linkedin.com/in/alice\n"
            "GitHub https://github.com/alice\n"
            "Site https://alice.dev")
    links = extract_links(body)
    assert "https://linkedin.com/in/alice" in links
    assert "https://github.com/alice" in links
    assert "https://alice.dev" in links


def test_links_exclude_social_services():
    body = "https://twitter.com/alice and https://linkedin.com/in/alice"
    links = extract_links(body)
    assert "https://linkedin.com/in/alice" in links
    assert "https://twitter.com/alice" not in links


def test_links_dedup():
    body = "https://github.com/bob https://github.com/bob https://bob.io"
    links = extract_links(body)
    assert links.count("https://github.com/bob") == 1
    assert "https://bob.io" in links


# ── Years of experience ─────────────────────────────────────────────────────
def test_years_simple():
    assert extract_years_of_experience("5 years of experience") == "5"


def test_years_decimal():
    assert extract_years_of_experience("3.5 years of experience") == "3.5"


def test_years_falls_back_to_other_pattern():
    assert extract_years_of_experience("over 7 years in the field") == "7"


def test_years_picks_largest_explicit():
    assert extract_years_of_experience("5 years of experience, 3 years coding") == "5"


def test_years_none_when_absent():
    assert extract_years_of_experience("No experience mentioned.") is None


# ── Salary ──────────────────────────────────────────────────────────────────
def test_salary_currency():
    assert extract_salary("Salary expectation: $120,000 per year") == "120000"


def test_salary_no_currency():
    assert extract_salary("compensation 95000 annually") == "95000"


def test_salary_none_without_context():
    # A bare number with no salary keyword is ignored.
    assert extract_salary("I charge 200 dollars per hour for freelancing") is None


# ── Notice period ───────────────────────────────────────────────────────────
def test_notice_period_days():
    assert extract_notice_period("Notice period: 30 days") == "30 days"


def test_notice_period_weeks():
    assert extract_notice_period("available in 2 weeks") == "2 weeks"


def test_notice_period_none_when_absent():
    assert extract_notice_period("Started last month") is None


# ── Full candidate dict ─────────────────────────────────────────────────────
def test_extract_candidate_full():
    from_header = '"Jane Doe" <jane.doe@example.com>'
    body = ("5 years of experience. Salary: $120,000. "
            "Notice period 30 days. Phone +1-555-019-9494. "
            "https://github.com/janedoe")
    details = extract_candidate(from_header, body)
    assert details["name"] == "Jane Doe"
    assert details["email"] == "jane.doe@example.com"
    assert details["years_of_experience"] == "5"
    assert details["salary_expectation"] == "120000"
    assert details["notice_period"] == "30 days"
    assert details["phone"] == ["+15550199494"]
    assert "https://github.com/janedoe" in details["links"]


# ── Markdown-tolerated sign-offs ─────────────────────────────────────────────
def test_signoff_name_with_markdown_bold():
    # Some clients wrap the name in markdown bold/italic markers (*Name*).
    from_header = "a@b.com"
    body = "Thanks for considering.\n\nBest regards,\n*Priyanshu Raj*"
    assert extract_name(from_header, body) == "Priyanshu Raj"


def test_signoff_name_plain_unchanged():
    from_header = "a@b.com"
    body = "Thank you.\n\nRegards,\nJane Doe"
    assert extract_name(from_header, body) == "Jane Doe"


# ── Proxy/forwarder From (real applicant signed the body) ────────────────────
def test_name_prefers_signoff_when_from_is_proxy():
    # The From header is a forwarder ("Shakti Singh"); the body is signed by
    # the real candidate ("Alex Kumar"). The From name never appears in body.
    from_header = "Shakti Singh <developer.shaktisingh@gmail.com>"
    body = ("Hello,\n\nI am excited to submit my application for the "
            "Data Analyst position.\n\nRegards,\nAlex Kumar")
    assert extract_name(from_header, body) == "Alex Kumar"


def test_email_not_attributed_to_proxy_forwarder():
    # The forwarder's address must not become the candidate's email.
    from_header = "Shakti Singh <developer.shaktisingh@gmail.com>"
    body = "Application for Data Analyst.\n\nRegards,\nAlex Kumar"
    assert extract_email(from_header, body) is None


def test_email_uses_body_address_when_from_is_proxy():
    # When the From header is a proxy, the candidate's own body email wins.
    from_header = "Shakti Singh <developer.shaktisingh@gmail.com>"
    body = "Reach me at alex.kumar@example.com\n\nRegards,\nAlex Kumar"
    assert extract_email(from_header, body) == "alex.kumar@example.com"


# ── Sender (message sender, distinct from candidate) ────────────────────────
def test_extract_sender_parses_from_header():
    sender = extract_sender('"Jane Doe" <jane.doe@example.com>')
    assert sender == {"name": "Jane Doe", "email": "jane.doe@example.com"}


def test_extract_sender_bare_email():
    assert extract_sender("jane.doe@example.com") == {
        "name": None, "email": "jane.doe@example.com"}


def test_extract_sender_none_when_empty():
    assert extract_sender("") == {"name": None, "email": None}


# ── Skills ────────────────────────────────────────────────────────────────────
def test_skills_multiple_and_ordered_by_position():
    body = ("background in statistics and hands-on experience with "
            "SQL, Python, and Tableau")
    assert extract_skills(body) == ["Statistics", "SQL", "Python", "Tableau"]


def test_skills_case_insensitive_canonical_form():
    # Lowercase mentions surface with canonical capitalisation.
    assert extract_skills("i know python, sql, and tableau") == \
        ["Python", "SQL", "Tableau"]


def test_skills_dedup_case_insensitive():
    assert extract_skills("Python and PYTHON skills").count("Python") == 1


def test_skills_multiword_not_subsumed_by_substring():
    # "SQL Server" must not also leak "SQL"; "PostgreSQL" stands on its own.
    skills = extract_skills("SQL Server and PostgreSQL")
    assert "SQL Server" in skills
    assert "PostgreSQL" in skills
    assert "SQL" not in skills


def test_skills_punctuated_tokens_match():
    skills = extract_skills("I code in C++ and C# and Node.js daily")
    assert "C++" in skills
    assert "C#" in skills
    assert "Node.js" in skills


def test_skills_mysql_not_reported_as_sql():
    # Subword boundary: "MySQL" must not produce a bare "SQL" hit.
    skills = extract_skills("I use mysql and MySQL")
    assert "SQL" not in skills
    assert "MySQL" in skills


def test_skills_none_when_absent():
    assert extract_skills("no technical skills mentioned here") == []


def test_skills_empty_body():
    assert extract_skills("") == []


# ── Education ─────────────────────────────────────────────────────────────────
def test_education_bachelors_and_masters():
    body = "I hold a Bachelor's degree in CS and am pursuing a Master's."
    assert extract_education(body) == ["Bachelor's", "Master's"]


def test_education_case_insensitive():
    assert "PhD" in extract_education("i have a phd and mba")


def test_education_none_when_absent():
    assert extract_education("statistics and SQL experience") == []


def test_education_empty_body():
    assert extract_education("") == []


# ── Seniority ─────────────────────────────────────────────────────────────────
def test_seniority_senior_level():
    assert extract_seniority("Senior Frontend Developer position") == "Senior"


def test_seniority_junior_entry_level():
    assert extract_seniority("An entry-level junior role") == "Entry-level"


def test_seniority_none_when_absent():
    # "Data Analyst" carries no career-level modifier.
    assert extract_seniority("Data Analyst position") is None


def test_seniority_not_confused_with_role_manager():
    # "Manager"/"Consultant" are role titles, not seniority levels.
    assert extract_seniority("Hiring for Product Manager role") is None


def test_seniority_not_confused_with_role_executive():
    # "Executive" in "Sales Executive" is a role noun, not a career level.
    assert extract_seniority("excited to apply for the Sales Executive position") is None


def test_seniority_lead_level():
    assert extract_seniority("Lead Developer with 4 years") == "Lead"


# ── Location ──────────────────────────────────────────────────────────────────
def test_location_based_in_with_country():
    assert extract_location("I am based in Berlin, Germany for this role") == \
        "Berlin, Germany"


def test_location_country_fallback():
    assert extract_location("I have worked in Canada for years") == "Canada"


def test_location_none_when_absent():
    # The classic "SQL, Python" body must not be misread as a place.
    body = "background in statistics and experience with SQL, Python, and Tableau"
    assert extract_location(body) is None


def test_location_rejects_prose():
    assert extract_location("Please consider this application") is None


# ── Company ───────────────────────────────────────────────────────────────────
def test_company_after_joining():
    assert extract_company("I am joining Acme Corp. excited to apply") == "Acme Corp"


def test_company_after_position_at():
    assert extract_company("the role at BrightWave Media, Inc.") == \
        "BrightWave Media"


def test_company_none_when_absent():
    assert extract_company("I would welcome the opportunity to discuss "
                           "how my skills align with your team's needs.") is None


def test_company_rejects_prose():
    assert extract_company("Please align with your team's needs.") is None


# ── Start date / availability ─────────────────────────────────────────────────
def test_start_date_captured():
    assert extract_start_date("I can start on 2025-01-15 for the role") == "2025-01-15"


def test_start_date_keyword_immediately():
    assert extract_start_date("Available immediately.") is None
    assert extract_start_date("I can start immediately.") == "immediately"
    assert extract_start_date("Availability: ASAP").lower() == "asap"


def test_start_date_none_when_absent():
    assert extract_start_date("Looking forward to hearing from you.") is None


# ── Work type ─────────────────────────────────────────────────────────────────
def test_work_type_remote():
    assert extract_work_type("This is a remote position") == "remote"


def test_work_type_hybrid():
    assert extract_work_type("A hybrid role based in NYC") == "hybrid"


def test_work_type_onsite_forms():
    assert extract_work_type("on-site in Chicago") == "onsite"
    assert extract_work_type("in-office role") == "onsite"


def test_work_type_none_when_absent():
    assert extract_work_type("An in-person role with no modality word") is None


# ── Languages ─────────────────────────────────────────────────────────────────
def test_languages_extracted_in_order():
    body = "Fluent in English and Spanish; native French speaker"
    assert extract_languages(body) == ["English", "Spanish", "French"]


def test_languages_case_insensitive():
    assert extract_languages("fluent in english") == ["English"]


def test_languages_none_when_absent():
    assert extract_languages("No languages section here") == []


# ── Certifications ────────────────────────────────────────────────────────────
def test_certifications_extracted():
    body = "I hold a PMP and AWS Certified badge, plus my CCNA."
    assert extract_certifications(body) == ["PMP", "AWS Certified", "CCNA"]


def test_certifications_none_when_absent():
    assert extract_certifications("No certifications listed") == []


# ── Full candidate dict carries the new fields ────────────────────────────────
def test_candidate_has_enriched_fields():
    from_header = '"Jane Doe" <jane.doe@example.com>'
    body = ("Senior Python engineer with a Bachelor's degree.\n"
            "Joining Acme Corp. Based in Berlin, Germany.\n"
            "Skills: SQL, Tableau, and C++.\n\n"
            "Best,\nJane Doe")
    details = extract_candidate(from_header, body)
    assert details["name"] == "Jane Doe"
    assert details["skills"] == ["Python", "SQL", "Tableau", "C++"]
    assert details["education"] == ["Bachelor's"]
    assert details["seniority"] == "Senior"
    assert details["location"] == "Berlin, Germany"
    assert details["company"] == "Acme Corp"
    # Newly-extracted fields are absent from this short body -> honestly null/empty.
    assert details["start_date"] is None
    assert details["work_type"] is None
    assert details["languages"] == []
    assert details["certifications"] == []


# ── End-to-end on a real .eml in the repo (extract each and every detail) ────
def test_alex_kumar_eml_extracts_all_details():
    eml = Path(__file__).parents[1] / "mails" / \
        "Application for Data Analyst Position - Alex Kumar.eml"
    raw = eml.read_text(encoding="utf-8", errors="replace")
    result = parse_job_application(raw)

    assert result["is_job_application"] is True
    assert result["job_role"] == "Data Analyst"
    c = result["candidate"]
    assert c["name"] == "Alex Kumar"
    # The only contact detail actually present in the message.
    assert c["email"] is None
    assert c["phone"] == []
    assert c["links"] == []
    assert c["years_of_experience"] is None
    assert c["salary_expectation"] is None
    assert c["notice_period"] is None
    # Every skill mentioned in the cover letter is captured, in order.
    assert c["skills"] == ["Statistics", "SQL", "Python", "Tableau"]
    # Fields genuinely absent from this short cover letter stay empty/null.
    assert c["education"] == []
    assert c["seniority"] is None
    assert c["location"] is None
    assert c["company"] is None
    # The message sender (recruiter) is surfaced separately from the candidate
    # so their contact details are not lost — nor mis-attributed to the applicant.
    assert result["sender"] == {
        "name": "Shakti Singh",
        "email": "developer.shaktisingh@gmail.com",
    }
    # Newly-extracted fields are honestly empty: this short cover letter
    # mentions no start date, work type, languages, certifications, or files.
    assert c["start_date"] is None
    assert c["work_type"] is None
    assert c["languages"] == []
    assert c["certifications"] == []
    assert result["attachments"] == []
    # The cover letter preserves the applicant's own prose.
    assert "background in statistics" in result["clean_cover_letter"]
    # Must round-trip through JSON (no sets / non-serialisable objects).
    import json
    json.dumps(result)
