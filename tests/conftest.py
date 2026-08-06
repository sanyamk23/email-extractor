"""Shared test fixtures — realistic sample emails covering edge cases."""
import pytest


# ── A complete, confident job application (the happy path) ──────────────────
FULL_APPLICATION = {
    "from": '"Jane Doe" <jane.doe@example.com>',
    "subject": "Application for Senior Frontend Developer",
    "body": (
        "Hi Hiring Manager,\n\n"
        "I am writing to apply for the Senior Frontend Developer position "
        "at Acme Corp. With 5 years of experience in software engineering, "
        "I have built scalable React applications.\n\n"
        "My resume, attached, highlights my background.\n\n"
        "Technical highlights:\n"
        "- 5 years of experience with React, TypeScript, and AWS\n"
        "- Salary expectation: $120,000 per year\n"
        "- Notice period: 30 days\n\n"
        "https://linkedin.com/in/janedoe\n"
        "https://github.com/janedoe\n"
        "https://janedoe.dev\n"
        "Phone: +1-555-019-9494\n\n"
        "Best regards,\n"
        "Jane Doe\n"
        "\n"
        "On Mon, Jul 1, 2026 John Smith wrote:\n"
        "> Did you see the open role?\n"
    ),
}


# ── Application missing a phone number ───────────────────────────────────────
APPLICATION_NO_PHONE = {
    "from": "John Smith <john.smith@email.com>",
    "subject": "Job Application - Software Engineer",
    "body": (
        "Dear Recruiter,\n\n"
        "Please accept this application for the Software Engineer position.\n"
        "I have 3 years of experience with Python and Django.\n"
        "You may reach me at john.smith@email.com.\n\n"
        "GitHub: https://github.com/jsmith\n\n"
        "Sincerely,\nJohn Smith"
    ),
}


# ── Ambiguous subject line (no explicit trigger, weak signal) ─────────────────
APPLICATION_AMBIGUOUS = {
    "from": "Alex Rivera <alex.rivera.dev@gmail.com>",
    "subject": "Following up on the engineering role",
    "body": (
        "Hi there,\n\n"
        "I wanted to follow up regarding the engineering role we discussed.\n"
        "My resume is attached for your review. I have 4 years of experience "
        "building backend systems in Go.\n\n"
        "Thanks,\nAlex Rivera"
    ),
}


# ── Multiple links of every supported type ───────────────────────────────────
APPLICATION_MULTIPLE_LINKS = {
    "from": "Maria Garcia <maria.g@outlook.com>",
    "subject": "Applying for Product Manager",
    "body": (
        "Hello,\n\n"
        "Applying for the Product Manager position.\n"
        "3 years of experience in product management.\n\n"
        "LinkedIn: https://www.linkedin.com/in/mariagarcia\n"
        "GitHub: https://github.com/mariag\n"
        "Portfolio: https://mariagarcia.design\n"
        "Blog: https://blog.mariagarcia.com\n"
        "Twitter: https://twitter.com/mariag\n"
        "Phone: +44 20 7946 0958\n\n"
        "Regards,\nMaria Garcia"
    ),
}


# ── A non-application email (should be classified as False) ───────────────────
NON_APPLICATION = {
    "from": "Newsletter <newsletter@techweekly.io>",
    "subject": "Tech Weekly: Issue #42",
    "body": (
        "Hello,\n\n"
        "Welcome to this week's Tech Weekly roundup. We cover the latest in "
        "AI, cloud, and developer tools.\n\n"
        "To unsubscribe, click here: https://techweekly.io/unsub\n\n"
        "— The Tech Weekly Team"
    ),
}


# ── Raw RFC-822 email string (tests the email-module parser path) ────────────
RAW_RFC822_EMAIL = (
    "From: Jane Doe <jane.doe@example.com>\r\n"
    "Sent: Tuesday, July 2, 2026 10:00 AM\r\n"
    "To: recruiting@acme.com\r\n"
    "Subject: Application for Data Engineer\r\n"
    "Content-Type: text/plain; charset=utf-8\r\n"
    "\r\n"
    "Hi team,\r\n"
    "\r\n"
    "I am applying for the Data Engineer position at Acme Corp.\r\n"
    "\r\n"
    "Please find my resume attached. I have 4 years of experience with "
    "Python and AWS.\r\n"
    "\r\n"
    "Salary expectation: $110,000\r\n"
    "Notice period: 2 weeks\r\n"
    "\r\n"
    "Best regards,\r\n"
    "Jane Doe\r\n"
)


# ── Name extracted from the closing sign-off (no display name in From) ──────
SIGNOFF_NAME_EMAIL = {
    "from": "candidate99@protonmail.com",
    "subject": "Application for QA Analyst",
    "body": (
        "Hi,\n\n"
        "I am excited to apply for the QA Analyst role.\n\n"
        "Thanks,\n"
        "Sam Wilson\n"
        "QA Tester, XYZ Games"
    ),
}


# ── Experience expressed as a range ──────────────────────────────────────────
EXPERIENCE_RANGE_EMAIL = {
    "from": '"Bob Lee" <bob.lee@corp.net>',
    "subject": "Job Application: DevOps Engineer",
    "body": (
        "Dear Hiring Manager,\n\n"
        "I wish to apply for the DevOps Engineer position.\n"
        "Bringing 3-5 years of experience in cloud infrastructure, I am a "
        "strong fit for this role.\n\n"
        "Regards,\nBob Lee"
    ),
}


# ── Application whose role is only present as a dictionary term ──────────────
DICTIONARY_ROLE_EMAIL = {
    "from": "Devon Rex <devon.rex@email.com>",
    "subject": "Interested in the position",
    "body": (
        "To whom it may concern,\n\n"
        "I am interested in the position. I have 6 years of experience as a "
        "Data Scientist. My resume is attached.\n\n"
        "Best,\nDevon Rex"
    ),
}


@pytest.fixture
def full_application():
    return dict(FULL_APPLICATION)


@pytest.fixture
def email_no_phone():
    return dict(APPLICATION_NO_PHONE)


@pytest.fixture
def email_ambiguous():
    return dict(APPLICATION_AMBIGUOUS)


@pytest.fixture
def email_multiple_links():
    return dict(APPLICATION_MULTIPLE_LINKS)


@pytest.fixture
def non_application_email():
    return dict(NON_APPLICATION)


@pytest.fixture
def raw_rfc822_email():
    return RAW_RFC822_EMAIL


@pytest.fixture
def signoff_name_email():
    return dict(SIGNOFF_NAME_EMAIL)


@pytest.fixture
def experience_range_email():
    return dict(EXPERIENCE_RANGE_EMAIL)


@pytest.fixture
def dictionary_role_email():
    return dict(DICTIONARY_ROLE_EMAIL)
