# email_extractor

A **strictly rule-based** Information Extraction pipeline that, given a job-application
email, decides whether it *is* a job application, identifies the target **job role**, and
extracts structured **candidate metadata** into a JSON object.

No AI, no LLM API calls, no cloud endpoints, and **zero runtime dependencies** — only the
Python standard library (`re`, `email`, `html`, `email.utils`). Lightweight NLP libraries
(spaCy / python-phonenumbers) are *optional* and auto-detected with graceful fallback.

## Features

- **Classification** — `is_job_application` + `confidence_score` (0–1) derived from
  subject/body keyword triggers (weighted, subject-boosted) and structural bonuses.
- **Role extraction** — priority-ordered regex patterns (`Application for …`, `Role: …`,
  `Position of …`, `the … position`) with a 200+ term **role dictionary** fallback.
- **Enriched candidate details** — full name (From header → sign-off → signature),
  email, normalised phone numbers (de-duplicated), LinkedIn/GitHub/portfolio links,
  years of experience, salary expectation, notice period, plus **skills, education,
  seniority, location, company, start date, work type, languages, and
  certifications** (all rule + dictionary based; absent fields are `null`/empty).
- **Attachments** — lists every MIME attachment as structured metadata
  (`filename`, `mime_type`, `size` in bytes, `content_id`, `disposition`). Raw file
  content is never embedded, keeping the JSON output small and serialisable.
- **Sender** — the actual message sender's {name, email} from the `From:` header,
  surfaced separately so a recruiter/forwarder's contact details aren't lost (and
  aren't mis-attributed to the candidate when they differ).
- **Clean cover letter** — strips forwarded/quoted threads, RFC-3676 signature
  separators, and attachment boilerplate.

## Installation

```bash
pip install -e .            # production (no dependencies)
pip install -e ".[dev]"     # with the pytest test suite
```

## Quick start

```python
import json
from email_extractor import parse_job_application

email = {
    "from": '"Jane Doe" <jane.doe@example.com>',
    "subject": "Application for Senior Frontend Developer",
    "body": """Hi Hiring Manager,

I am writing to apply for the Senior Frontend Developer position at Acme Corp.
With 5 years of experience in software engineering ...

Salary expectation: $120,000 per year
Notice period: 30 days

https://linkedin.com/in/janedoe
https://github.com/janedoe
Phone: +1-555-019-9494

Best regards,
Jane Doe

-----Original Message-----
From: recruiter@acme.com
...""",
}

print(json.dumps(parse_job_application(email), indent=2))
```

### Raw RFC-822 email

```python
result = parse_job_application(open("application.eml").read())
```

A string whose first line is a `Header: value` pair is parsed as RFC-822; otherwise it is
treated as a plain body. A `dict` with `from`/`subject`/`body` keys is also accepted.

### CLI

```bash
email-extractor application.eml          # from an .eml file
cat application.eml | email-extractor --stdin
echo '{"from":"...","subject":"...","body":"..."}' | email-extractor --stdin
python -m email_extractor application.eml
```

## Output schema

```json
{
  "is_job_application": true,
  "confidence_score": 0.95,
  "job_role": "Senior Frontend Developer",
  "candidate": {
    "name": "Jane Doe",
    "email": "jane.doe@example.com",
    "phone": ["+15550199494"],
    "links": ["https://linkedin.com/in/janedoe", "https://github.com/janedoe"],
    "years_of_experience": "5",
    "salary_expectation": "120000",
    "notice_period": "30 days",
    "skills": ["JavaScript", "React"],
    "education": ["Bachelor's degree"],
    "seniority": "Senior",
    "location": "San Francisco",
    "company": "Acme Corp",
    "start_date": "2025-01-15",
    "work_type": "remote",
    "languages": ["English", "Spanish"],
    "certifications": ["PMP"]
  },
  "sender": {"name": "Jane Doe", "email": "jane.doe@example.com"},
  "attachments": [
    {"filename": "resume.pdf", "mime_type": "application/pdf", "size": 1024,
     "content_id": "", "disposition": "attachment"}
  ],
  "clean_cover_letter": "Hi Hiring Manager, ..."
}
```

> `candidate.phone` is a **list** (an email may contain several numbers) and
> `salary_expectation` / `notice_period` / `location` / `company` are `null` when
> not found. List fields are empty `[]` when nothing matches.

## Architecture

```
email_extractor/
├── config.py                 # modular regex patterns + role dictionary + thresholds
├── email_utils.py            # RFC-822 parsing, From-header sender, attachments, HTML→text
├── classifier.py             # is_job_application + confidence scoring
├── role_extractor.py         # job-role extraction (patterns → dictionary fallback)
├── candidate_extractor.py    # contact + experience + salary + notice + skills/education/seniority/location/company/start/languages/certs
├── body_cleaner.py           # clean cover-letter body
├── pipeline.py               # orchestrates the modules into the final dict
├── cli.py                    # command-line interface
└── __main__.py               # `python -m email_extractor`
```

All patterns live in [`config.py`](email_extractor/config.py) so the system can be
tuned without touching extraction logic:

| Section | What it controls |
|---|---|
| `APPLICATION_TRIGGERS` | classification keywords, per-pattern weight & scope |
| `CLASSIFICATION_THRESHOLD` / `SUBJECT_BOOST` | decision boundary & subject amplification |
| `ROLE_PATTERNS` | ordered role-extraction regexes |
| `ROLE_DICTIONARY` | 90+ canonical tech/business roles (fallback + case normalisation) |
| `SIGNOFF_PATTERNS` / `TITLE_WORDS` | sign-off name capture & title trimming |
| `EMAIL_REGEX` / `PHONE_REGEX` | contact-info patterns |
| `LINKEDIN_REGEX` / `GITHUB_REGEX` / `URL_REGEX` / `NON_PERSONAL_DOMAINS` | profile link extraction |
| `EXPERIENCE_PATTERNS` / `SALARY_REGEX` / `NOTICE_PERIOD_REGEXES` | key-metric extraction |
| `THREAD_MARKERS` / `SIGNATURE_DASH` / `ATTACHMENT_BOILERPLATE` | body cleaning |

## Confidence scoring

`confidence = min(subject_trigger_score * 1.5 + body_trigger_score + bonuses, 1.0)`.

Bonus points are added for resume/CV mentions, a recognizable sign-off, a phone number,
and an explicit years-of-experience statement. An email is classified as an application
when `confidence >= CLASSIFICATION_THRESHOLD` (default `0.30`).

## Phone handling

`PHONE_REGEX` enforces a minimum digit density so years (`2026`) and short numbers
(`5 years`) are **not** mistaken for phone numbers. `phonenumbers` (if installed) can be
plugged into `_normalise_phone` for E.164 validation; absent that, a built-in normaliser
produces `+<digits>` with US default country code for 10-digit numbers.

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

70 tests cover the happy path, missing phones, ambiguous subjects, multiple links,
sign-off-only names, experience ranges, dictionary-role fallback, RFC-822 parsing,
HTML bodies, and non-application rejection.

## Limitations

- Phone detection is regex-based; install [`phonenumbers`](https://github.com/daviddrysdale/python-phonenumbers)
  for strict validation (optional).
- No external NLP is used (by design). Named-entity recognition would improve accuracy
  on very noisy free-text, but the rule set covers the common formats above.
