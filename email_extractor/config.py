"""Modular configuration for the rule-based job-application email parser.

All regex patterns, role dictionary, and tunable thresholds live here so the
extraction logic can be retuned without touching code.  Patterns are compiled
once at import time for performance.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 1. Classification triggers
# ---------------------------------------------------------------------------
# Each tuple: (regex, weight, scope)
#   scope in {"subject", "body", "both"} — subject hits are boosted at runtime.
APPLICATION_TRIGGERS: list[tuple[str, float, str]] = [
    (r"applying\s+(?:for|to|at)",              0.30, "both"),
    (r"\bapply\s+(?:for|to|at)\b",             0.25, "both"),
    (r"application\s+for",                     0.30, "both"),
    (r"job\s+application",                    0.30, "both"),
    (r"application\s*[:\-)]\s*\S",             0.25, "both"),
    (r"\bresume\b\s*[:\-,]?",                  0.25, "both"),
    (r"\bcv\b\s*[:\-,]?",                      0.20, "both"),
    (r"please\s+find\s+(?:my\s+)?(?:resume|cv|application)", 0.25, "both"),
    (r"interested\s+in\s+(?:the\s+)?(?:role|position|opportunity)", 0.25, "both"),
    (r"seeking\s+the\s+position\s+of",         0.20, "both"),
    (r"excited\s+to\s+apply",                  0.20, "body"),
    (r"please\s+accept\s+my\s+(?:application|resume|cv)", 0.30, "both"),
    (r"my\s+resume\s+and\s+cover\s+letter",    0.20, "both"),
    (r"position\s+of\s+\S",                    0.20, "both"),
    (r"\brole\s*:\s*\S",                       0.15, "both"),
    (r"\bposition\s*:\s*\S",                   0.15, "both"),
    (r"join\s+(?:your\s+)?team",              0.15, "body"),
    (r"career\s+opportunity",                  0.15, "both"),
    (r"talent\s+acquisition|hiring\s+manager|recruitment|recruiter",
     0.10, "both"),
    (r"open\s+role|open\s+position",           0.15, "both"),
    (r"submit(?:ting)?\s+my\s+application",    0.20, "both"),
]

# A raw application is decided when confidence reaches this threshold.
CLASSIFICATION_THRESHOLD: float = 0.30
# Subject-line hits count more heavily than body hits.
SUBJECT_BOOST: float = 1.5

# Bonus points added when the body looks genuinely like an application.
BONUS_RESUME_MENTION = 0.10
BONUS_SIGNOFF = 0.05
BONUS_PHONE = 0.05
BONUS_EXPERIENCE = 0.05
BONUS_NOTICE_PERIOD = 0.05  # mentioning a notice period signals an application


# Pre-compiled patterns reused by the classifier structural bonuses.
RESUME_CV_REGEX: re.Pattern = re.compile(r"\b(?:resume|cv)\b", re.IGNORECASE)
SIGNOFF_REGEX: re.Pattern = re.compile(
    r"(?:Sincerely|Best\s+regards|Thanks|Thank\s+you|Regards|Cheers)"
    r"\s*,?\s*\n", re.IGNORECASE)


def _compile_triggers() -> list[dict]:
    compiled: list[dict] = []
    for pattern, weight, scope in APPLICATION_TRIGGERS:
        compiled.append(
            {"regex": re.compile(pattern, re.IGNORECASE),
             "weight": weight,
             "scope": scope}
        )
    return compiled


COMPILED_TRIGGERS: list[dict] = _compile_triggers()


# ---------------------------------------------------------------------------
# 2. Job-role extraction patterns (tried in priority order, first match wins)
# ---------------------------------------------------------------------------
# The capture group holds the candidate role text; trailing company/role
# words are stripped by the normaliser in ``role_extractor.py``.
ROLE_PATTERNS: list[str] = [
    r"Re:\s*Application\s+for\s+([A-Z][^.,/\n]{1,80})",
    r"Application\s+for\s+([A-Z][^.,/\n]{1,80})",
    r"applying\s+(?:for|to)\s+(?:the\s+)?([A-Z][^.,/\n]{1,80})",
    r"[Rr]ole\s*:\s*([A-Z][^.,/\n]{1,80})",
    r"[Pp]osition\s*:\s*([A-Z][^.,/\n]{1,80})",
    r"[Pp]osition\s+of\s+([A-Z][^.,/\n]{1,80})",
    r"role\s+of\s+the\s+([A-Z][^.,/\n]{1,80})",
    r"the\s+([A-Z][A-Za-z][A-Za-z\s\-]{2,50}?)\s+(?:position|role|opening)\b",
]

COMPILED_ROLE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in ROLE_PATTERNS
]


# Canonical role dictionary — used as a fallback lookup and for case
# normalisation of pattern-extracted roles.  Listed most-specific/longest first.
ROLE_DICTIONARY: list[str] = [
    # ── Engineering ──────────────────────────────────────────────────────
    "Principal Software Engineer",
    "Staff Software Engineer",
    "Senior Staff Software Engineer",
    "Senior Software Engineer",
    "Software Engineer",
    "Software Developer",
    "Staff Frontend Engineer",
    "Senior Frontend Developer",
    "Frontend Engineer",
    "Frontend Developer",
    "Senior Backend Developer",
    "Backend Engineer",
    "Backend Developer",
    "Full Stack Developer",
    "Full Stack Engineer",
    "Mobile Developer",
    "iOS Developer",
    "Android Developer",
    "React Native Developer",
    "Flutter Developer",
    "Embedded Software Engineer",
    "Firmware Engineer",
    "DevOps Engineer",
    "Cloud Engineer",
    "Platform Engineer",
    "Site Reliability Engineer",
    "Infrastructure Engineer",
    "SRE",
    "QA Engineer",
    "QA Analyst",
    "Test Automation Engineer",
    "Manual Tester",
    "Release Engineer",
    "Build Engineer",
    "Compiler Engineer",
    "API Engineer",
    "Integration Engineer",
    "Data Infrastructure Engineer",
    # ── Data, Analytics & Machine Learning ───────────────────────────────
    "Machine Learning Engineer",
    "Senior Machine Learning Engineer",
    "Deep Learning Engineer",
    "Applied ML Engineer",
    "AI Research Scientist",
    "Research Scientist",
    "Applied Research Scientist",
    "Data Scientist",
    "Senior Data Scientist",
    "Staff Data Scientist",
    "Data Engineer",
    "Senior Data Engineer",
    "Data Analyst",
    "Senior Data Analyst",
    "Analytics Engineer",
    "Business Intelligence Engineer",
    "Data Science Manager",
    "Machine Learning Scientist",
    "Research Engineer",
    "Research Associate",
    "Postdoctoral Researcher",
    "Clinical Data Scientist",
    "Quantitative Analyst",
    "Risk Analyst",
    "Statistical Modeling Analyst",
    # ── Product & Project Management ────────────────────────────────────
    "Group Product Manager",
    "Senior Product Manager",
    "Product Manager",
    "Technical Product Manager",
    "Associate Product Manager",
    "Product Owner",
    "Technical Program Manager",
    "Senior Technical Program Manager",
    "Program Manager",
    "Project Manager",
    "Senior Project Manager",
    "Scrum Master",
    "Agile Coach",
    "Product Marketing Manager",
    "Growth Marketing Manager",
    "Growth Product Manager",
    # ── Design ──────────────────────────────────────────────────────────
    "Head of Design",
    "Lead Product Designer",
    "Product Designer",
    "Senior Product Designer",
    "UI Designer",
    "Senior UI Designer",
    "UX Designer",
    "Senior UX Designer",
    "Visual Designer",
    "Interaction Designer",
    "User Researcher",
    "Design Systems Engineer",
    "Design Engineer",
    "Brand Designer",
    "Motion Designer",
    "Graphic Designer",
    "Industrial Designer",
    # ── Leadership & Strategy ───────────────────────────────────────────
    "VP of Engineering",
    "CTO",
    "Chief Technology Officer",
    "Head of Engineering",
    "Director of Engineering",
    "Engineering Director",
    "Senior Engineering Manager",
    "Engineering Manager",
    "VP of Product",
    "Chief Product Officer",
    "Product Director",
    "VP of Data Science",
    "Data Science Director",
    "VP of Design",
    "Design Director",
    "VP of Sales",
    "VP of Marketing",
    "Chief Information Officer",
    "Chief Security Officer",
    "Head of Product",
    "Tech Lead",
    "Team Lead",
    "Staff Engineer",
    "Principal Engineer",
    # ── Security ────────────────────────────────────────────────────────
    "Security Engineer",
    "Senior Security Engineer",
    "Security Operations Engineer",
    "Penetration Tester",
    "Security Analyst",
    "Cybersecurity Analyst",
    "Information Security Analyst",
    "Compliance Analyst",
    "Network Security Engineer",
    "Application Security Engineer",
    "Security Architect",
    "Identity & Access Management Engineer",
    "SOC Analyst",
    # ── Infrastructure & Systems ────────────────────────────────────────
    "Network Engineer",
    "Systems Engineer",
    "Database Administrator",
    "DBA",
    "Systems Administrator",
    "Cloud Solutions Architect",
    "Solutions Architect",
    "Systems Architect",
    "Storage Engineer",
    "Telecommunications Engineer",
    # ── Business & Operations ───────────────────────────────────────────
    "Business Analyst",
    "Senior Business Analyst",
    "Operations Analyst",
    "Financial Analyst",
    "Management Consultant",
    "Strategy Consultant",
    "IT Consultant",
    "Data Governance Analyst",
    "Operations Manager",
    "Consultant",
    "Business Development Representative",
    "Business Development Manager",
    "Account Manager",
    "Sales Development Representative",
    "Account Executive",
    "Sales Engineer",
    "Customer Success Manager",
    "Senior Customer Success Manager",
    "Technical Account Manager",
    "Support Engineer",
    "Technical Support Engineer",
    "People Operations",
    "HR Business Partner",
    "Talent Acquisition Specialist",
    "Talent Acquisition Manager",
    "Recruiter",
    "University Recruiter",
    "HR Generalist",
    "Compensation & Benefits Manager",
    "Office Manager",
    "Facilities Manager",
    # ── Research & Academia ─────────────────────────────────────────────
    "Clinical Research Associate",
    "Quantitative Researcher",
    "Lab Technician",
    "Teaching Assistant",
    "Lecturer",
    "Assistant Professor",
    "Associate Professor",
    "Professor",
    # ── Creative & Content ──────────────────────────────────────────────
    "Content Strategist",
    "Technical Writer",
    "Copywriter",
    "Editor",
    "Content Designer",
    "Marketing Designer",
    "Video Editor",
    "Motion Graphics Designer",
    # ── Emerging & Specialised ──────────────────────────────────────────
    "Blockchain Engineer",
    "Smart Contract Engineer",
    "Cryptographer",
    "Game Developer",
    "AR/VR Developer",
    "Computer Vision Engineer",
    "Robotics Engineer",
    "Autonomous Vehicle Engineer",
    "Quantum Computing Researcher",
    "Bioinformatics Engineer",
    "Geospatial Engineer",
    "Embedded Systems Engineer",
    "Hardware Engineer",
    "RF Engineer",
    "Semiconductor Process Engineer",
    "DevSecOps Engineer",
    "Chaos Engineer",
    "Data Center Engineer",
]

# Lower-cased lookup preserving the canonical capitalisation.
ROLE_CANONICAL: dict[str, str] = {role.lower(): role for role in ROLE_DICTIONARY}

# Role dictionary pre-sorted longest-first so multi-word roles win during
# substring matching (replaces the per-call ``sorted()`` in _match_dictionary).
SORTED_ROLE_DICTIONARY: list[str] = sorted(
    ROLE_DICTIONARY, key=len, reverse=True)


# ---------------------------------------------------------------------------
# 3. Candidate name extraction — sign-off patterns
# ---------------------------------------------------------------------------
# Each captures the name that appears on the line(s) immediately after a sign-off.
# Use inline (?i:...) for sign-off phrase only; name capture must be case-sensitive
# (so [A-Z] only matches uppercase) to avoid false positives like "appreciate".
_NAME_TAIL = r"[*_]{0,2}[A-Z][A-Za-z’’.\-]{1,30}(?:\s+[A-Z][A-Za-z’’.\-]{1,30}){0,2}"
SIGNOFF_PATTERNS: list[re.Pattern] = [
    re.compile(p) for p in [
        r"(?i:Sincerely)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Thanks?)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Thank\s+you)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Best\s+regards)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Best)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Warm\s+regards)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Kind\s+regards)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Regards)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Cheers)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Many\s+thanks)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Many\s+regards)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Take\s+care)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Looking\s+forward\s+to\s+hearing\s+from\s+you)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
        r"(?i:Talk\s+soon)[,.]?\s*\n\s*(" + _NAME_TAIL + r")",
    ]
]

# Words that are titles/descriptors rather than names — used to trim sign-off
# captures that accidentally include a title line.
TITLE_WORDS: set[str] = {
    "engineer", "developer", "manager", "analyst", "scientist", "lead",
    "director", "specialist", "architect", "designer", "consultant",
    "officer", "associate", "senior", "junior", "principal", "staff",
    "program", "project", "marketing", "sales", "recruiter", "hr",
    "student", "intern", "phd", "ms", "bs", "mba", "team", "lead",
    "tech", "software", "data", "product", "growth", "customer",
    "account", "success", "operations", "finance", "legal",
    "qa", "tester",
}


# ---------------------------------------------------------------------------
# 4. Contact-info patterns
# ---------------------------------------------------------------------------
EMAIL_REGEX: re.Pattern = re.compile(
    r'''(?x)
    [a-zA-Z0-9._%+\-]+          # local part
    @
    [a-zA-Z0-9.\-]+             # domain
    \.[a-zA-Z]{2,}              # TLD
    '''
)

# Strict phone regex — requires phone-like formatting to avoid matching
# bare numbers like years, job IDs, etc. Must have at least one phone indicator:
# +, parentheses, country code prefix, or spaces/dashes in standard positions.
PHONE_REGEX: re.Pattern = re.compile(
    r'''(?x)
    (?<![\d#])                      # not preceded by digit or # (avoid Req #123)
    (?:
        (?:\+|00)\d{1,3}[\s.-]?     # country code (+1, 001, etc.)
        | \(\d{2,4}\)[\s.-]?        # OR area code in parens ((415), etc.)
        | \d{2,4}[\s.-]             # OR area code with separator (415-, 415 )
    )?
    \d{3,4}(?:[\s./-]*\d{1,4}){1,3}  # main 7-16 digit block with separators
    (?:[\s./]*(?:ext\.?|x|\#)\s*\d+)? # optional extension
    ''')


# Reusable: strip all non-digit characters (phone normalisation).
DIGITS_ONLY: re.Pattern = re.compile(r"\D")

LINKEDIN_REGEX: re.Pattern = re.compile(
    r"https?://(?:www\.|ca\.)?linkedin\.com/[^\s,)<>]+", re.IGNORECASE)
GITHUB_REGEX: re.Pattern = re.compile(
    r"https?://(?:www\.)?github\.com/[^\s,)<>]+", re.IGNORECASE)
# Generic URL grabber — used to find portfolio / personal sites.
URL_REGEX: re.Pattern = re.compile(
    r"https?://[^\s,)>]+", re.IGNORECASE)

# Services whose URLs should not be treated as the candidate's own site.
NON_PERSONAL_DOMAINS: set[str] = {
    "linkedin.com", "github.com", "twitter.com", "x.com", "facebook.com",
    "instagram.com", "angel.co", "wellfound.com", "stackshare.io",
    "stackoverflow.com", "medium.com",
}


# ---------------------------------------------------------------------------
# 5. Key-metric patterns
# ---------------------------------------------------------------------------
EXPERIENCE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(\d+(?:\.\d+)?)\s+years?\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\s+year\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\s+years?\s+experience",
        r"(\d+(?:\.\d+)?)\s+years?\s+in\s+the\s+field",
        r"(\d+(?:\.\d+)?)\s+years?\s+in\s+(?:software|tech|engineering|data)",
        r"(?:over|more than|~)?\s*(\d+(?:\.\d+)?)\s*years?\s*experience",
        r"with\s+(?:over\s+)?(\d+(?:\.\d+)?)\s+years?",
        r"(\d+(?:\.\d+)?)\+?\s*years?\s*experience",
        # Fallback: bare "X years" / "X+ years" (excludes ages such as "25 years old").
        r"(\d+(?:\.\d+)?)\+?\s*years?(?!\s*(?:old|ago))",
    ]]

# Salary: amount must sit near a salary-context keyword.
SALARY_REGEX: re.Pattern = re.compile(
    r'''(?ix)
    (?:salary|compensation|expected|rate|per\s*-\s*year|per\s+year|annually)
    [^0-9$]{0,40}
    \$?\s*(\d[\d,]+\.?\d*)(?:\s*(?:k|K| thousand))?
    ''')

# Notice period / availability.
NOTICE_PERIOD_REGEXES: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"notice\s*(?:period\s*)?[^0-9]{0,12}(\d+)\s*(weeks?|days?|months?)",
        r"available\s*(?:in|within)\s*(\d+)\s*(weeks?|days?|months?)",
        r"can\s+start\s*(?:in|within|after)\s*(\d+)\s*(weeks?|days?|months?)",
        r"(\d+)\s*(weeks?|days?|months?)\s*(?:notice\s*period?|notice)",
        r"(\d+)\s*(weeks?|days?|months?)\s*notice",
    ]]


# ---------------------------------------------------------------------------
# 6. Body-cleaning markers
# ---------------------------------------------------------------------------
# Forward separators ("Forwarded message"): the forwarded body after the
# separator + envelope headers (From/Date/Subject/To:...) is the *real* content
# (e.g. an applicant's cover letter) and is preserved; only the separator and
# envelope header block are stripped.  The forwarded sender is also surfaced to
# the pipeline so the candidate is attributed to the sender, not the forwarder.
FORWARD_MARKER_PATTERNS: list[str] = [
    r"-+\s*forwarded\s*message\s*-+",
]
FORWARD_MARKERS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in FORWARD_MARKER_PATTERNS
]

# Quote-reply markers: everything from the first match onward is a quoted/reply
# thread and is discarded.  "Original message" headers (a quoted reply's
# context) are treated the same way.  MULTILINE so "^" anchors each line.
QUOTE_MARKER_PATTERNS: list[str] = [
    r"-+\s*original\s+message\s*-+",
    r"^\s*from:\s+.+@\S+",
    r"^\s*sent:\s+",
    r"^\s*to:\s+",
    r"on .+ wrote:",
    r"on .+ wrote \d",
]
QUOTE_MARKERS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.MULTILINE) for p in QUOTE_MARKER_PATTERNS
]

# Lines that are signature dashes (RFC 3676 "universal signature separator").
SIGNATURE_DASH: re.Pattern = re.compile(r"^\s*(?:--|---+)\s*$")

# Raw boilerplate pattern strings (compiled below to keep definitions tidy).
ATTACHMENT_BOILERPLATE_PATTERNS: list[str] = [
    r"please\s+find\s+(?:my\s+)?(?:resume|cv|application|cover\s+letter)?\s*(?:attached|below|included|in\s+the\s+envelope)[^\.\n]*\.?",
    r"(?:i\s+am\s+sending|i\s+am\s+attaching|i\s+have\s+attached)\s+(?:my\s+)?(?:resume|cv)[^\.\n]*\.?",
    r"please\s+let\s+me\s+know\s+if\s+you\s+need\s+anything\s+else[^\.\n]*\.?",
    r"thank\s+you\s+for\s+your\s+time\s+and\s+consideration[^\.\n]*\.?",
    r"looking\s+forward\s+to\s+hearing\s+from\s+you[^\.\n]*\.?",
]
ATTACHMENT_BOILERPLATE: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in ATTACHMENT_BOILERPLATE_PATTERNS
]


# ---------------------------------------------------------------------------
# 7. Candidate-detail keyword dictionaries
# ---------------------------------------------------------------------------
# All keyword lists are de-duplicated (case-insensitive), canonicalised, and
# ordered longest-first so that multi-word skills take precedence over a
# single-word sub-token during matching (e.g. "SQL Server" before "SQL").
def _dedupe_keyword_list(keywords: list[str]) -> list[str]:
    """Drop duplicates (case-insensitive, keeping the first canonical form)
    and order longest-first so multi-word phrases win during matching."""
    seen: set[str] = set()
    out: list[str] = []
    for kw in keywords:
        key = kw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(kw)
    out.sort(key=lambda k: len(k), reverse=True)
    return out


# Technical skills / technologies / domain keywords mentioned in a body.
SKILLS: list[str] = _dedupe_keyword_list([
    # ── data / analytics / ml ───────────────────────────────────────────────
    "Statistical modeling", "Statistics", "Data visualization", "Machine learning",
    "Deep learning", "Artificial intelligence", "Natural language processing",
    "NLP", "Computer vision", "Data mining", "Experimentation", "A/B testing",
    "Dashboarding", "Data analysis", "Reporting",
    # ── programming languages ───────────────────────────────────────────────
    # NOTE: bare "R" and "Go" are intentionally omitted — they are common
    # English words and produce false positives under word-boundary matching.
    "Python", "SQL", "Scala", "Java", "JavaScript", "TypeScript",
    "C++", "C#", "Rust", "MATLAB", "SAS", "SPSS", "Stata", "Bash",
    "Shell scripting", "Shell",
    # ── data stores & warehousing ───────────────────────────────────────────
    "PostgreSQL", "Postgres", "MySQL", "MongoDB", "Redis", "Snowflake",
    "BigQuery", "Redshift", "ClickHouse", "Elasticsearch", "DynamoDB",
    "SQL Server", "Oracle", "Apache Hive", "Hive", "Presto", "Trino",
    "Databricks",
    # ── analytics & bi tools ────────────────────────────────────────────────
    "Tableau", "Power BI", "PowerBI", "Looker", "LookML", "Qlik Sense",
    "QlikView", "Apache Superset", "Superset", "Mode Analytics",
    "Periscope Data", "Periscope",
    # ── data engineering & orchestration ────────────────────────────────────
    "Apache Spark", "Spark", "Apache Airflow", "Airflow", "Apache Kafka",
    "Kafka", "Apache Beam", "Beam", "dbt", "Dataform", "Fivetran", "Stitch",
    # ── cloud & infrastructure ────────────────────────────────────────────────
    "Amazon Web Services", "AWS", "Google Cloud Platform", "GCP", "Azure",
    "Google Cloud", "Docker", "Kubernetes", "K8s", "Terraform",
    # ── ml frameworks & libraries ────────────────────────────────────────────
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "scikit-learn",
    "XGBoost", "LightGBM", "CatBoost", "Hugging Face", "Transformers",
    # ── web & devops ────────────────────────────────────────────────────────
    "HTML", "CSS", "React", "Angular", "Vue", "Next.js", "Node.js", "Nodejs",
    "Git", "Linux", "CI/CD", "Docker Compose",
    # ── backend & data libraries ────────────────────────────────────────────
    "Django", "Flask", "FastAPI", "Express.js", "Express", "Spring",
    "Pandas", "NumPy", "matplotlib", "Seaborn", "PySpark", "Polars",
    # ── product / business ─────────────────────────────────────────────────
    "Excel", "Advanced Excel", "Jira", "Confluence", "Notion",
    "Figma", "Sketch", "Adobe Analytics",
    # ── HR & recruitment ──────────────────────────────────────────────────────
    "Recruitment", "Talent Acquisition", "Employee Onboarding",
    "Resume Screening", "Interview Coordination", "HR Documentation",
    "HRMS", "Payroll", "Communication Skills",
    "Employee Relations", "Benefits Administration", "Performance Management",
    "Background Verification", "Exit Interview", "Compensation",
])


# Education-level keywords. Matched as whole words (case-insensitive); multi-word
# forms are matched before their bare sub-tokens.
EDUCATION: list[str] = _dedupe_keyword_list([
    "Doctor of Philosophy", "PhD", "Doctorate", "Doctoral",
    "MBA", "Master of Business Administration",
    "Master of Science", "Master of Engineering", "Master of Arts",
    "Master of Philosophy", "Master's", "Masters", "Master",
    "MSc", "MEng",
    "Bachelor of Science", "Bachelor of Engineering", "Bachelor of Arts",
    "Bachelor of Business", "Bachelor's", "Bachelors", "Bachelor",
    "BSc", "BTech", "MTech", "MCA", "MCom",
    "Postgraduate", "Post-Graduate", "Undergraduate", "Graduate",
    "Associate degree", "Professional degree",
])


# Seniority / career-level keywords.  Kept to unambiguous LEVEL adjectives —
# title-noun words like "Director"/"VP"/"Chief"/"Head"/"Executive" are roles
# (e.g. "Sales Executive") rather than levels, so matching them would create
# false positives; such titles live in ``ROLE_DICTIONARY``.
SENIORITY: list[str] = _dedupe_keyword_list([
    "Entry level", "Entry-level", "Intern", "Trainee", "Junior", "Associate",
    "Mid", "Mid-level", "Senior", "Lead", "Principal", "Staff",
])

# Pre-compiled seniority patterns (word-boundary \b, not lookahead/lookbehind,
# to match the original extract_seniority behaviour).
COMPILED_SENIORITY: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), kw)
    for kw in SENIORITY
]


# Recognised countries (canonical title-case form) used to validate / surface a
# location when only the country name appears in the text.
COUNTRIES: list[str] = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina",
    "Australia", "Austria", "Bahamas", "Bahrain", "Bangladesh", "Barbados",
    "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon",
    "Canada", "Chile", "Colombia", "Comoros", "Congo", "Costa Rica",
    "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti",
    "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador",
    "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia",
    "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany",
    "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guyana", "Haiti",
    "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq",
    "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Kiribati", "Korea", "Kuwait", "Kyrgyzstan", "Laos", "Latvia",
    "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania",
    "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali",
    "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico",
    "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco",
    "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands",
    "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia",
    "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea",
    "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar",
    "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and "
    "Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra "
    "Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
    "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka",
    "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan",
    "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu",
    "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom",
    "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City",
    "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
]
COUNTRY_LOOKUP: dict[str, str] = {c.lower(): c for c in COUNTRIES}

# Countries pre-sorted longest-first so multi-word names are matched before
# single-word ones (replaces the per-call ``sorted()`` in extract_location).
SORTED_COUNTRIES: list[str] = sorted(COUNTRIES, key=len, reverse=True)

# Pre-compiled country patterns for the location fallback search.
COMPILED_COUNTRIES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(c) + r"\b", re.IGNORECASE), c)
    for c in SORTED_COUNTRIES
]


# Structural location phrases. Kept conservative: only explicit location
# prepositions ("based in", "located in", "from") are accepted, capturing a
# chain of Capitalized tokens separated by spaces/commas (so "Berlin, Germany"
# or "San Francisco, CA" is captured as one place).  This avoids mistaking
# "SQL, Python" or "in Python," for a place; bare countries are caught by the
# COUNTRIES fallback.
LOCATION_PATTERNS: list[re.Pattern] = [
    # Compiled case-sensitively: the captured place tokens MUST be Capitalized
    # (the core signal).  Only the leading preposition is matched
    # case-insensitively via the inline ``(?i:...)`` group.
    re.compile(
        r"(?i:(?:based\s+in|located\s+in|from))\s+"
        r"([A-Z][A-Za-z]+(?:[\s,\-]+[A-Z][A-Za-z]+)*)"
    )
]


# Structural company-name phrases.
COMPANY_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(?:joining|position\s+(?:at|with))\s+(.+?)(?:[\.\n,]|$)",
        r"[Rr]ole\s+at\s+(.+?)(?:[\.\n,]|$)",
        r"work\s+at\s+(.+?)(?:[\.\n,]|$)",
    ]
]


# Tokens that may trail a captured place / company and should be trimmed.
LOCATION_TRAILING_WORDS: set[str] = {
    "position", "role", "opportunity", "job", "opening", "team", "teams",
    "department", "division", "remote", "hybrid",
}


# ── Additional candidate fields ────────────────────────────────────────────────
# Date / availability (e.g. "can start 2025-01-15", "available immediately").
# Each pattern captures the date or the ASAP/immediately keyword directly.
START_DATE_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(?:can|ready to|could)\s+start\s+(?:on|by|from)?\s*"
        r"(\d{4}[/-]\d{1,2}[/-]\d{1,2}"
        r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|\b(?:immediately|asap)\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:earliest\s+)?(?:start|availability)\s*[:=]?\s*"
        r"(\d{4}[/-]\d{1,2}[/-]\d{1,2}"
        r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|\b(?:immediately|asap)\b)",
        re.IGNORECASE,
    ),
]

# Work mode (surface forms → canonical). First match wins.
WORK_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"\b(work[-\s]?from[-\s]?home|wfh|fully\s+remote|remote)\b",
        re.IGNORECASE), "remote"),
    (re.compile(r"\b(hybrid|flexible)\b", re.IGNORECASE), "hybrid"),
    (re.compile(r"\b(on[-\s]?site|onsite|in[-\s]?office)\b", re.IGNORECASE),
     "onsite"),
]

# Languages mentioned in the body (order of first appearance, de-duplicated).
LANGUAGES: list[str] = _dedupe_keyword_list([
    "English", "Spanish", "French", "German", "Italian", "Portuguese",
    "Mandarin", "Cantonese", "Chinese", "Japanese", "Korean", "Hindi",
    "Arabic", "Russian", "Dutch", "Swedish", "Norwegian", "Finnish",
    "Polish", "Turkish", "Greek", "Hebrew", "Thai", "Vietnamese",
])

# Certifications / professional credentials (order of first appearance).
CERTIFICATIONS: list[str] = _dedupe_keyword_list([
    "PMP", "CAPM", "CISSP", "CISM", "CISA", "CEH", "AWS Certified",
    "Azure Certified", "Google Cloud Certified", "CCNA", "CCNP", "CCIE",
    "Google Ads Certified", "Salesforce Certified", "Scrum Master", "CSM",
    "Six Sigma", "ITIL", "Oracle Certified",
])


# ── Pre-compiled keyword pattern lists ─────────────────────────────────────────
# Each entry is (compiled_regex, canonical_keyword).  Pre-compiled once at import
# time so the extraction layer never pays re.compile() per keyword per email.
# The lookarounds use (?<![A-Za-z0-9])…(?![A-Za-z0-9]) instead of \b so that
# punctuation-bearing tokens like "C++", "C#", "CI/CD", "Node.js" match and
# so that "SQL" is not found inside "MySQL".
def _compile_keywords(keywords: list[str]) -> list[tuple[re.Pattern, str]]:
    return [
        (re.compile(r"(?<![A-Za-z0-9])" + re.escape(kw) + r"(?![A-Za-z0-9])",
                     re.IGNORECASE), kw)
        for kw in keywords
    ]


COMPILED_SKILLS: list[tuple[re.Pattern, str]] = _compile_keywords(SKILLS)
COMPILED_EDUCATION: list[tuple[re.Pattern, str]] = _compile_keywords(EDUCATION)
COMPILED_LANGUAGES: list[tuple[re.Pattern, str]] = _compile_keywords(LANGUAGES)
COMPILED_CERTIFICATIONS: list[tuple[re.Pattern, str]] = _compile_keywords(CERTIFICATIONS)


# ---------------------------------------------------------------------------
# 8. Additional candidate fields (headers + body patterns)
# ---------------------------------------------------------------------------

# Current company / employer
CURRENT_COMPANY_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(?:currently|presently)\s+(?:at|with)\s+([A-Z][A-Za-z0-9&.\-']{1,80}?)(?:\s*[.,;]|$|\s+(?:as|since|for))",
        r"(?:currently|presently)\s+(?:a|an|the)?\s+[A-Z][A-Za-z\s\-]{2,80}?\s+(?:at|with)\s+([A-Z][A-Za-z0-9&.\-']{1,80}?)(?:\s*[.,;]|$|\s+(?:as|since|for))",
        r"(?:working|employed)\s+(?:at|for|with)\s+([A-Z][A-Za-z0-9&.\-']{1,80}?)(?:\s*[.,;]|$|\s+(?:as|since|for))",
        r"current\s+(?:employer|company|organisation|organization)\s*[:=]?\s*([A-Z][A-Za-z0-9&.\-']{1,80})",
        r"my\s+current\s+(?:role|position)\s+is\s+(?:at|with)\s+([A-Z][A-Za-z0-9&.\-']{1,80})",
    ]
]

# Current role / title
CURRENT_ROLE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(?:currently|presently)\s+(?:a|an|the)\s+([A-Z][A-Za-z\s\-]{2,80}?)(?:\s+at|\s+with|\s*,|\.|$)",
        r"(?:currently|presently)\s+(?:working\s+as|employed\s+as)?\s+([A-Z][A-Za-z\s\-]{2,80}?)(?:\s+at|\s+with|\s*,|\.|$)",
        r"(?:working|employed)\s+as\s+(?:a|an)?\s*([A-Z][A-Za-z\s\-]{2,80}?)(?:\s+at|\s+with|\s*,|\.|$)",
        r"current\s+(?:role|title|position)\s*[:=]?\s*([A-Z][A-Za-z\s\-]{2,80})",
        r"my\s+(?:current|present)\s+(?:role|title|position)\s+(?:is|:)\s*([A-Z][A-Za-z\s\-]{2,80})",
    ]
]

# Visa / work authorization status
VISA_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(H-?1B|H1B|L-?1|L1|O-?1|O1|TN|E-?3|E3|F-?1|F1|J-?1|J1|H-?4|H4|L-?2|L2|OPT|CPT|STEM\s+OPT)\b",
        r"\b(green\s+card|permanent\s+resident|US\s+citizen|U\.S\.\s*citizen|work\s+authorization|work\s+permit|visa\s+status|right\s+to\s+work)\b",
        r"\b(require\s+sponsorship|need\s+sponsorship|visa\s+sponsorship|sponsorship\s+required)\b",
        r"\b(eligible\s+to\s+work|authorized\s+to\s+work|legally\s+authorized)\b",
    ]
]

# Relocation willingness
RELOCATION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(willing\s+to\s+relocate|open\s+to\s+relocation|relocation\s+(?:possible|available|ok|yes)|can\s+relocate|happy\s+to\s+relocate)\b",
        r"\b(not\s+willing\s+to\s+relocate|cannot\s+relocate|unwilling\s+to\s+relocate|no\s+relocation|relocation\s+not\s+possible)\b",
        r"relocation\s*[:=]?\s*(yes|no|maybe|negotiable|preferred)",
    ]
]

# Travel willingness
TRAVEL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(willing\s+to\s+travel|open\s+to\s+travel|travel\s+(?:ok|yes|possible|up\s+to))\b",
        r"\b(not\s+willing\s+to\s+travel|cannot\s+travel|no\s+travel|travel\s+not\s+possible)\b",
        r"(?:up\s+to|around|about)\s*(\d{1,2})\s*%\s*travel",
        r"travel\s*[:=]?\s*(\d{1,2})\s*%",
        r"travel\s*[:=]?\s*(yes|no|minimal|moderate|extensive|negotiable)",
    ]
]

# Security clearance
SECURITY_CLEARANCE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(security\s+clearance|clearance\s+level)\s*[:=]?\s*(top\s+secret|secret|confidential|public\s+trust|TS|SCI|SAP|DV|SC|CTC|NATO\s+secret|NATO\s+cosmic)\b",
        r"\b(top\s+secret|secret\s+clearance|confidential\s+clearance|public\s+trust\s+clearance|TS\s+clearance|SCI\s+clearance|DV\s+clearance|SC\s+clearance)\b",
        r"\b(active\s+clearance|current\s+clearance|eligible\s+for\s+clearance)\b",
    ]
]

# GPA
GPA_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bGPA\s*[:=]?\s*(\d\.\d{1,2}(?:\/\d\.\d{1,2})?)",
        r"\b(\d\.\d{1,2})\s*(?:\/|out\s+of)\s*4\.0",
        r"\b(first\s+class\s+honours?|second\s+class\s+upper|second\s+class\s+lower|upper\s+second|lower\s+second|distinction|merit|pass)\b",
    ]
]

# Graduation year
GRADUATION_YEAR_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(?:graduat(?:e|ing)|class\s+of|complet(?:e|ing))\s+(?:in\s+)?(20\d{2}|19\d{2})\b",
        r"\b(20\d{2}|19\d{2})\s+graduat(?:e|ion)\b",
        r"\bexpected\s+graduation\s*[:=]?\s*(20\d{2}|19\d{2})\b",
    ]
]

# References
REFERENCES_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\breferences?\s+(?:available|upon\s+request|on\s+request|provided\s+upon\s+request)\b",
        r"\breferences?\s*[:=]\s*(available|upon\s+request|attached|included)\b",
        r"\b(reference\s+contacts?|professional\s+references?)\s+(?:available|upon\s+request)\b",
    ]
]

# Job ID / Reference number
JOB_ID_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(?:job|position|role|req|requisition)\s*(?:id|reference|ref|number|#)\s*[:=#]?\s*([A-Z0-9\-]{3,30})\b",
        r"\b(?:ref|reference)\s*[:=#]\s*([A-Z0-9\-]{3,30})\b",
        r"#([A-Z0-9\-]{4,30})\b",
    ]
]

# Team / Department
TEAM_DEPARTMENT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(?:team|department|division|group|unit)\s*[:=]?\s*([A-Z][A-Za-z\s\-]{2,80}?)(?:\s*[.,;]|$)",
        r"(?:join|joining)\s+(?:the\s+)?([A-Z][A-Za-z\s\-]{2,80}?)\s+(?:team|department|group)",
    ]
]

# Hiring manager name
HIRING_MANAGER_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"dear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*,",
        r"(?:hiring\s+manager|recruiter)\s*[:=]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        r"attn[\.:]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
    ]
]

# Email headers we want to extract
HEADER_FIELDS: list[str] = [
    "date", "message-id", "in-reply-to", "references", "to", "cc", "reply-to",
]

