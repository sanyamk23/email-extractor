"""Extract structured candidate details (name, contact, links, metrics)."""
from __future__ import annotations

import email
import re

from . import config
from .email_utils import name_from_from_header

# `phonenumbers` is an optional dependency. When present it gives strict E.164
# validation/formatting; otherwise we fall back to the built-in regex normaliser
# so the package has zero required runtime dependencies.
try:  # pragma: no cover - exercised only when phonenumbers is installed
    import phonenumbers
    _HAS_PHONENUMBERS = True
except ImportError:  # pragma: no cover
    _HAS_PHONENUMBERS = False


def _trim_title_from_name(name: str) -> str:
    """If a captured sign-off name ends with a job-title word, drop it.

    e.g. ``"Jane Doe Engineer"`` -> ``"Jane Doe"``.
    """
    if not name:
        return name
    parts = name.split()
    while len(parts) > 2 and parts[-1].lower() in config.TITLE_WORDS:
        parts.pop()
    return " ".join(parts)


def _find_signoff_name(body: str) -> str | None:
    """Return the best closing sign-off name found in ``body``, or ``None``.

    Markdown bold/italic markers (e.g. ``*Priyanshu Raj*``) wrapping the name
    are tolerated and stripped.  When several sign-off phrases match, the
    fullest (most-token) name is preferred, so a phrase such as
    "Thank you.\\n\\nRegards,\\nJane Doe" resolves to "Jane Doe" rather than
    the stray "Regards" captured by the "Thank you" phrase.
    """
    candidates: list[str] = []
    for pattern in config.SIGNOFF_PATTERNS:
        matches = list(pattern.finditer(body))
        if matches:
            raw = matches[-1].group(1).strip().strip("*_").strip()
            if raw:
                candidates.append(_trim_title_from_name(raw))
    if not candidates:
        return None
    # Prefer the fullest name (most tokens, then longest) so a 2-token
    # "Jane Doe" beats a 1-token "Regards" captured by another phrase.
    return max(candidates, key=lambda n: (len(n.split()), len(n)))


def _is_proxy_sender(from_name: str, signoff_name: str | None, body: str) -> bool:
    """True when the ``From:`` display name is a forwarder/proxy, not the author.

    A 2+ token From name that does **not** appear anywhere in the body, while a
    *different* 2+ token name signed the message in the body's closing sign-off,
    means the From header belongs to whoever relayed the message — the real
    candidate is the body's sign-off author (e.g. a recruiter submitting an
    application on a candidate's behalf).
    """
    if not from_name or "@" in from_name:
        return False
    if len(from_name.split()) < 2:
        return False
    if not signoff_name or len(signoff_name.split()) < 2:
        return False
    if signoff_name == from_name:
        return False
    return from_name not in body


def extract_name(from_header: str, body: str, signoff_name: str | None = None) -> str | None:
    """Determine the candidate's full name.

    Strategy:
    1. Display name from the ``From:`` header (most reliable) — unless it is a
       proxy/forwarder whose name is absent from the body while a different
       fuller name signed the message, in which case the sign-off author wins.
    2. Closing sign-off in the body.
    3. Signature line ``"Name <email>"`` / bare name near the bottom.
    """
    # 1. From header (most reliable).
    from_name = name_from_from_header(from_header)
    if from_name and "@" not in from_name:
        from_name = _trim_title_from_name(from_name)
        if signoff_name is None:
            signoff_name = _find_signoff_name(body)
        if len(from_name.split()) >= 2:
            # A 2+ token From name wins outright, unless it's a proxy/forwarder.
            if _is_proxy_sender(from_name, signoff_name, body) and signoff_name:
                return signoff_name
            return from_name
        # Single-token From name: recover a fuller (2+ token) sign-off if any.
        if signoff_name:
            return signoff_name
        return from_name

    # 2. Closing sign-off in the body.
    if signoff_name is None:
        signoff_name = _find_signoff_name(body)
    if signoff_name:
        return signoff_name

    # 3. Signature-style "Name <email>" near the end.
    m = re.search(
        r"([A-Z][A-Za-z'’.\-]{1,30}(?:\s+[A-Z][A-Za-z'’.\-]{1,30}){0,2})"
        r"\s*<[^>]+@[^>]+>",
        body,
    )
    if m:
        raw = m.group(1).strip()
        return _trim_title_from_name(raw)

    # 4. Last non-empty line that looks like a bare name.
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    for ln in reversed(lines[-6:]):
        if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z.]+){1,2}", ln) \
                and "@" not in ln and len(ln.split()) <= 3:
            return _trim_title_from_name(ln)

    return None


def extract_email(from_header: str, body: str, signoff_name: str | None = None) -> str | None:
    """Extract the candidate's primary email address.

    The From header is preferred — unless it belongs to a proxy/forwarder
    (its name is absent from the body while a different name signed the
    message), in which case that address is the forwarder's and we fall back
    to an address found inside the message body instead.
    """
    from_name = name_from_from_header(from_header)
    if from_name and "@" not in from_name:
        if signoff_name is None:
            signoff_name = _find_signoff_name(body)
        if not _is_proxy_sender(from_name, signoff_name, body):
            m = config.EMAIL_REGEX.search(from_header)
            if m:
                return m.group(0)
    m = config.EMAIL_REGEX.search(body)
    if m:
        return m.group(0)
    return None


def extract_sender(from_header: str) -> dict:
    """Return the *message sender's* ``{name, email}`` parsed from the From header.

    This is the actual sender of the email (who mailed it), distinct from the
    candidate. When the sender is a recruiter/forwarder who submitted an
    application on the candidate's behalf, their contact details are real,
    extractable data and are surfaced here rather than mis-attributed to the
    candidate.
    """
    name = name_from_from_header(from_header)
    if name and "@" in name:
        # Bare address with no display name carries no human name.
        name = None
    email = None
    m = config.EMAIL_REGEX.search(from_header or "")
    if m:
        email = m.group(0)
    return {"name": name, "email": email}


def extract_phones(body: str) -> list[str]:
    """Return de-duplicated, normalised phone numbers found in the body."""
    phones: list[str] = []
    seen: set[str] = set()
    for match in config.PHONE_REGEX.finditer(body):
        raw = match.group(0)
        # Require a minimum digit count so years/short numbers are rejected.
        digits = config.DIGITS_ONLY.sub("", raw)
        if len(digits) < 7:
            continue
        normalised = _format_phone(raw)
        if normalised and normalised not in seen:
            seen.add(normalised)
            phones.append(normalised)
    return phones


def _format_phone(raw: str) -> str:
    """Return a canonical phone string, preferring ``phonenumbers`` when available."""
    if _HAS_PHONENUMBERS:
        try:
            parsed = phonenumbers.parse(raw, None)
            if (phonenumbers.is_possible_number(parsed)
                    and phonenumbers.is_valid_number(parsed)):
                return phonenumbers.format_number(
                    parsed, phonenumbers.PhoneNumberFormat.E164)
        except Exception:  # malformed input — degrade gracefully
            pass
    return _normalise_phone(raw)


def _normalise_phone(raw: str) -> str:
    """Normalise a phone string to a canonical ``+<country><digits>`` form."""
    digits = config.DIGITS_ONLY.sub("", raw)
    has_plus = raw.lstrip().startswith("+") or "+00" in raw
    if not digits:
        return ""
    # Drop leading zeroes of a country code that were written with 00/.
    if has_plus:
        return "+" + digits
    # No explicit country code — assume US default for 10-digit numbers.
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    if has_plus:
        return "+" + digits
    return digits


def _is_personal_url(url: str) -> bool:
    """True when a URL is plausibly the candidate's own portfolio/site."""
    host = re.match(r"https?://([^/]+)", url, re.IGNORECASE)
    if not host:
        return False
    domain = host.group(1).lower().lstrip("www.")
    if domain in config.NON_PERSONAL_DOMAINS:
        return False
    if any(domain.endswith("." + d) or d in domain for d in
           {"linkedin", "github", "twitter", "facebook"}):
        return False
    return True


def extract_links(body: str) -> list[str]:
    """Return LinkedIn, GitHub and personal/portfolio URLs (de-duplicated).

    LinkedIn URLs come first, followed by GitHub, then personal/portfolio
    sites (any URL not already captured and not a known social platform).
    """
    links: list[str] = []
    seen: set[str] = set()

    # Social-platform finders (highest priority).
    for finder in (config.LINKEDIN_REGEX, config.GITHUB_REGEX):
        for match in finder.finditer(body):
            url = match.group(0).strip().rstrip(",.)>")
            if url not in seen:
                seen.add(url)
                links.append(url)

    # Personal/portfolio sites — every URL that is not LinkedIn/GitHub and is
    # not a recognised social platform.
    for match in config.URL_REGEX.finditer(body):
        url = match.group(0).strip().rstrip(",.)>")
        low = url.lower()
        if low.startswith(("https://linkedin", "http://linkedin",
                           "https://github", "http://github")):
            continue
        if _is_personal_url(url) and url not in seen:
            seen.add(url)
            links.append(url)

    return links


def extract_years_of_experience(body: str) -> str | None:
    """Return the candidate's years of experience as a string, or None."""
    best: str | None = None
    best_val = -1.0
    for pattern in config.EXPERIENCE_PATTERNS:
        for match in pattern.finditer(body):
            num = match.group(1)
            try:
                val = float(num)
            except ValueError:
                continue
            # Prefer the largest, most explicit figure.
            if val > best_val:
                best_val = val
                best = num
    return best


def extract_salary(body: str) -> str | None:
    """Return a normalised salary expectation string, or None."""
    match = config.SALARY_REGEX.search(body)
    if not match:
        return None
    amount = match.group(1)
    amount = amount.replace(",", "").rstrip(".")
    return amount if amount else None


def extract_notice_period(body: str) -> str | None:
    """Return a normalised notice-period string, e.g. '30 days', or None."""
    for pattern in config.NOTICE_PERIOD_REGEXES:
        match = pattern.search(body)
        if match:
            number = match.group(1)
            unit = match.group(2)
            return f"{number} {unit}"
    return None


def _extract_keywords(body: str, compiled_keywords: list[tuple[re.Pattern, str]],
                      limit: int = 10) -> list[str]:
    """Return canonical keyword hits in ``body`` in source order.

    ``compiled_keywords`` must be ordered longest-first (as produced by
    :func:`config._dedupe_keyword_list`) so multi-word phrases are consumed
    before any shorter sub-token they contain.  Matched spans are blanked in a
    working copy (length-preserving) so a keyword such as "SQL" is not also
    reported under "SQL Server".  De-duplication is case-insensitive while
    canonical capitalisation is preserved.

    Lookarounds ``(?<![A-Za-z0-9])``/``(?![A-Za-z0-9])`` are used instead of
    ``\\b`` so that tokens containing punctuation (e.g. ``C++``, ``C#``,
    ``CI/CD``, ``Node.js``) match correctly, and so that ``SQL`` is not matched
    inside ``MySQL``.
    """
    if not body:
        return []
    found: list[tuple[int, str]] = []
    seen: set[str] = set()
    buf = body
    for pattern, canon in compiled_keywords:
        match = pattern.search(buf)
        if not match:
            continue
        if canon.lower() not in seen:
            seen.add(canon.lower())
            found.append((match.start(), canon))
        # Blank the matched span so shorter sub-tokens can't also match it.
        buf = buf[:match.start()] + "\x00" * (match.end() - match.start()) + buf[match.end():]
    found.sort(key=lambda item: item[0])
    return [canon for _, canon in found][:limit]


def extract_skills(body: str) -> list[str]:
    """Return technical skills / technologies mentioned in ``body``."""
    if not body:
        return []
    return _extract_keywords(body, config.COMPILED_SKILLS, limit=15)


def extract_education(body: str) -> list[str]:
    """Return education-level keywords mentioned in ``body``."""
    if not body:
        return []
    return _extract_keywords(body, config.COMPILED_EDUCATION, limit=5)


def extract_seniority(body: str) -> str | None:
    """Return the first seniority-level keyword found in ``body``, or None."""
    for pattern, kw in config.COMPILED_SENIORITY:
        if pattern.search(body):
            return kw
    return None


def _clean_entity(text: str) -> str | None:
    """Trim and validate a captured place/company name.

    Rejects captures that read like prose: every whitespace token (after
    punctuation stripping) must start with an uppercase letter, which is a
    strong place-name signal and lets us discard phrases such as "your team"
    or "Data Analyst".  Trailing role/team words are dropped first.
    """
    if not text:
        return None
    text = text.strip().strip(" ,.;:-")
    parts = text.split()
    while len(parts) > 1 and parts[-1].lower() in config.LOCATION_TRAILING_WORDS:
        parts.pop()
        text = " ".join(parts)
    text = text.strip().strip(" ,.;:-")
    if not text:
        return None
    toks = [t.strip(",") for t in text.split() if t.strip(",")]
    if not toks or not all(t[:1].isupper() for t in toks):
        return None
    # A place/company name is short: reject long runs (likely prose) that
    # slipped past the capitalization guard.
    if len(toks) > 4:
        return None
    return text or None


def extract_location(body: str) -> str | None:
    """Return a location (place or country) mentioned in ``body``, or None."""
    if not body:
        return None
    for pattern in config.LOCATION_PATTERNS:
        match = pattern.search(body)
        if match and match.group(1):
            loc = _clean_entity(match.group(1))
            if loc:
                return loc
    # Fallback: a recognised country name anywhere in the body.
    for pattern, country in config.COMPILED_COUNTRIES:
        if pattern.search(body):
            return country
    return None


def extract_company(body: str) -> str | None:
    """Return a company name mentioned via a structural phrase, or None."""
    if not body:
        return None
    for pattern in config.COMPANY_PATTERNS:
        match = pattern.search(body)
        if match and match.group(1):
            company = _clean_entity(match.group(1))
            if company:
                return company
    return None


def extract_start_date(body: str) -> str | None:
    """Return the first availability / start-date mention in ``body``, or None.

    Returns the captured date token (e.g. ``"2025-01-15"``) or the literal
    ``"immediately"`` / ``"ASAP"`` when those keywords are present.
    """
    if not body:
        return None
    for pattern in config.START_DATE_PATTERNS:
        match = pattern.search(body)
        if match:
            return match.group(1)
    return None


def extract_work_type(body: str) -> str | None:
    """Return a canonical work mode (remote/hybrid/onsite) or None."""
    if not body:
        return None
    for pattern, canonical in config.WORK_TYPE_PATTERNS:
        if pattern.search(body):
            return canonical
    return None


def extract_languages(body: str) -> list[str]:
    """Return languages mentioned in ``body`` in source order (de-duplicated)."""
    if not body:
        return []
    return _extract_keywords(body, config.COMPILED_LANGUAGES, limit=10)


def extract_certifications(body: str) -> list[str]:
    """Return professional certifications mentioned in ``body`` in source order."""
    if not body:
        return []
    return _extract_keywords(body, config.COMPILED_CERTIFICATIONS, limit=15)


def _clean_entity(text: str | None) -> str | None:
    """Trim and validate a captured entity (company, role, team, etc.)."""
    if not text:
        return None
    text = text.strip().strip(" ,.;:-")
    if not text:
        return None
    # Reject if it looks like prose (not all tokens capitalized)
    toks = [t.strip(",") for t in text.split() if t.strip(",")]
    if not toks or not all(t[:1].isupper() for t in toks):
        return None
    if len(toks) > 6:
        return None
    return text


def extract_current_company(body: str) -> str | None:
    """Return the candidate's current company/employer, or None."""
    if not body:
        return None
    for pattern in config.CURRENT_COMPANY_PATTERNS:
        match = pattern.search(body)
        if match and match.group(1):
            company = _clean_entity(match.group(1))
            if company:
                return company
    return None


def extract_current_role(body: str) -> str | None:
    """Return the candidate's current role/title, or None."""
    if not body:
        return None
    for pattern in config.CURRENT_ROLE_PATTERNS:
        match = pattern.search(body)
        if match and match.group(1):
            role = _clean_entity(match.group(1))
            if role:
                # Filter out common false positives
                role_lower = role.lower()
                if role_lower in {"the", "a", "an", "my", "your", "our", "this", "that"}:
                    continue
                return role
    return None


def extract_visa_status(body: str) -> str | None:
    """Return visa/work authorization status, or None."""
    if not body:
        return None
    for pattern in config.VISA_PATTERNS:
        match = pattern.search(body)
        if match:
            # Return the matched text (canonicalize common forms)
            text = match.group(0).lower()
            if "h1b" in text or "h-1b" in text:
                return "H-1B"
            if "l1" in text or "l-1" in text:
                return "L-1"
            if "opt" in text:
                return "OPT"
            if "cpt" in text:
                return "CPT"
            if "green card" in text or "permanent resident" in text:
                return "Green Card"
            if "citizen" in text:
                return "US Citizen"
            if "sponsor" in text:
                return "Requires Sponsorship"
            if "authorized" in text or "eligible" in text:
                return "Authorized to Work"
            return match.group(0)
    return None


def extract_relocation(body: str) -> str | None:
    """Return relocation willingness: 'yes', 'no', 'maybe', or None."""
    if not body:
        return None
    for pattern in config.RELOCATION_PATTERNS:
        match = pattern.search(body)
        if match:
            text = match.group(0).lower()
            if any(w in text for w in ["not willing", "cannot", "unwilling", "no relocation", "not possible"]):
                return "no"
            if any(w in text for w in ["willing", "open to", "possible", "ok", "yes", "happy to", "can relocate"]):
                return "yes"
            # Check for explicit yes/no in capture group
            if match.lastindex and match.group(1):
                val = match.group(1).lower()
                if val in {"yes", "no", "maybe", "negotiable", "preferred"}:
                    return val
            return "yes"
    return None


def extract_travel(body: str) -> str | None:
    """Return travel willingness: percentage, 'yes', 'no', 'minimal', etc., or None."""
    if not body:
        return None
    for pattern in config.TRAVEL_PATTERNS:
        match = pattern.search(body)
        if match:
            text = match.group(0).lower()
            if any(w in text for w in ["not willing", "cannot", "no travel", "not possible"]):
                return "no"
            if any(w in text for w in ["willing", "open to", "ok", "yes"]):
                return "yes"
            # Percentage capture
            if match.lastindex and match.group(1) and match.group(1).isdigit():
                return f"{match.group(1)}%"
            # Explicit level capture
            if match.lastindex and match.group(1):
                val = match.group(1).lower()
                if val in {"yes", "no", "minimal", "moderate", "extensive", "negotiable"}:
                    return val
            return "yes"
    return None


def extract_security_clearance(body: str) -> str | None:
    """Return security clearance level, or None."""
    if not body:
        return None
    for pattern in config.SECURITY_CLEARANCE_PATTERNS:
        match = pattern.search(body)
        if match:
            text = match.group(0).lower()
            if "top secret" in text or "ts" == text.strip() or "ts " in text:
                return "Top Secret"
            if "secret" in text and "top" not in text:
                return "Secret"
            if "confidential" in text:
                return "Confidential"
            if "public trust" in text:
                return "Public Trust"
            if "dv" in text:
                return "DV"
            if "sc " in text or " sc" == text.strip():
                return "SC"
            if "ctc" in text:
                return "CTC"
            if "nato" in text:
                return "NATO"
            if "active" in text or "current" in text or "eligible" in text:
                return "Active/Eligible"
            return match.group(0)
    return None


def extract_gpa(body: str) -> str | None:
    """Return GPA or degree classification, or None."""
    if not body:
        return None
    for pattern in config.GPA_PATTERNS:
        match = pattern.search(body)
        if match:
            if match.lastindex and match.group(1):
                val = match.group(1)
                # If it's a classification, return as-is
                if any(w in val.lower() for w in ["class", "honours", "distinction", "merit", "pass"]):
                    return val
                # Otherwise it's a numeric GPA
                return val
            return match.group(0)
    return None


def extract_graduation_year(body: str) -> str | None:
    """Return graduation year (YYYY), or None."""
    if not body:
        return None
    for pattern in config.GRADUATION_YEAR_PATTERNS:
        match = pattern.search(body)
        if match and match.group(1):
            year = match.group(1)
            if year.isdigit() and 1950 <= int(year) <= 2035:
                return year
    return None


def extract_references(body: str) -> str | None:
    """Return references availability: 'available', 'upon_request', or None."""
    if not body:
        return None
    for pattern in config.REFERENCES_PATTERNS:
        match = pattern.search(body)
        if match:
            text = match.group(0).lower()
            if "upon request" in text or "on request" in text:
                return "upon_request"
            return "available"
    return None


def extract_job_id(body: str) -> str | None:
    """Return job ID / requisition number, or None."""
    if not body:
        return None
    for pattern in config.JOB_ID_PATTERNS:
        match = pattern.search(body)
        if match and match.group(1):
            return match.group(1).upper()
    return None


def extract_team_department(body: str) -> str | None:
    """Return team/department name, or None."""
    if not body:
        return None
    for pattern in config.TEAM_DEPARTMENT_PATTERNS:
        match = pattern.search(body)
        if match and match.group(1):
            team = _clean_entity(match.group(1))
            if team:
                return team
    return None


def extract_hiring_manager(body: str) -> str | None:
    """Return hiring manager name from salutation, or None."""
    if not body:
        return None
    for pattern in config.HIRING_MANAGER_PATTERNS:
        match = pattern.search(body)
        if match and match.group(1):
            name = match.group(1).strip()
            # Filter out generic salutations
            if name.lower() not in {"hiring manager", "sir", "madam", "team", "all", "recruiter"}:
                return name
    return None


def extract_email_headers(raw_email: str) -> dict:
    """Extract selected headers from raw RFC-822 email string."""
    headers = {}
    if not raw_email:
        return headers
    # Only parse if it looks like RFC-822 (starts with Header: value)
    first_line = next((ln for ln in raw_email.splitlines() if ln.strip()), "")
    if not re.match(r"^[A-Za-z][A-Za-z0-9-]*\s*:", first_line):
        return headers

    try:
        msg = email.message_from_string(raw_email)
        for field in config.HEADER_FIELDS:
            val = msg.get(field)
            if val:
                headers[field.replace("-", "_")] = val.strip()
    except Exception:
        pass
    return headers


def extract_candidate(from_header: str, body: str) -> dict:
    """Assemble the full ``candidate`` sub-document."""
    signoff_name = _find_signoff_name(body)
    return {
        "name": extract_name(from_header, body, signoff_name),
        "email": extract_email(from_header, body, signoff_name),
        "phone": extract_phones(body),
        "links": extract_links(body),
        "years_of_experience": extract_years_of_experience(body),
        "salary_expectation": extract_salary(body),
        "notice_period": extract_notice_period(body),
        "skills": extract_skills(body),
        "education": extract_education(body),
        "seniority": extract_seniority(body),
        "location": extract_location(body),
        "company": extract_company(body),
        "start_date": extract_start_date(body),
        "work_type": extract_work_type(body),
        "languages": extract_languages(body),
        "certifications": extract_certifications(body),
        "current_company": extract_current_company(body),
        "current_role": extract_current_role(body),
        "visa_status": extract_visa_status(body),
        "relocation": extract_relocation(body),
        "travel": extract_travel(body),
        "security_clearance": extract_security_clearance(body),
        "gpa": extract_gpa(body),
        "graduation_year": extract_graduation_year(body),
        "references": extract_references(body),
        "job_id": extract_job_id(body),
        "team_department": extract_team_department(body),
        "hiring_manager": extract_hiring_manager(body),
    }
