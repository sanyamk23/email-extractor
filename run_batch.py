"""Run the job-application parser over every ``.eml`` in a folder and print a
one-line summary per file, so you can eyeball results across a real corpus.

Usage:
    python run_batch.py                 # globs ./mails/*.eml
    python run_batch.py path/to/folder  # custom folder
    python run_batch.py '*.eml'         # custom glob

Each row shows: classification, confidence, job_role, candidate name/email/
phone/links, years, salary, notice, start_date/work_type/languages/certifications,
attachments, then the file path.
"""
import glob
import os
import sys

from email_extractor.pipeline import parse_job_application


def _summarise(raw: str) -> dict:
    return parse_job_application(raw)


def _row(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            r = _summarise(fh.read())
    except Exception as exc:
        return f"ERROR {path}: {type(exc).__name__}: {exc}\n"

    c = r.get("candidate", {})
    phone = (c.get("phone") or ["-"])[0]
    app = "APP" if r.get("is_job_application") else "no"
    s = r.get("sender") or {}
    if s.get("email"):
        sender = f"{s.get('name')} <{s.get('email')}>"
    else:
        sender = s.get("name")
    return (
        f"{app:3} conf={r.get('confidence_score', 0.0):.2f} "
        f"role={r.get('job_role')}\n"
        f"     name={c.get('name')} email={c.get('email')} phone={phone} "
        f"sender={sender}\n"
        f"     yoe={c.get('years_of_experience')} salary={c.get('salary_expectation')} "
        f"notice={c.get('notice_period')}\n"
        f"     skills={c.get('skills')} education={c.get('education')} "
        f"seniority={c.get('seniority')} location={c.get('location')} "
        f"company={c.get('company')}\n"
        f"     start_date={c.get('start_date')} work_type={c.get('work_type')} "
        f"languages={c.get('languages')} certifications={c.get('certifications')}\n"
        f"     links={c.get('links')}\n"
        f"     attachments={len(r.get('attachments') or [])}"
        f" {[a.get('filename', '') for a in r.get('attachments') or []]}\n"
        f"     {path}\n"
    )


def main(argv: list[str]) -> int:
    pattern = argv[0] if argv else os.path.join("mails", "*.eml")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no .eml files matched: {pattern}")
        return 0
    print(f"{len(files)} file(s) matched {pattern}\n")
    for f in files:
        print(_row(f))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
