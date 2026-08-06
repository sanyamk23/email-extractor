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

from email_extractor import parse_job_application


def _summarise(raw: str) -> dict:
    return parse_job_application(raw)


def _row(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        r = _summarise(fh.read())
    c = r["candidate"]
    phone = (c["phone"] or ["-"])[0]
    app = "APP" if r["is_job_application"] else "no"
    s = r.get("sender") or {}
    if s.get("email"):
        sender = f"{s.get('name')} <{s.get('email')}>"
    else:
        sender = s.get("name")
    return (
        f"{app:3} conf={r['confidence_score']:.2f} role={r['job_role']}\n"
        f"     name={c['name']} email={c['email']} phone={phone} sender={sender}\n"
        f"     yoe={c['years_of_experience']} salary={c['salary_expectation']} "
        f"notice={c['notice_period']}\n"
        f"     skills={c['skills']} education={c['education']} "
        f"seniority={c['seniority']} location={c['location']} company={c['company']}\n"
        f"     start_date={c['start_date']} work_type={c['work_type']} "
        f"languages={c['languages']} certifications={c['certifications']}\n"
        f"     links={c['links']}\n"
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
