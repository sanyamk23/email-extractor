# Email Extraction Engine

A generic, zero-cost Email Extraction Engine in Python. Extracts structured data from `.eml` files or raw email text based on a topic/requirement string.

## Key Features

- **Zero LLM calls** — No OpenAI, Anthropic, Gemini, Ollama, or local generative LLMs
- **Zero runtime token costs** — Pure local Python with lightweight ML/NLP libraries
- **Two-tier field resolution** — Known topics (e.g. "job application", "dmarc report", "invoice") resolve to a manual field registry; arbitrary topics fall back to dynamic discovery from the email body

## Installation

```bash
cd email_extractor_engine
pip install -r requirements.txt
```

### Optional Dependencies

All ML backends are **optional**. The engine runs with zero runtime dependencies using regex-based fallback extraction:

```bash
# Install with all optional ML backends
pip install -r requirements.txt
pip install gliner torch sentence-transformers beautifulsoup4 python-dateutil
```

## Usage

### Python API

```python
from core.engine import TopicEmailExtractor

extractor = TopicEmailExtractor()

# Extract from a .eml file
result = extractor.extract(eml_source="sample.eml", topic="job application")
print(result.to_json(indent=2))

# Extract from raw email text
result = extractor.extract(
    eml_source="sample.eml",
    topic="job application",
    raw_text="From: Sarah Jenkins <s.jenkins@email.com>\nSubject: Application\n\n..."
)
print(result.to_dict())
```

### CLI

```bash
# From a file (defaults to "job application")
python main.py --eml sample.eml --pretty

# Specify a topic via --requirement (or --topic alias)
python main.py --eml sample.eml --requirement "dmarc report" --pretty

# From stdin
cat sample.eml | python main.py --stdin --requirement "vendor inspection" --pretty

# With custom GLiNER threshold
python main.py --eml sample.eml --requirement "invoice" --threshold 0.5
```

## Architecture

```
email_extractor_engine/
├── requirements.txt
├── main.py              # CLI Entry Point (--requirement / --topic)
├── core/
│   ├── __init__.py
│   ├── templates.py     # Manual field registry: topic → field lists + alias resolution
│   ├── eml_parser.py    # EML loader, body text decoder, table cleaner
│   ├── dynamic_schema.py # Dynamic field discovery (KV labels, noun phrases, topic-token inference)
│   ├── gliner_extractor.py # Local GLiNER zero-shot model wrapper
│   └── engine.py        # Main orchestrator linking parser, registry, dynamic discovery, and extractor
└── tests/
    └── test_engine.py   # Unit test suite
```

### Extraction Flow

1. **Parse** the email (`.eml` file or raw RFC-822 text) into a `ParsedEmail` structure
2. **Resolve topic** — Check the manual registry (`core/templates.py`) for a matching topic or alias. If matched, use the pre-defined field list (e.g. `candidate_name`, `job_role`, `years_experience` for "job application"). Otherwise, fall back to dynamic discovery.
3. **Discover fields** (fallback only) — Derive field names from:
   - Key-value pair labels in the body (e.g. `Auditor: Mark Vance` → `auditor_name`)
   - Topic-token inference — common field names associated with the topic's words
   - Noun-phrase labels from the body, ranked by semantic similarity to the topic
4. **Extract values** — Feed target field names and email body into GLiNER (or regex fallback) for zero-shot extraction
5. **Return structured JSON** mapping extracted values to fields

### Field Name Generation

For **known topics** (registry path), field names are pre-defined in `core/templates.py`
(e.g. `candidate_name`, `applicant_email`, `job_role` for "job application").

For **arbitrary topics** (dynamic discovery fallback), field names are generated:

- **KV-pair labels**: `Auditor: Mark Vance` → field name `auditor_name` (label "Auditor" → snake_case + semantic suffix inference)
- **Topic tokens**: The topic string's words (e.g. "audit", "report") map to common business fields (e.g. `audit_date`, `findings`, `report_title`)
- **Noun phrases**: Capitalized phrases in the body are semantically ranked against the topic
- **GLiNER labels**: Field names → human-readable labels by replacing underscores with spaces (e.g. `candidate_name` → `candidate name`)

## Examples

### Job Application

```python
result = extractor.extract(
    eml_source="job.eml",
    topic="job application",
    raw_text="""From: Sarah Jenkins <s.jenkins@email.com>
To: hr@techcompany.com
Subject: Job Application - Senior Python Engineer

Hi HR, applying for the Senior Python Engineer role. Name: Sarah Jenkins (s.jenkins@email.com). 6 years experience, B.S. in Computer Science from MIT. Expected salary: $130,000/yr."""
)
```

Output:
```json
{
  "topic": "job application",
  "field_source": "manual_registry",
  "extracted_data": {
    "candidate_name": "Sarah Jenkins",
    "applicant_email": "s.jenkins@email.com",
    "target_role": "Senior Python Engineer",
    "years_experience": "6 years",
    "education": "B.S. in Computer Science from MIT",
    "expected_salary": "$130,000/yr"
  }
}
```

### Any Arbitrary Topic

The engine works with any topic — no manual registry needed:

```python
result = extractor.extract(
    eml_source="inspection.eml",
    topic="vendor inspection",
    raw_text="""From: quality@auditor.com
To: vendor@mfg.com
Subject: Vendor Inspection Report

Vendor Quality Check Report #8841. Auditor: Mark Vance. Facility: Detroit Plant 4. Status: CONDITIONAL PASS. Defects Found: 3 minor paint scratches."""
)
```

Output:
```json
{
  "topic": "vendor inspection",
  "field_source": "dynamic_discovery",
  "extracted_data": {
    "auditor_name": "Mark Vance",
    "facility_location": "Detroit Plant 4",
    "status": "CONDITIONAL PASS",
    "defects_found": "3 minor paint scratches"
  }
}
```

## Testing

```bash
python -m pytest tests/ -v
```
