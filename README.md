# Email Extraction Engine

A **generic, zero-cost Email Extraction Engine** in Python. Extracts structured data from `.eml` files or raw email text based on a topic/requirement string.

## Key Features

- **Zero LLM calls** — No OpenAI, Anthropic, Gemini, Ollama, or local generative LLMs
- **Zero runtime token costs** — Pure local Python with lightweight ML/NLP libraries
- **Two extraction paths:**
  1. **Manual Field Map/Registry** (`core/templates.py`) — Pre-defined topic → field mappings for common domains (job applications, DMARC reports, invoices, etc.)
  2. **Dynamic Fallback Discovery** (`core/dynamic_schema.py`) — Automatically derives target fields from email text using semantic similarity when a topic isn't in the registry

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
cat sample.eml | python main.py --stdin --requirement "invoice" --pretty

# With custom GLiNER threshold
python main.py --eml sample.eml --requirement "invoice" --threshold 0.5
```

## Architecture

```
email_extractor_engine/
├── requirements.txt
├── main.py              # CLI Entry Point
├── core/
│   ├── __init__.py
│   ├── eml_parser.py    # EML loader, body text decoder, table cleaner
│   ├── templates.py     # Dictionary of topics and manually assigned field lists
│   ├── dynamic_schema.py # SentenceTransformer fallback for unmapped topics
│   ├── gliner_extractor.py # Local GLiNER zero-shot model wrapper
│   └── engine.py        # Main orchestrator linking parser, registry, and extractor
└── tests/
    └── test_engine.py   # Unit test suite verifying manual & dynamic extraction
```

### Extraction Flow

1. **Parse** the email (`.eml` file or raw RFC-822 text) into a `ParsedEmail` structure
2. **Resolve the topic** — Check the manual registry (`core/templates.py`) for a matching topic
3. **Determine fields** — If matched, use pre-defined fields; otherwise, dynamically discover fields via semantic similarity
4. **Extract values** — Feed target fields and email body into GLiNER (or regex fallback) for zero-shot extraction
5. **Return structured JSON** mapping extracted values to fields

## Supported Topics (Manual Registry)

| Topic | Fields |
|-------|--------|
| `job application` | candidate_name, applicant_email, target_role, years_experience, education, expected_salary, ... |
| `dmarc report` | target_domain, reporting_period, submitter_email, source_ip, total_messages, ... |
| `invoice` | invoice_number, invoice_date, due_date, total_amount, vendor_name, ... |
| `e-commerce order` | order_number, order_date, customer_name, items, total_amount, ... |
| `event invitation` | event_title, event_date, start_time, end_time, location, ... |
| `travel itinerary` | booking_reference, passenger_name, flight_number, origin, destination, ... |
| `support ticket` | ticket_id, ticket_subject, ticket_status, priority, ... |
| `meeting minutes` | meeting_title, meeting_date, attendees, action_items, ... |
| `purchase order` | po_number, po_date, buyer_name, supplier_name, ... |
| `delivery notice` | tracking_number, carrier, delivery_status, ... |
| `interview scheduling` | candidate_name, interviewer_name, interview_date, ... |
| `contract` | contract_id, contract_date, parties, effective_date, ... |
| `newsletter` | newsletter_title, newsletter_date, author_name, ... |

Any topic not in the registry automatically falls back to dynamic field discovery.

## Testing

```bash
python -m pytest tests/ -v
```
