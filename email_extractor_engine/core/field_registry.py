"""Comprehensive Semantic Field Registry.

This module provides LLM-like field generation capabilities by mapping semantic
concepts (from topics/queries) to comprehensive field lists with extraction
patterns. It replaces the need for manual per-topic registration by using a
rich, data-driven approach.

Key design principles:
1. Fields are generated from semantic understanding of the topic
2. Each field has associated extraction patterns (regex, KV keys, GLiNER labels)
3. The registry is comprehensive enough to handle ANY topic/query
4. No LLM calls - pure Python with optional ML backends
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FieldSpec:
    """Specification for a single extraction field."""
    name: str
    label: str
    kv_keys: list[str] = field(default_factory=list)
    regex_patterns: list[str] = field(default_factory=list)
    gliner_label: str = ""
    field_type: str = "text"
    description: str = ""
    priority: int = 10
    
    def __post_init__(self):
        if not self.gliner_label:
            self.gliner_label = self.name.replace("_", " ")
        if not self.kv_keys:
            self.kv_keys = [self.label, self.label.title()]


@dataclass
class DomainConcept:
    """A semantic domain concept that generates fields."""
    name: str
    keywords: list[str]
    fields: list[FieldSpec]
    sub_concepts: list[str] = field(default_factory=list)
    description: str = ""


# ── Universal Field Specifications ─────────────────────────────────────────────
# These are fields that can appear in ANY email domain

UNIVERSAL_FIELDS = [
    # Identity & Contact
    FieldSpec("sender_name", "Sender Name", ["from", "sender", "sent by"], field_type="person"),
    FieldSpec("sender_email", "Sender Email", ["from email", "sender email", "email address"], field_type="email"),
    FieldSpec("recipient_name", "Recipient Name", ["to", "recipient", "dear"], field_type="person"),
    FieldSpec("recipient_email", "Recipient Email", ["to email", "recipient email"], field_type="email"),
    FieldSpec("cc_recipients", "CC Recipients", ["cc", "carbon copy"], field_type="email_list"),
    FieldSpec("bcc_recipients", "BCC Recipients", ["bcc"], field_type="email_list"),
    
    # Temporal
    FieldSpec("date", "Date", ["date", "sent date", "email date"], field_type="date"),
    FieldSpec("time", "Time", ["time", "sent time"], field_type="time"),
    FieldSpec("datetime", "DateTime", ["datetime", "timestamp"], field_type="datetime"),
    
    # Subject & Metadata
    FieldSpec("subject", "Subject", ["subject", "re", "regarding"], field_type="text"),
    FieldSpec("message_id", "Message ID", ["message id", "message-id"], field_type="identifier"),
    FieldSpec("thread_id", "Thread ID", ["thread id", "conversation id"], field_type="identifier"),
    FieldSpec("references", "References", ["references", "in-reply-to"], field_type="identifier_list"),
    FieldSpec("priority", "Priority", ["priority", "importance", "urgency"], field_type="enum"),
    FieldSpec("category", "Category", ["category", "type", "classification"], field_type="enum"),
    
    # Attachments
    FieldSpec("attachment_count", "Attachment Count", ["attachments", "attached files"], field_type="integer"),
    FieldSpec("attachment_names", "Attachment Names", ["attachment names", "attached"], field_type="text_list"),
]


# ── Domain-Specific Field Specifications ──────────────────────────────────────

JOB_APPLICATION_FIELDS = [
    # Candidate Identity
    FieldSpec("candidate_name", "Candidate Name", ["candidate name", "applicant name", "name", "full name"], field_type="person", priority=1),
    FieldSpec("applicant_email", "Applicant Email", ["email", "email address", "contact email"], field_type="email", priority=1),
    FieldSpec("phone_number", "Phone Number", ["phone", "phone number", "mobile", "telephone", "contact number"], field_type="phone", priority=2),
    FieldSpec("linkedin_url", "LinkedIn URL", ["linkedin", "linkedin profile", "linkedin url"], field_type="url", priority=3),
    FieldSpec("github_url", "GitHub URL", ["github", "github profile", "github url"], field_type="url", priority=3),
    FieldSpec("portfolio_url", "Portfolio URL", ["portfolio", "portfolio url", "website", "personal website"], field_type="url", priority=3),
    
    # Role & Position
    FieldSpec("target_role", "Target Role", ["position", "role", "job title", "position applied for", "job role", "title"], field_type="role", priority=1),
    FieldSpec("job_id", "Job ID", ["job id", "requisition id", "req id", "job number"], field_type="identifier", priority=2),
    FieldSpec("department", "Department", ["department", "team", "division"], field_type="text", priority=3),
    
    # Experience & Qualifications
    FieldSpec("years_experience", "Years Experience", ["experience", "years of experience", "years exp", "total experience"], field_type="experience", priority=1),
    FieldSpec("current_role", "Current Role", ["current position", "current role", "present position", "current title"], field_type="role", priority=2),
    FieldSpec("current_company", "Current Company", ["current company", "current employer", "present employer", "employer"], field_type="organization", priority=2),
    FieldSpec("previous_companies", "Previous Companies", ["previous companies", "past employers", "work history"], field_type="organization_list", priority=3),
    FieldSpec("seniority_level", "Seniority Level", ["seniority", "level", "grade", "experience level"], field_type="seniority", priority=2),
    FieldSpec("education", "Education", ["education", "degree", "qualification", "academic background"], field_type="education", priority=2),
    FieldSpec("skills", "Skills", ["skills", "technical skills", "competencies", "expertise", "technologies"], field_type="skills", priority=2),
    FieldSpec("certifications", "Certifications", ["certifications", "certificates", "credentials"], field_type="text_list", priority=3),
    FieldSpec("languages", "Languages", ["languages", "spoken languages", "language proficiency"], field_type="text_list", priority=3),
    
    # Compensation & Availability
    FieldSpec("expected_salary", "Expected Salary", ["expected salary", "salary expectation", "desired salary", "compensation", "salary requirement"], field_type="money", priority=2),
    FieldSpec("notice_period", "Notice Period", ["notice period", "availability", "available from", "start date", "joining date"], field_type="notice_period", priority=2),
    FieldSpec("work_type", "Work Type", ["work type", "remote", "hybrid", "onsite", "work arrangement", "location preference"], field_type="work_type", priority=3),
    FieldSpec("relocation", "Relocation", ["relocation", "willing to relocate", "relocate"], field_type="boolean", priority=3),
    
    # References & Additional
    FieldSpec("references", "References", ["references", "professional references", "referees"], field_type="text_list", priority=3),
    FieldSpec("cover_letter", "Cover Letter", ["cover letter", "motivation letter", "personal statement"], field_type="text", priority=3),
]


INVOICE_FIELDS = [
    FieldSpec("invoice_number", "Invoice Number", ["invoice number", "invoice #", "inv #", "bill number"], field_type="identifier", priority=1),
    FieldSpec("invoice_date", "Invoice Date", ["invoice date", "date", "issued date", "billing date"], field_type="date", priority=1),
    FieldSpec("due_date", "Due Date", ["due date", "payment due", "pay by"], field_type="date", priority=1),
    FieldSpec("total_amount", "Total Amount", ["total", "amount due", "grand total", "balance due", "total amount"], field_type="money", priority=1),
    FieldSpec("subtotal", "Subtotal", ["subtotal", "sub total", "net amount"], field_type="money", priority=2),
    FieldSpec("tax_amount", "Tax Amount", ["tax", "vat", "gst", "sales tax", "tax amount"], field_type="money", priority=2),
    FieldSpec("tax_rate", "Tax Rate", ["tax rate", "vat rate", "gst rate"], field_type="percentage", priority=3),
    FieldSpec("discount_amount", "Discount", ["discount", "discount amount", "deduction"], field_type="money", priority=3),
    FieldSpec("shipping_fee", "Shipping Fee", ["shipping", "delivery", "freight", "shipping fee"], field_type="money", priority=3),
    FieldSpec("vendor_name", "Vendor Name", ["vendor", "supplier", "seller", "from", "billed by"], field_type="organization", priority=1),
    FieldSpec("vendor_email", "Vendor Email", ["vendor email", "supplier email"], field_type="email", priority=2),
    FieldSpec("vendor_address", "Vendor Address", ["vendor address", "supplier address", "remit to"], field_type="address", priority=3),
    FieldSpec("customer_name", "Customer Name", ["customer", "client", "bill to", "sold to"], field_type="organization", priority=1),
    FieldSpec("customer_email", "Customer Email", ["customer email", "client email"], field_type="email", priority=2),
    FieldSpec("customer_address", "Customer Address", ["customer address", "client address", "ship to"], field_type="address", priority=3),
    FieldSpec("payment_method", "Payment Method", ["payment method", "payment terms", "how to pay"], field_type="text", priority=2),
    FieldSpec("payment_status", "Payment Status", ["status", "payment status", "paid", "unpaid", "partial"], field_type="enum", priority=2),
    FieldSpec("purchase_order", "Purchase Order", ["po number", "purchase order", "po #", "order number"], field_type="identifier", priority=2),
    FieldSpec("vat_number", "VAT Number", ["vat number", "vat id", "tax id", "ein"], field_type="identifier", priority=3),
    FieldSpec("currency", "Currency", ["currency", "curr"], field_type="currency", priority=2),
    FieldSpec("line_items", "Line Items", ["items", "line items", "description", "products", "services"], field_type="line_items", priority=1),
]


DMARC_REPORT_FIELDS = [
    FieldSpec("target_domain", "Target Domain", ["domain", "target domain", "policy domain"], field_type="domain", priority=1),
    FieldSpec("reporting_period", "Reporting Period", ["reporting period", "period", "date range", "begin", "end"], field_type="date_range", priority=1),
    FieldSpec("submitter_email", "Submitter Email", ["submitter", "reported by", "email"], field_type="email", priority=1),
    FieldSpec("source_ip", "Source IP", ["source ip", "ip", "sending ip", "originating ip"], field_type="ip", priority=1),
    FieldSpec("total_messages", "Total Messages", ["total", "count", "messages", "total messages"], field_type="integer", priority=1),
    FieldSpec("passed_messages", "Passed Messages", ["passed", "pass", "dmarc pass", "success"], field_type="integer", priority=1),
    FieldSpec("failed_messages", "Failed Messages", ["failed", "fail", "dmarc fail", "reject"], field_type="integer", priority=1),
    FieldSpec("dmarc_policy", "DMARC Policy", ["policy", "dmarc policy", "p=", "policy published"], field_type="enum", priority=1),
    FieldSpec("policy_dkim", "DKIM Policy", ["dkim", "dkim policy", "dkim pass", "dkim fail"], field_type="enum", priority=2),
    FieldSpec("policy_spf", "SPF Policy", ["spf", "spf policy", "spf pass", "spf fail"], field_type="enum", priority=2),
    FieldSpec("auth_results", "Authentication Results", ["auth results", "authentication", "results", "arc"], field_type="text", priority=2),
    FieldSpec("organization_name", "Organization Name", ["org name", "organization", "reporting org"], field_type="organization", priority=2),
    FieldSpec("report_id", "Report ID", ["report id", "report_id"], field_type="identifier", priority=2),
]


ECOMMERCE_ORDER_FIELDS = [
    FieldSpec("order_number", "Order Number", ["order number", "order #", "order id", "confirmation number"], field_type="identifier", priority=1),
    FieldSpec("order_date", "Order Date", ["order date", "placed on", "date ordered"], field_type="date", priority=1),
    FieldSpec("customer_name", "Customer Name", ["customer", "name", "billing name"], field_type="person", priority=1),
    FieldSpec("customer_email", "Customer Email", ["email", "customer email"], field_type="email", priority=1),
    FieldSpec("items", "Items", ["items", "products", "order items", "line items"], field_type="line_items", priority=1),
    FieldSpec("quantities", "Quantities", ["quantity", "qty", "quantities"], field_type="integer_list", priority=2),
    FieldSpec("prices", "Prices", ["price", "unit price", "prices"], field_type="money_list", priority=2),
    FieldSpec("subtotal", "Subtotal", ["subtotal", "sub total", "merchandise total"], field_type="money", priority=2),
    FieldSpec("tax_amount", "Tax", ["tax", "sales tax", "vat", "estimated tax"], field_type="money", priority=2),
    FieldSpec("shipping_fee", "Shipping", ["shipping", "shipping & handling", "delivery", "shipping fee"], field_type="money", priority=2),
    FieldSpec("discount_amount", "Discount", ["discount", "promo", "coupon", "savings"], field_type="money", priority=3),
    FieldSpec("total_amount", "Total", ["total", "order total", "grand total", "amount charged"], field_type="money", priority=1),
    FieldSpec("payment_method", "Payment Method", ["payment", "card", "payment method", "paid with"], field_type="text", priority=2),
    FieldSpec("payment_status", "Payment Status", ["status", "payment status"], field_type="enum", priority=2),
    FieldSpec("shipping_address", "Shipping Address", ["shipping address", "ship to", "delivery address"], field_type="address", priority=2),
    FieldSpec("billing_address", "Billing Address", ["billing address", "bill to"], field_type="address", priority=3),
    FieldSpec("shipping_carrier", "Carrier", ["carrier", "shipped via", "shipping method"], field_type="text", priority=3),
    FieldSpec("tracking_number", "Tracking Number", ["tracking", "tracking number", "track #"], field_type="identifier", priority=2),
    FieldSpec("estimated_delivery", "Estimated Delivery", ["estimated delivery", "delivery by", "arrives by"], field_type="date", priority=3),
]


SUPPORT_TICKET_FIELDS = [
    FieldSpec("ticket_id", "Ticket ID", ["ticket id", "case number", "case #", "incident id", "request id"], field_type="identifier", priority=1),
    FieldSpec("ticket_subject", "Subject", ["subject", "summary", "title", "issue"], field_type="text", priority=1),
    FieldSpec("ticket_status", "Status", ["status", "state", "current status"], field_type="enum", priority=1),
    FieldSpec("ticket_priority", "Priority", ["priority", "urgency", "severity", "impact"], field_type="enum", priority=1),
    FieldSpec("customer_name", "Customer Name", ["customer", "requester", "reporter", "name"], field_type="person", priority=1),
    FieldSpec("customer_email", "Customer Email", ["email", "customer email", "requester email"], field_type="email", priority=1),
    FieldSpec("issue_description", "Description", ["description", "details", "issue", "problem", "error"], field_type="text", priority=1),
    FieldSpec("assigned_agent", "Assigned Agent", ["assigned to", "agent", "owner", "assignee"], field_type="person", priority=2),
    FieldSpec("created_date", "Created Date", ["created", "opened", "submitted", "reported on"], field_type="date", priority=2),
    FieldSpec("resolved_date", "Resolved Date", ["resolved", "closed", "resolved on"], field_type="date", priority=2),
    FieldSpec("category", "Category", ["category", "type", "classification", "component"], field_type="enum", priority=2),
    FieldSpec("tags", "Tags", ["tags", "labels", "keywords"], field_type="text_list", priority=3),
    FieldSpec("resolution", "Resolution", ["resolution", "solution", "fix", "root cause"], field_type="text", priority=2),
]


CONTRACT_FIELDS = [
    FieldSpec("contract_id", "Contract ID", ["contract id", "contract #", "agreement number"], field_type="identifier", priority=1),
    FieldSpec("contract_date", "Contract Date", ["date", "contract date", "execution date", "signed date"], field_type="date", priority=1),
    FieldSpec("parties", "Parties", ["parties", "between", "party", "signatories"], field_type="organization_list", priority=1),
    FieldSpec("effective_date", "Effective Date", ["effective", "effective date", "commencement", "start date"], field_type="date", priority=1),
    FieldSpec("expiration_date", "Expiration Date", ["expiration", "expiry", "end date", "termination date"], field_type="date", priority=1),
    FieldSpec("contract_value", "Contract Value", ["value", "amount", "contract value", "total value", "consideration"], field_type="money", priority=1),
    FieldSpec("currency", "Currency", ["currency", "curr"], field_type="currency", priority=2),
    FieldSpec("contract_type", "Contract Type", ["type", "agreement type", "contract type"], field_type="enum", priority=2),
    FieldSpec("governing_law", "Governing Law", ["governing law", "jurisdiction", "law"], field_type="text", priority=2),
    FieldSpec("termination_clause", "Termination Clause", ["termination", "termination clause", "notice period"], field_type="text", priority=2),
    FieldSpec("signatures", "Signatures", ["signed by", "signatures", "executed by"], field_type="text_list", priority=2),
]


EVENT_INVITATION_FIELDS = [
    FieldSpec("event_title", "Event Title", ["event", "title", "subject", "meeting", "webinar"], field_type="text", priority=1),
    FieldSpec("event_date", "Event Date", ["date", "when", "event date", "meeting date"], field_type="date", priority=1),
    FieldSpec("start_time", "Start Time", ["time", "start time", "begins at", "starts at"], field_type="time", priority=1),
    FieldSpec("end_time", "End Time", ["end time", "ends at", "until", "finishes at"], field_type="time", priority=2),
    FieldSpec("location", "Location", ["location", "where", "venue", "address", "meeting link", "zoom", "teams"], field_type="location", priority=1),
    FieldSpec("organizer", "Organizer", ["organizer", "host", "hosted by", "presented by"], field_type="person", priority=2),
    FieldSpec("organizer_email", "Organizer Email", ["organizer email", "host email", "contact"], field_type="email", priority=2),
    FieldSpec("meeting_link", "Meeting Link", ["link", "meeting link", "join", "url", "zoom link", "teams link"], field_type="url", priority=2),
    FieldSpec("agenda", "Agenda", ["agenda", "schedule", "topics", "program"], field_type="text", priority=2),
    FieldSpec("rsvp_email", "RSVP Email", ["rsvp", "reply to", "confirm to"], field_type="email", priority=3),
    FieldSpec("attendees", "Attendees", ["attendees", "participants", "guests"], field_type="person_list", priority=3),
]


TRAVEL_ITINERARY_FIELDS = [
    FieldSpec("booking_reference", "Booking Reference", ["booking reference", "confirmation", "pnr", "record locator", "booking #"], field_type="identifier", priority=1),
    FieldSpec("passenger_name", "Passenger Name", ["passenger", "traveler", "name", "guest"], field_type="person", priority=1),
    FieldSpec("flight_number", "Flight Number", ["flight", "flight number", "flight #"], field_type="identifier", priority=1),
    FieldSpec("origin", "Origin", ["from", "origin", "departure", "departing from"], field_type="location", priority=1),
    FieldSpec("destination", "Destination", ["to", "destination", "arrival", "arriving at"], field_type="location", priority=1),
    FieldSpec("departure_datetime", "Departure", ["departure", "departs", "departure time", "leave"], field_type="datetime", priority=1),
    FieldSpec("arrival_datetime", "Arrival", ["arrival", "arrives", "arrival time", "land"], field_type="datetime", priority=1),
    FieldSpec("seat_number", "Seat", ["seat", "seat number"], field_type="text", priority=2),
    FieldSpec("airline", "Airline", ["airline", "carrier", "operated by"], field_type="organization", priority=1),
    FieldSpec("booking_status", "Status", ["status", "confirmation", "confirmed", "booking status"], field_type="enum", priority=2),
    FieldSpec("ticket_number", "Ticket Number", ["ticket", "ticket number", "e-ticket"], field_type="identifier", priority=2),
    FieldSpec("travel_class", "Class", ["class", "cabin", "travel class", "fare class"], field_type="enum", priority=2),
    FieldSpec("departure_terminal", "Departure Terminal", ["terminal", "departure terminal", "gate"], field_type="text", priority=3),
    FieldSpec("arrival_terminal", "Arrival Terminal", ["arrival terminal", "terminal"], field_type="text", priority=3),
    FieldSpec("baggage", "Baggage", ["baggage", "bags", "checked bags", "luggage"], field_type="text", priority=3),
]


MEETING_MINUTES_FIELDS = [
    FieldSpec("meeting_title", "Meeting Title", ["meeting", "title", "subject", "call"], field_type="text", priority=1),
    FieldSpec("meeting_date", "Meeting Date", ["date", "meeting date"], field_type="date", priority=1),
    FieldSpec("start_time", "Start Time", ["start", "start time", "began at"], field_type="time", priority=1),
    FieldSpec("end_time", "End Time", ["end", "end time", "ended at", "adjourned"], field_type="time", priority=2),
    FieldSpec("attendees", "Attendees", ["attendees", "present", "participants", "present:", "attending"], field_type="person_list", priority=1),
    FieldSpec("absentees", "Absentees", ["absent", "apologies", "absentees"], field_type="person_list", priority=3),
    FieldSpec("agenda_items", "Agenda Items", ["agenda", "items", "topics", "discussion items"], field_type="text_list", priority=1),
    FieldSpec("action_items", "Action Items", ["action items", "actions", "todos", "follow up", "next steps"], field_type="text_list", priority=1),
    FieldSpec("decisions_made", "Decisions", ["decisions", "resolved", "agreed", "decided"], field_type="text_list", priority=2),
    FieldSpec("meeting_owner", "Meeting Owner", ["organizer", "chair", "facilitator", "host"], field_type="person", priority=2),
    FieldSpec("location", "Location", ["location", "venue", "meeting link", "room"], field_type="location", priority=2),
    FieldSpec("recording_link", "Recording", ["recording", "video", "recording link"], field_type="url", priority=3),
]


PURCHASE_ORDER_FIELDS = [
    FieldSpec("po_number", "PO Number", ["po number", "purchase order", "po #", "order number"], field_type="identifier", priority=1),
    FieldSpec("po_date", "PO Date", ["date", "po date", "order date", "issued date"], field_type="date", priority=1),
    FieldSpec("buyer_name", "Buyer", ["buyer", "purchaser", "ordered by", "customer"], field_type="organization", priority=1),
    FieldSpec("buyer_email", "Buyer Email", ["buyer email", "purchaser email"], field_type="email", priority=2),
    FieldSpec("supplier_name", "Supplier", ["supplier", "vendor", "seller", "ship from"], field_type="organization", priority=1),
    FieldSpec("supplier_email", "Supplier Email", ["supplier email", "vendor email"], field_type="email", priority=2),
    FieldSpec("total_amount", "Total Amount", ["total", "amount", "order total", "grand total"], field_type="money", priority=1),
    FieldSpec("currency", "Currency", ["currency", "curr"], field_type="currency", priority=2),
    FieldSpec("line_items", "Line Items", ["items", "line items", "products", "services", "description"], field_type="line_items", priority=1),
    FieldSpec("delivery_date", "Delivery Date", ["delivery", "delivery date", "ship by", "required by"], field_type="date", priority=2),
    FieldSpec("payment_terms", "Payment Terms", ["terms", "payment terms", "net", "pay terms"], field_type="text", priority=2),
    FieldSpec("shipping_address", "Shipping Address", ["ship to", "shipping address", "delivery address"], field_type="address", priority=2),
    FieldSpec("billing_address", "Billing Address", ["bill to", "billing address"], field_type="address", priority=3),
]


DELIVERY_NOTICE_FIELDS = [
    FieldSpec("tracking_number", "Tracking Number", ["tracking", "tracking number", "track #", "tracking code"], field_type="identifier", priority=1),
    FieldSpec("carrier", "Carrier", ["carrier", "shipping carrier", "courier", "delivered by"], field_type="organization", priority=1),
    FieldSpec("delivery_status", "Status", ["status", "delivery status", "state"], field_type="enum", priority=1),
    FieldSpec("origin", "Origin", ["from", "origin", "shipped from"], field_type="location", priority=2),
    FieldSpec("destination", "Destination", ["to", "destination", "delivered to", "address"], field_type="location", priority=1),
    FieldSpec("estimated_delivery", "Estimated Delivery", ["estimated delivery", "delivery by", "expected", "arrives by"], field_type="date", priority=2),
    FieldSpec("actual_delivery", "Actual Delivery", ["delivered", "actual delivery", "delivery date", "signed for"], field_type="date", priority=2),
    FieldSpec("recipient_name", "Recipient", ["recipient", "received by", "signed by", "name"], field_type="person", priority=2),
    FieldSpec("parcel_weight", "Weight", ["weight", "parcel weight", "mass"], field_type="text", priority=3),
    FieldSpec("number_of_packages", "Packages", ["packages", "parcels", "number of packages", "items"], field_type="integer", priority=3),
]


INTERVIEW_SCHEDULING_FIELDS = [
    FieldSpec("candidate_name", "Candidate Name", ["candidate", "applicant", "name", "interviewee"], field_type="person", priority=1),
    FieldSpec("candidate_email", "Candidate Email", ["candidate email", "applicant email", "email"], field_type="email", priority=1),
    FieldSpec("interviewer_name", "Interviewer Name", ["interviewer", "interviewed by", "panel", "host"], field_type="person", priority=1),
    FieldSpec("interviewer_email", "Interviewer Email", ["interviewer email", "panel email"], field_type="email", priority=2),
    FieldSpec("interview_date", "Interview Date", ["date", "interview date", "when", "scheduled for"], field_type="date", priority=1),
    FieldSpec("start_time", "Start Time", ["time", "start time", "begins at", "at"], field_type="time", priority=1),
    FieldSpec("end_time", "End Time", ["end time", "ends at", "until", "duration"], field_type="time", priority=2),
    FieldSpec("location", "Location", ["location", "where", "venue", "meeting link", "zoom", "teams", "room"], field_type="location", priority=1),
    FieldSpec("meeting_link", "Meeting Link", ["link", "meeting link", "join", "url", "video link"], field_type="url", priority=2),
    FieldSpec("job_role", "Job Role", ["role", "position", "job", "title", "interview for"], field_type="role", priority=1),
    FieldSpec("interview_round", "Round", ["round", "stage", "interview round", "phase"], field_type="text", priority=2),
    FieldSpec("interview_stage", "Stage", ["stage", "phase", "type", "screening", "technical", "final"], field_type="enum", priority=2),
    FieldSpec("duration", "Duration", ["duration", "length", "minutes", "hours"], field_type="duration", priority=2),
    FieldSpec("status", "Status", ["status", "confirmed", "pending", "cancelled", "rescheduled"], field_type="enum", priority=2),
]


NEWSLETTER_FIELDS = [
    FieldSpec("newsletter_title", "Newsletter Title", ["title", "subject", "newsletter", "issue"], field_type="text", priority=1),
    FieldSpec("newsletter_date", "Date", ["date", "published", "issue date"], field_type="date", priority=1),
    FieldSpec("author_name", "Author", ["author", "by", "written by", "from"], field_type="person", priority=1),
    FieldSpec("author_email", "Author Email", ["author email", "contact", "reply to"], field_type="email", priority=2),
    FieldSpec("section_titles", "Sections", ["sections", "articles", "topics", "contents", "in this issue"], field_type="text_list", priority=1),
    FieldSpec("unsubscribe_link", "Unsubscribe", ["unsubscribe", "opt out", "preferences"], field_type="url", priority=3),
    FieldSpec("issue_number", "Issue Number", ["issue", "issue #", "volume", "edition"], field_type="identifier", priority=2),
]


# ── Domain Concept Registry ────────────────────────────────────────────────────

DOMAIN_CONCEPTS: dict[str, DomainConcept] = {
    "job_application": DomainConcept(
        name="job_application",
        keywords=["job application", "job applications", "job app", "career application", "job inquiry", 
                  "employment application", "position application", "apply for job", "job seeker",
                  "candidate application", "resume", "cv", "cover letter"],
        fields=JOB_APPLICATION_FIELDS,
        description="Job applications, resumes, cover letters, candidate submissions"
    ),
    "invoice": DomainConcept(
        name="invoice",
        keywords=["invoice", "invoices", "billing", "bill", "receipt", "receipts", 
                  "financial report", "financial reports", "fee", "charge", "bill payment",
                  "accounts payable", "accounts receivable", "proforma"],
        fields=INVOICE_FIELDS,
        description="Invoices, bills, receipts, financial statements"
    ),
    "dmarc_report": DomainConcept(
        name="dmarc_report",
        keywords=["dmarc report", "dmarc reports", "dmarc aggregate report",
                  "dmarc forensic report", "domain report", "email authentication report",
                  "spf/dmarc report", "dmarc aggregate", "dmarc forensic"],
        fields=DMARC_REPORT_FIELDS,
        description="DMARC aggregate and forensic reports"
    ),
    "ecommerce_order": DomainConcept(
        name="ecommerce_order",
        keywords=["e-commerce order", "ecommerce order", "order confirmation",
                  "order", "purchase confirmation", "purchase", "shop order",
                  "online order", "order receipt", "order summary"],
        fields=ECOMMERCE_ORDER_FIELDS,
        description="E-commerce order confirmations, purchase receipts"
    ),
    "support_ticket": DomainConcept(
        name="support_ticket",
        keywords=["support ticket", "support request", "help desk", "incident report",
                  "ticket", "customer support", "service request", "it ticket",
                  "support case", "help request", "technical support"],
        fields=SUPPORT_TICKET_FIELDS,
        description="Support tickets, help desk requests, incident reports"
    ),
    "contract": DomainConcept(
        name="contract",
        keywords=["contract", "agreement", "nda", "non-disclosure agreement",
                  "service agreement", "employment contract", "master agreement",
                  "mou", "memorandum of understanding", "legal agreement"],
        fields=CONTRACT_FIELDS,
        description="Contracts, agreements, NDAs, legal documents"
    ),
    "event_invitation": DomainConcept(
        name="event_invitation",
        keywords=["event invitation", "event invite", "meeting invitation",
                  "calendar invite", "conference invitation", "webinar invitation",
                  "event registration", "save the date", "invitation"],
        fields=EVENT_INVITATION_FIELDS,
        description="Event invitations, meeting invites, conference registrations"
    ),
    "travel_itinerary": DomainConcept(
        name="travel_itinerary",
        keywords=["travel itinerary", "travel booking", "flight confirmation",
                  "trip confirmation", "itinerary", "travel confirmation",
                  "flight itinerary", "booking confirmation", "travel plans"],
        fields=TRAVEL_ITINERARY_FIELDS,
        description="Travel itineraries, flight confirmations, booking details"
    ),
    "meeting_minutes": DomainConcept(
        name="meeting_minutes",
        keywords=["meeting minutes", "meeting notes", "meeting summary",
                  "minutes", "board meeting", "meeting record", "meeting log",
                  "action items", "meeting recap"],
        fields=MEETING_MINUTES_FIELDS,
        description="Meeting minutes, notes, summaries, action items"
    ),
    "purchase_order": DomainConcept(
        name="purchase_order",
        keywords=["purchase order", "po", "purchase order form", "procurement",
                  "po request", "buy order", "purchasing"],
        fields=PURCHASE_ORDER_FIELDS,
        description="Purchase orders, procurement requests"
    ),
    "delivery_notice": DomainConcept(
        name="delivery_notice",
        keywords=["delivery notice", "shipping notification", "package tracking",
                  "delivery update", "shipment status", "tracking update",
                  "delivery confirmation", "shipment notification"],
        fields=DELIVERY_NOTICE_FIELDS,
        description="Delivery notices, shipping notifications, package tracking"
    ),
    "interview_scheduling": DomainConcept(
        name="interview_scheduling",
        keywords=["interview scheduling", "interview invitation",
                  "interview request", "interview confirmation", "interview",
                  "job interview", "interview schedule", "interview appointment"],
        fields=INTERVIEW_SCHEDULING_FIELDS,
        description="Interview scheduling, invitations, confirmations"
    ),
    "newsletter": DomainConcept(
        name="newsletter",
        keywords=["newsletter", "email newsletter", "announcement",
                  "weekly digest", "monthly digest", "bulletin", "digest",
                  "newsletter issue", "mailing list"],
        fields=NEWSLETTER_FIELDS,
        description="Newsletters, digests, announcements, bulletins"
    ),
}


# ── Semantic Field Type Extraction Patterns ────────────────────────────────────

# These patterns are used by the regex extractor based on field_type
FIELD_TYPE_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"(?<![\w])(?:1[-\s.]?)?\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}(?![\w])",
    "date": r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}",
    "time": r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b",
    "datetime": r"\d{4}[-/]\d{1,2}[-/]\d{1,2}[T\s]\d{1,2}:\d{2}",
    "money": r"[\$\£\€]\s?\d[\d,]*\.?\d*(?:\s*[kKmMbBtT])?|\d[\d,]*\.?\d*\s*(?:USD|EUR|GBP|INR|CAD|AUD)",
    "percentage": r"\d+(?:\.\d+)?\s*%",
    "ip": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "domain": r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b",
    "url": r"https?://[^\s,)>]+\.(?:com|org|net|io|dev|co|me|in)(?:[^\s,)<>]*)?",
    "identifier": r"\b[A-Z0-9][A-Z0-9\-_]{3,}\b",
    "integer": r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",
    "person": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b",
    "organization": r"\b[A-Z][A-Za-z0-9&.'-]+(?:\s+[A-Z][A-Za-z0-9&.'-]+){0,3}\b",
    "address": r"\d+\s+[A-Z][A-Za-z\s]+(?:,\s*[A-Z][A-Za-z\s]+){1,3}",
    "role": r"\b(?:Senior|Junior|Lead|Principal|Staff|Architect|Manager|Director|VP|Head|CTO|CFO|CEO|Intern|Trainee|Associate|Entry|Mid)\s+[A-Z][A-Za-z\s\-]{2,}",
    "experience": r"\b\d+(?:\.\d+)?\s*(?:years?|yrs?)\b",
    "notice_period": r"\b\d+\s*(?:weeks?|days?|months?)\b",
    "seniority": r"\b(?:intern|trainee|junior|associate|entry.level|mid|mid.level|senior|lead|principal|staff|architect|manager|director|vp|head.of|cto|cfo|ceo)\b",
    "work_type": r"\b(?:fully.remote|work.from.home|wfh|remote|hybrid|flexible|onsite|on.site|in.office|office)\b",
    "education": r"\b(?:B\.S\.?|B\.A\.?|M\.S\.?|M\.A\.?|M\.B\.A\.?|Ph\.D\.?|M\.Tech|B\.Tech|A\.B\.?|Bachelor|Master|Doctorate|Diploma|Associate)\b",
    "skills": r"\b(?:Python|Java|JavaScript|TypeScript|React|Node|SQL|AWS|Docker|Kubernetes|Git|Linux|Agile|Scrum|REST|GraphQL|ML|AI|NLP|Data|Analytics)\b",
    "currency": r"\b(?:USD|EUR|GBP|INR|CAD|AUD|JPY|CNY|CHF)\b",
    "enum": r"\b(?:pending|confirmed|approved|rejected|completed|cancelled|in.progress|open|closed|resolved|shipped|delivered|processing)\b",
    "boolean": r"\b(?:yes|no|true|false|y|n)\b",
}


# ── Topic to Concept Mapping ──────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> set[str]:
    return {t for t in text.split() if t}


def resolve_topic_to_concept(topic: str) -> str | None:
    """Resolve a topic string to a domain concept name."""
    if not topic:
        return None
    
    norm = _normalize(topic)
    topic_tokens = _tokens(norm)
    
    # 1. Exact keyword match
    for concept_name, concept in DOMAIN_CONCEPTS.items():
        for kw in concept.keywords:
            if _normalize(kw) == norm:
                return concept_name
    
    # 2. Token-set containment (fuzzy matching)
    best: tuple[int, str] | None = None
    for concept_name, concept in DOMAIN_CONCEPTS.items():
        for kw in concept.keywords:
            kw_norm = _normalize(kw)
            kw_tokens = _tokens(kw_norm)
            
            if len(kw_tokens) < 2:
                continue
                
            # Alias tokens subset of topic tokens
            if kw_tokens <= topic_tokens:
                score = len(kw_tokens)
                if best is None or score > best[0]:
                    best = (score, concept_name)
            
            # Topic tokens subset of alias tokens
            if topic_tokens and topic_tokens <= kw_tokens:
                score = len(topic_tokens)
                if best is None or score > best[0]:
                    best = (score, concept_name)
    
    if best:
        return best[1]
    
    return None


def get_fields_for_concept(concept_name: str) -> list[FieldSpec]:
    """Get field specifications for a domain concept."""
    concept = DOMAIN_CONCEPTS.get(concept_name)
    if concept:
        return concept.fields
    return []


def get_all_field_specs(topic: str) -> list[FieldSpec]:
    """Get all field specs for a topic (universal + domain-specific)."""
    concept_name = resolve_topic_to_concept(topic)
    fields = list(UNIVERSAL_FIELDS)
    
    if concept_name:
        fields.extend(get_fields_for_concept(concept_name))
    
    # Sort by priority
    fields.sort(key=lambda f: f.priority)
    return fields


def get_field_names_for_topic(topic: str) -> list[str]:
    """Get just the field names for a topic (for backward compatibility)."""
    specs = get_all_field_specs(topic)
    seen = set()
    names = []
    for spec in specs:
        if spec.name not in seen:
            seen.add(spec.name)
            names.append(spec.name)
    return names


# ── Dynamic Field Generation from Topic (LLM-like) ────────────────────────────

# Semantic concept to field mappings for arbitrary topics
SEMANTIC_CONCEPT_FIELDS = {
    # Core identity concepts
    "application": ["applicant_name", "applicant_email", "applicant_phone", "applied_role", "application_date", "status"],
    "report": ["report_title", "report_date", "reporting_period", "author", "summary", "findings", "recommendations", "metrics"],
    "audit": ["audit_id", "audit_date", "auditor", "auditee", "scope", "findings", "rating", "recommendations", "next_audit_date"],
    "review": ["review_period", "reviewee", "reviewer", "rating", "strengths", "weaknesses", "goals", "feedback"],
    "evaluation": ["evaluation_date", "evaluator", "evaluatee", "criteria", "scores", "overall_rating", "comments"],
    "assessment": ["assessment_date", "assessor", "subject", "criteria", "results", "score", "recommendations"],
    "inspection": ["inspection_date", "inspector", "auditor", "location", "scope", "findings", "violations", "rating", "corrective_actions"],
    "vendor_inspection": ["inspection_id", "inspection_date", "auditor", "vendor", "facility", "location", "scope", "findings", "violations", "rating", "status", "corrective_actions", "report_number"],
    "survey": ["survey_title", "respondent", "date", "questions", "responses", "completion_rate", "nps"],
    "feedback": ["feedback_date", "from", "to", "topic", "rating", "comments", "suggestions"],
    "complaint": ["complaint_id", "date", "complainant", "subject", "description", "severity", "status", "resolution"],
    "reservation": ["reservation_id", "reserver_name", "reserver_email", "reserver_phone", "restaurant", "date", "time", "party_size", "table_number", "location", "confirmation_code", "status", "special_requests"],
    "booking": ["booking_id", "booker_name", "booker_email", "resource", "date", "time", "duration", "status", "confirmation", "departure", "destination"],
    "incident": ["incident_id", "date", "time", "location", "reporter", "description", "severity", "impact", "root_cause", "resolution", "status"],
    "request": ["request_id", "date", "requester", "type", "description", "priority", "status", "assignee", "due_date"],
    "proposal": ["proposal_id", "date", "proposer", "client", "title", "scope", "budget", "timeline", "deliverables", "terms"],
    "quote": ["quote_number", "date", "vendor", "client", "items", "prices", "total", "validity", "terms"],
    "estimate": ["estimate_number", "date", "estimator", "client", "scope", "breakdown", "total", "assumptions", "validity"],
    "order": ["order_number", "date", "customer", "items", "quantities", "prices", "total", "status", "shipping_address"],
    "purchase": ["purchase_order", "date", "buyer", "supplier", "items", "total", "delivery_date", "terms"],
    "payment": ["payment_id", "date", "payer", "payee", "amount", "currency", "method", "reference", "status"],
    "receipt": ["receipt_number", "date", "merchant", "customer", "items", "total", "payment_method", "tax"],
    "statement": ["statement_period", "account", "balance", "transactions", "opening_balance", "closing_balance"],
    "reimbursement": ["request_id", "date", "employee", "category", "amount", "currency", "receipts", "status", "approver"],
    "expense": ["expense_id", "date", "employee", "category", "amount", "currency", "description", "receipt", "status"],
    "invoice": ["invoice_number", "date", "due_date", "vendor", "customer", "items", "subtotal", "tax", "total", "payment_terms"],
    "contract": ["contract_id", "date", "parties", "effective_date", "expiration_date", "value", "terms", "signatures"],
    "agreement": ["agreement_id", "date", "parties", "type", "terms", "effective_date", "expiration_date", "signatures"],
    "nda": ["nda_id", "date", "parties", "purpose", "duration", "confidential_info", "exceptions", "signatures"],
    "offer": ["offer_id", "date", "candidate", "role", "salary", "benefits", "start_date", "conditions", "expiry"],
    "resignation": ["employee", "date", "last_day", "reason", "notice_period", "handover", "exit_interview"],
    "termination": ["employee", "date", "reason", "effective_date", "severance", "notice", "final_pay"],
    "promotion": ["employee", "date", "old_role", "new_role", "old_salary", "new_salary", "effective_date", "reason"],
    "performance": ["employee", "period", "reviewer", "rating", "achievements", "goals", "development_plan", "compensation_change"],
    "appraisal": ["employee", "period", "appraiser", "rating", "strengths", "areas_for_improvement", "goals", "salary_review"],
    "onboarding": ["employee", "start_date", "role", "manager", "buddy", "equipment", "access", "training", "orientation"],
    "offboarding": ["employee", "last_day", "reason", "handover", "equipment_return", "access_revocation", "exit_interview", "final_pay"],
    "meeting": ["meeting_title", "date", "time", "duration", "organizer", "attendees", "location", "agenda", "action_items", "decisions"],
    "interview": ["candidate", "interviewer", "date", "time", "role", "round", "type", "location", "feedback", "decision"],
    "screening": ["candidate", "screener", "date", "role", "result", "notes", "next_steps"],
    "reference_check": ["candidate", "reference", "date", "relationship", "duration", "rating", "strengths", "weaknesses", "rehire"],
    "background_check": ["candidate", "date", "provider", "checks", "results", "status", "adverse_action"],
    "onboarding_doc": ["employee", "document", "date", "status", "signed_date", "expiry"],
    "policy": ["policy_name", "version", "effective_date", "owner", "scope", "summary", "compliance", "review_date"],
    "announcement": ["title", "date", "author", "audience", "summary", "details", "action_required", "deadline"],
    "newsletter": ["title", "date", "issue", "author", "sections", "highlights", "unsubscribe"],
    "notification": ["title", "date", "recipient", "type", "priority", "message", "action_required", "link"],
    "alert": ["alert_id", "timestamp", "severity", "source", "message", "affected_systems", "acknowledged", "resolved"],
    "log": ["timestamp", "level", "source", "message", "trace_id", "user_id", "session_id"],
    "metric": ["metric_name", "timestamp", "value", "unit", "tags", "threshold", "status"],
    "kpi": ["kpi_name", "period", "target", "actual", "variance", "trend", "owner", "status"],
    "dashboard": ["dashboard_name", "date", "metrics", "insights", "actions", "owner"],
    "forecast": ["forecast_period", "metric", "predicted", "actual", "confidence", "methodology", "assumptions"],
    "budget": ["budget_period", "department", "allocated", "spent", "remaining", "variance", "owner", "approvals"],
    "financial": ["period", "revenue", "expenses", "profit", "margin", "cash_flow", "assets", "liabilities", "equity"],
    "tax": ["tax_period", "entity", "jurisdiction", "tax_type", "taxable_income", "tax_due", "payments", "refund", "filing_status"],
    "compliance": ["regulation", "period", "entity", "requirements", "status", "gaps", "evidence", "next_review"],
    "audit_trail": ["timestamp", "user", "action", "resource", "old_value", "new_value", "ip_address", "session"],
    "security": ["event_id", "timestamp", "type", "severity", "source", "target", "action_taken", "status", "investigation"],
    "vulnerability": ["vuln_id", "date", "cve", "severity", "affected_systems", "description", "exploit", "patch", "status"],
    "patch": ["patch_id", "date", "systems", "vulnerabilities", "status", "tested", "deployed", "issues"],
    "deployment": ["deployment_id", "date", "environment", "version", "services", "status", "rollback", "duration"],
    "release": ["release_version", "date", "features", "fixes", "breaking_changes", "migration", "status", "notes"],
    "changelog": ["version", "date", "added", "changed", "deprecated", "removed", "fixed", "security"],
    "documentation": ["doc_title", "version", "author", "date", "section", "content", "status", "reviewers"],
    "specification": ["spec_id", "version", "author", "date", "requirements", "design", "api", "data_model", "acceptance_criteria"],
    "requirements": ["req_id", "title", "description", "priority", "status", "assignee", "dependencies", "acceptance_criteria"],
    "test_plan": ["plan_id", "version", "scope", "features", "test_cases", "environment", "schedule", "resources", "risks"],
    "test_case": ["case_id", "title", "preconditions", "steps", "expected", "actual", "status", "defects"],
    "defect": ["defect_id", "title", "description", "severity", "priority", "status", "assignee", "reporter", "found_in", "fixed_in"],
    "bug": ["bug_id", "title", "description", "severity", "priority", "status", "assignee", "reporter", "steps_to_reproduce", "environment"],
    "feature": ["feature_id", "title", "description", "priority", "status", "assignee", "epic", "story_points", "acceptance_criteria"],
    "user_story": ["story_id", "title", "as_a", "i_want", "so_that", "acceptance_criteria", "story_points", "priority", "sprint"],
    "epic": ["epic_id", "title", "description", "owner", "status", "start_date", "end_date", "features"],
    "sprint": ["sprint_id", "name", "start_date", "end_date", "goal", "capacity", "committed", "completed", "velocity"],
    "retrospective": ["sprint", "date", "participants", "what_went_well", "what_didnt", "action_items", "owner"],
    "standup": ["date", "team", "participants", "yesterday", "today", "blockers"],
    "planning": ["sprint", "date", "participants", "stories_committed", "story_points", "capacity", "risks"],
    "review": ["sprint", "date", "participants", "demoed", "accepted", "rejected", "feedback"],
    "daily": ["date", "team", "completed", "in_progress", "planned", "blockers", "metrics"],
}


def generate_fields_from_topic(topic: str, max_fields: int = 50) -> list[str]:
    """LLM-like field generation from any topic string.
    
    Analyzes the topic semantically and generates comprehensive field names
    based on domain concepts, similar to how an LLM would understand the domain.
    """
    if not topic:
        return [f.name for f in UNIVERSAL_FIELDS[:15]]
    
    norm = _normalize(topic)
    tokens = _tokens(norm)
    
    # 1. Check for known domain concepts
    concept_name = resolve_topic_to_concept(topic)
    if concept_name:
        fields = get_field_names_for_topic(topic)
        return fields[:max_fields]
    
    # 2. Semantic concept matching - find matching concepts from topic tokens
    matched_fields: list[str] = []
    matched_concepts: set[str] = set()
    
    for concept_key, concept_fields in SEMANTIC_CONCEPT_FIELDS.items():
        concept_tokens = _tokens(concept_key)
        # Check if concept is mentioned in topic
        if concept_tokens <= tokens or any(ct in norm for ct in concept_tokens if len(ct) > 3):
            if concept_key not in matched_concepts:
                matched_concepts.add(concept_key)
                for f in concept_fields:
                    if f not in matched_fields:
                        matched_fields.append(f)
    
    # 3. Token-based field generation - generate fields from topic tokens
    token_fields = _generate_fields_from_tokens(tokens)
    for f in token_fields:
        if f not in matched_fields:
            matched_fields.append(f)
    
    # 4. Add universal fields
    for f in [f.name for f in UNIVERSAL_FIELDS]:
        if f not in matched_fields:
            matched_fields.append(f)
    
    return matched_fields[:max_fields]


def _generate_fields_from_tokens(tokens: set[str]) -> list[str]:
    """Generate field names from topic tokens using semantic rules."""
    fields = []
    
    # Entity-type mappings
    token_to_fields = {
        # People/roles
        "candidate": ["candidate_name", "candidate_email", "candidate_phone", "candidate_skills", "candidate_experience"],
        "applicant": ["applicant_name", "applicant_email", "applicant_phone", "applied_position", "resume"],
        "employee": ["employee_name", "employee_id", "employee_email", "department", "role", "start_date", "manager"],
        "customer": ["customer_name", "customer_email", "customer_phone", "customer_address", "account_number"],
        "client": ["client_name", "client_email", "client_company", "contact_person", "account_manager"],
        "vendor": ["vendor_name", "vendor_email", "vendor_address", "vendor_contact", "payment_terms"],
        "inspection": ["inspection_id", "inspection_date", "auditor", "inspector", "vendor", "facility", "location", "scope", "findings", "violations", "rating", "status", "corrective_actions", "report_number"],
        "supplier": ["supplier_name", "supplier_email", "supplier_address", "contact_person", "payment_terms"],
        "buyer": ["buyer_name", "buyer_email", "buyer_company", "purchase_order", "billing_address"],
        "seller": ["seller_name", "seller_email", "seller_company", "items", "shipping_terms"],
        "user": ["user_name", "user_email", "user_id", "role", "permissions", "last_login"],
        "admin": ["admin_name", "admin_email", "admin_role", "permissions", "actions"],
        "manager": ["manager_name", "manager_email", "team", "department", "reports"],
        "director": ["director_name", "director_email", "department", "division", "reports"],
        "lead": ["lead_name", "lead_email", "lead_source", "lead_status", "lead_score", "assigned_to"],
        "prospect": ["prospect_name", "prospect_email", "prospect_company", "interest_level", "next_steps"],
        "partner": ["partner_name", "partner_company", "partner_type", "agreement", "contact_person"],
        "contractor": ["contractor_name", "contractor_email", "contractor_company", "contract", "rate", "duration"],
        "consultant": ["consultant_name", "consultant_email", "consultant_firm", "engagement", "rate", "deliverables"],
        "auditor": ["auditor_name", "auditor_firm", "audit_standard", "scope", "findings", "opinion"],
        "inspector": ["inspector_name", "inspector_agency", "inspection_type", "location", "results", "violations"],
        "reviewer": ["reviewer_name", "reviewer_role", "reviewee", "period", "rating", "feedback"],
        "evaluator": ["evaluator_name", "evaluator_role", "criteria", "scores", "recommendation"],
        "assessor": ["assessor_name", "assessor_qualification", "subject", "criteria", "results"],
        "interviewer": ["interviewer_name", "interviewer_role", "candidate", "date", "feedback", "decision"],
        "candidate": ["candidate_name", "candidate_email", "candidate_phone", "role", "experience", "skills"],
        "recruiter": ["recruiter_name", "recruiter_email", "agency", "role", "pipeline", "status"],
        "hr": ["hr_name", "hr_email", "department", "employee_relations", "policies"],
        "finance": ["finance_contact", "finance_email", "department", "budget", "approvals", "payments"],
        "legal": ["legal_contact", "legal_email", "department", "contracts", "compliance", "litigation"],
        "it": ["it_contact", "it_email", "systems", "tickets", "projects", "infrastructure"],
        "support": ["support_tier", "support_channel", "sla", "response_time", "resolution_time"],
        "sales": ["sales_rep", "sales_email", "territory", "quota", "pipeline", "forecast"],
        "marketing": ["campaign", "channel", "audience", "budget", "roi", "leads", "conversions"],
        "product": ["product_name", "product_version", "features", "roadmap", "release_date", "owner"],
        "project": ["project_name", "project_id", "manager", "start_date", "end_date", "status", "budget", "team"],
        "task": ["task_id", "title", "assignee", "status", "priority", "due_date", "estimate", "actual"],
        "issue": ["issue_id", "title", "description", "severity", "priority", "status", "assignee", "reporter", "root_cause"],
        "risk": ["risk_id", "description", "likelihood", "impact", "mitigation", "owner", "status", "residual"],
        "opportunity": ["opportunity_id", "name", "account", "stage", "amount", "probability", "close_date", "owner"],
        "deal": ["deal_id", "name", "account", "stage", "amount", "probability", "close_date", "owner", "terms"],
        "account": ["account_name", "account_id", "owner", "industry", "size", "tier", "contacts", "revenue"],
        "contact": ["contact_name", "contact_email", "contact_phone", "account", "role", "owner", "status"],
        "lead": ["lead_name", "lead_email", "lead_company", "source", "status", "score", "assigned_to", "next_steps"],
    }
    
    for token in tokens:
        if token in token_to_fields:
            for f in token_to_fields[token]:
                if f not in fields:
                    fields.append(f)
    
    # Action/process tokens
    action_to_fields = {
        "apply": ["application_id", "applicant", "position", "date", "status", "documents", "referral"],
        "register": ["registration_id", "registrant", "event", "date", "status", "payment", "confirmation"],
        "subscribe": ["subscription_id", "subscriber", "plan", "start_date", "end_date", "status", "payment_method"],
        "unsubscribe": ["subscriber", "plan", "date", "reason", "confirmation"],
        "purchase": ["purchase_id", "buyer", "items", "total", "date", "payment", "shipping", "status"],
        "order": ["order_id", "customer", "items", "quantities", "prices", "total", "date", "status", "shipping"],
        "cancel": ["cancellation_id", "original_order", "reason", "date", "refund", "status", "confirmation"],
        "refund": ["refund_id", "original_payment", "amount", "reason", "date", "status", "method"],
        "return": ["return_id", "original_order", "items", "reason", "date", "status", "refund", "resolution"],
        "exchange": ["exchange_id", "original_order", "old_items", "new_items", "date", "status", "difference"],
        "upgrade": ["upgrade_id", "from_plan", "to_plan", "date", "price_difference", "effective_date", "confirmation"],
        "downgrade": ["downgrade_id", "from_plan", "to_plan", "date", "price_difference", "effective_date", "confirmation"],
        "renew": ["renewal_id", "subscription", "old_term", "new_term", "price", "date", "status", "auto_renew"],
        "extend": ["extension_id", "original_end", "new_end", "reason", "date", "approval", "terms"],
        "terminate": ["termination_id", "contract", "reason", "effective_date", "notice_period", "settlement", "confirmation"],
        "expire": ["expiration_id", "item", "expiry_date", "grace_period", "renewal_options", "notification"],
        "approve": ["approval_id", "request", "approver", "date", "decision", "comments", "conditions"],
        "reject": ["rejection_id", "request", "rejector", "date", "reason", "appeal_process"],
        "request": ["request_id", "requester", "type", "description", "priority", "date", "status", "assignee"],
        "submit": ["submission_id", "submitter", "type", "content", "date", "status", "reviewer"],
        "review": ["review_id", "reviewer", "subject", "date", "rating", "comments", "decision", "next_steps"],
        "evaluate": ["evaluation_id", "evaluator", "subject", "criteria", "date", "scores", "recommendation"],
        "assess": ["assessment_id", "assessor", "subject", "criteria", "date", "results", "score", "recommendations"],
        "inspect": ["inspection_id", "inspector", "location", "scope", "date", "findings", "rating", "violations", "corrective_actions"],
        "audit": ["audit_id", "auditor", "auditee", "scope", "standard", "date", "findings", "opinion", "recommendations"],
        "survey": ["survey_id", "respondent", "survey", "date", "responses", "completion", "nps", "feedback"],
        "feedback": ["feedback_id", "from", "to", "topic", "date", "rating", "comments", "suggestions", "action_items"],
        "complain": ["complaint_id", "complainant", "subject", "description", "date", "severity", "status", "resolution", "root_cause"],
        "report": ["report_id", "reporter", "type", "subject", "date", "summary", "findings", "recommendations", "attachments"],
        "incident": ["incident_id", "reporter", "date", "time", "location", "description", "severity", "impact", "root_cause", "resolution", "status"],
        "alert": ["alert_id", "timestamp", "severity", "source", "message", "affected", "acknowledged", "resolved", "owner"],
        "notify": ["notification_id", "recipient", "type", "priority", "message", "date", "read", "action_required", "link"],
        "announce": ["announcement_id", "author", "audience", "title", "date", "summary", "details", "action_required", "deadline"],
        "publish": ["publication_id", "author", "title", "date", "content", "channel", "audience", "metrics"],
        "schedule": ["schedule_id", "event", "date", "time", "duration", "organizer", "attendees", "location", "status"],
        "book": ["booking_id", "booker", "resource", "date", "time", "duration", "status", "confirmation", "cancellation_policy"],
        "reserve": ["reservation_id", "reserver", "resource", "date", "time", "duration", "status", "confirmation", "deposit"],
        "confirm": ["confirmation_id", "reference", "details", "date", "confirmed_by", "status", "next_steps"],
        "invite": ["invitation_id", "inviter", "invitees", "event", "date", "time", "location", "rsvp", "status"],
        "attend": ["attendance_id", "attendee", "event", "date", "check_in", "check_out", "status", "feedback"],
        "participate": ["participant", "activity", "date", "role", "contribution", "hours", "feedback"],
        "complete": ["completion_id", "item", "completer", "date", "status", "quality", "certificate", "next_steps"],
        "finish": ["finish_id", "item", "finisher", "date", "status", "deliverables", "acceptance", "handover"],
        "deliver": ["delivery_id", "item", "recipient", "date", "method", "tracking", "status", "proof", "confirmation"],
        "ship": ["shipment_id", "shipper", "recipient", "items", "date", "carrier", "tracking", "estimated_delivery", "status"],
        "receive": ["receipt_id", "receiver", "items", "date", "condition", "quantity", "discrepancies", "confirmation"],
        "accept": ["acceptance_id", "item", "accepter", "date", "criteria", "status", "comments", "sign_off"],
        "sign": ["signature_id", "document", "signer", "date", "role", "witness", "status", "verification"],
        "execute": ["execution_id", "contract", "parties", "date", "effective_date", "witnesses", "notary", "registration"],
        "file": ["filing_id", "document", "filer", "date", "authority", "reference", "status", "confirmation"],
        "record": ["record_id", "event", "recorder", "date", "details", "witnesses", "evidence", "classification"],
        "document": ["document_id", "title", "author", "date", "version", "status", "reviewers", "approvers", "distribution"],
        "archive": ["archive_id", "item", "archiver", "date", "retention", "location", "access_policy", "reason"],
        "delete": ["deletion_id", "item", "deleter", "date", "reason", "approval", "backup", "confirmation"],
        "remove": ["removal_id", "item", "remover", "date", "reason", "approval", "replacement", "confirmation"],
        "add": ["addition_id", "item", "adder", "date", "details", "reason", "approval", "effective_date"],
        "create": ["creation_id", "item", "creator", "date", "type", "details", "status", "owner"],
        "generate": ["generation_id", "item", "generator", "date", "parameters", "output", "format", "status"],
        "produce": ["production_id", "item", "producer", "date", "quantity", "specifications", "quality", "batch"],
        "manufacture": ["manufacturing_id", "product", "manufacturer", "date", "quantity", "specifications", "quality", "batch", "lot"],
        "assemble": ["assembly_id", "product", "assembler", "date", "components", "specifications", "quality", "serial_numbers"],
        "install": ["installation_id", "item", "installer", "date", "location", "configuration", "testing", "status", "documentation"],
        "configure": ["configuration_id", "item", "configurer", "date", "settings", "parameters", "version", "status", "backup"],
        "setup": ["setup_id", "system", "setter", "date", "configuration", "environment", "credentials", "status", "documentation"],
        "deploy": ["deployment_id", "version", "environment", "date", "services", "strategy", "status", "rollback_plan", "duration"],
        "release": ["release_id", "version", "date", "features", "fixes", "breaking_changes", "migration", "status", "notes", "artifacts"],
        "publish": ["publication_id", "artifact", "publisher", "date", "registry", "version", "checksum", "metadata", "status"],
        "distribute": ["distribution_id", "artifact", "distributor", "date", "channels", "regions", "status", "metrics", "feedback"],
        "monitor": ["monitoring_id", "system", "monitor", "date", "metrics", "alerts", "thresholds", "dashboards", "on_call"],
        "observe": ["observation_id", "system", "observer", "date", "metrics", "logs", "traces", "anomalies", "insights"],
        "analyze": ["analysis_id", "subject", "analyst", "date", "method", "findings", "conclusions", "recommendations", "artifacts"],
        "investigate": ["investigation_id", "subject", "investigator", "date", "scope", "evidence", "findings", "root_cause", "recommendations", "status"],
        "diagnose": ["diagnosis_id", "subject", "diagnostician", "date", "symptoms", "tests", "results", "diagnosis", "treatment_plan", "prognosis"],
        "troubleshoot": ["troubleshooting_id", "issue", "engineer", "date", "symptoms", "steps", "root_cause", "resolution", "prevention", "duration"],
        "debug": ["debug_session_id", "issue", "engineer", "date", "environment", "steps", "breakpoints", "variables", "root_cause", "fix"],
        "fix": ["fix_id", "issue", "engineer", "date", "root_cause", "solution", "testing", "deployment", "verification", "regression_test"],
        "patch": ["patch_id", "vulnerability", "engineer", "date", "affected_versions", "fixed_versions", "testing", "deployment", "cve"],
        "update": ["update_id", "component", "engineer", "date", "from_version", "to_version", "changes", "testing", "deployment", "rollback"],
        "upgrade": ["upgrade_id", "system", "engineer", "date", "from_version", "to_version", "changes", "testing", "deployment", "downtime", "rollback"],
        "migrate": ["migration_id", "system", "engineer", "date", "from", "to", "strategy", "data_volume", "downtime", "validation", "rollback"],
        "backup": ["backup_id", "system", "operator", "date", "type", "size", "location", "verification", "retention", "encryption"],
        "restore": ["restore_id", "backup", "operator", "date", "target", "point_in_time", "validation", "duration", "data_loss", "status"],
        "recover": ["recovery_id", "system", "engineer", "date", "rto", "rpo", "steps", "validation", "duration", "data_loss", "lessons_learned"],
    }
    
    for token in tokens:
        if token in action_to_fields:
            for f in action_to_fields[token]:
                if f not in fields:
                    fields.append(f)
    
    return fields


# ── Field Spec Lookup ─────────────────────────────────────────────────────────

FIELD_SPECS_BY_NAME: dict[str, FieldSpec] = {}

def _build_field_spec_index():
    global FIELD_SPECS_BY_NAME
    for f in UNIVERSAL_FIELDS:
        FIELD_SPECS_BY_NAME[f.name] = f
    for concept in DOMAIN_CONCEPTS.values():
        for f in concept.fields:
            FIELD_SPECS_BY_NAME[f.name] = f

_build_field_spec_index()


def get_field_spec(field_name: str) -> FieldSpec | None:
    """Get the FieldSpec for a field name."""
    return FIELD_SPECS_BY_NAME.get(field_name)


def get_field_type(field_name: str) -> str:
    """Get the field type for a field name."""
    spec = get_field_spec(field_name)
    if spec:
        return spec.field_type
    # Infer from name
    name_lower = field_name.lower()
    if "email" in name_lower:
        return "email"
    if "phone" in name_lower or "mobile" in name_lower:
        return "phone"
    if "date" in name_lower:
        return "date"
    if "time" in name_lower:
        return "time"
    if "amount" in name_lower or "total" in name_lower or "price" in name_lower or "cost" in name_lower or "salary" in name_lower:
        return "money"
    if "ip" in name_lower:
        return "ip"
    if "url" in name_lower or "link" in name_lower:
        return "url"
    if "count" in name_lower or "number" in name_lower or "quantity" in name_lower:
        return "integer"
    if "name" in name_lower:
        return "person"
    if "company" in name_lower or "organization" in name_lower or "org" in name_lower or "employer" in name_lower:
        return "organization"
    if "address" in name_lower or "location" in name_lower:
        return "address"
    if "status" in name_lower or "state" in name_lower:
        return "enum"
    if "priority" in name_lower or "severity" in name_lower:
        return "enum"
    # Experience / years
    if "experience" in name_lower or "years" in name_lower or "year" in name_lower or "exp" in name_lower:
        return "experience"
    # Role / position / title
    if "role" in name_lower or "position" in name_lower or "title" in name_lower:
        return "role"
    if "skill" in name_lower:
        return "skills"
    if "education" in name_lower or "degree" in name_lower or "qualification" in name_lower:
        return "education"
    if "senior" in name_lower or "junior" in name_lower or "lead" in name_lower or "level" in name_lower:
        return "seniority"
    if "notice" in name_lower:
        return "notice_period"
    if "work_type" in name_lower or "employment_type" in name_lower or "job_type" in name_lower:
        return "work_type"
    if "salary" in name_lower or "compensation" in name_lower or "pay" in name_lower:
        return "money"
    return "text"


def get_kv_keys_for_field(field_name: str) -> list[str]:
    """Get KV keys for a field name."""
    spec = get_field_spec(field_name)
    if spec:
        return spec.kv_keys
    # Generate from field name
    label = field_name.replace("_", " ")
    return [label, label.title()]


def get_regex_pattern_for_field(field_name: str) -> str | None:
    """Get regex pattern for a field name based on its type."""
    field_type = get_field_type(field_name)
    return FIELD_TYPE_PATTERNS.get(field_type)


def get_gliner_label(field_name: str) -> str:
    """Get GLiNER label for a field name."""
    spec = get_field_spec(field_name)
    if spec and spec.gliner_label:
        return spec.gliner_label
    return field_name.replace("_", " ")