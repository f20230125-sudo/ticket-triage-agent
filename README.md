# Support Ticket Triage Agent

A CLI tool that classifies a raw customer support message using Gemini and
returns **validated structured output** — category, urgency, an internal
summary, and a drafted reply — instead of free-form text you'd have to
parse yourself.

## Why structured output

Production systems that route or act on an LLM's response can't work with
"the model probably returned something JSON-shaped." This uses Gemini's
native structured output mode with a **Pydantic schema**
(`response_mime_type="application/json"`, `response_schema=TicketTriage`),
so every response is guaranteed to match the shape the code expects —
an enum-constrained category and urgency, always present.

## How it works

```
customer message -> Gemini (structured output mode + Pydantic schema)
                  -> validated TicketTriage object
                        - category: billing | technical | account | feedback | other
                        - urgency: low | medium | high
                        - summary: one-line internal note
                        - suggested_reply: draft customer-facing reply
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then paste in your Gemini API key
```

## Usage

Triage a single message:
```bash
python triage.py "My withdrawal has been pending for 3 days, please help!"
```

Or run with no arguments to triage 5 bundled sample tickets:
```bash
python triage.py
```

## Tests

```bash
pip install pytest
pytest
```

The tests swap Gemini for a stub client, so they run without an API key
and make no network calls. They cover the request sent to the model, the
enum-constrained schema, the retry on a temporary 503, and the CLI.

## Stack

Python, Google Gemini API (`google-genai`), Pydantic (schema-validated
structured output).
