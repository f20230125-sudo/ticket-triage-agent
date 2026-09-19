"""
triage.py
---------
A support-ticket triage agent: given a raw customer message, it asks
Gemini to classify it and draft a reply, returned as VALIDATED structured
JSON (not free text you have to parse yourself).

Why this matters: production systems can't work with "the model probably
said something JSON-shaped." This uses Gemini's structured output mode
with a Pydantic schema, so the response is guaranteed to match the shape
we define -- the same pattern used for routing customer support tickets,
categorizing tickets in a helpdesk, or feeding downstream automation.

Usage:
    python triage.py "My withdrawal has been pending for 3 days, please help!"

Or run without an argument to triage the bundled sample tickets:
    python triage.py
"""

import os
import sys
import time
from enum import Enum

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

load_dotenv()

MODEL = "gemini-3.6-flash"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

SAMPLE_TICKETS = [
    "My withdrawal has been pending for 3 days, please help!",
    "How do I change the leverage on my trading account?",
    "The app crashes every time I try to open the charts page.",
    "I was charged twice for the same deposit, please refund the extra charge.",
    "Just wanted to say I love the new UI update, great work!",
]


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    FEEDBACK = "feedback"
    OTHER = "other"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TicketTriage(BaseModel):
    category: Category
    urgency: Urgency
    summary: str
    suggested_reply: str


def triage_ticket(client: genai.Client, message: str) -> TicketTriage:
    prompt = f"""You are a customer support triage assistant for a trading
platform. Read the customer message and classify it.

Customer message: "{message}"

Provide:
- category: the best-fitting category
- urgency: how urgently this needs a human response
- summary: a one-sentence internal summary for the support agent
- suggested_reply: a short, polite draft reply to send the customer
"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TicketTriage,
                ),
            )
            return response.parsed
        except genai_errors.ServerError as e:
            # Gemini's servers occasionally return a temporary 503 under
            # high demand -- this is not a bug in our request, so back off
            # and retry a few times before giving up.
            if attempt == MAX_RETRIES:
                raise
            print(f"  (model temporarily unavailable, retrying in {RETRY_DELAY_SECONDS}s... "
                  f"attempt {attempt}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY_SECONDS)


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    client = genai.Client(api_key=api_key)

    messages = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else SAMPLE_TICKETS

    for msg in messages:
        result = triage_ticket(client, msg)
        print("=" * 70)
        print(f"Ticket:   {msg}")
        print(f"Category: {result.category.value}")
        print(f"Urgency:  {result.urgency.value}")
        print(f"Summary:  {result.summary}")
        print(f"Reply:    {result.suggested_reply}")
    print("=" * 70)


if __name__ == "__main__":
    main()
