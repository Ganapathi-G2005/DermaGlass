"""Quick check that OPENAI_API_KEY works and gpt-5-nano is available."""

import os
from pathlib import Path

from dotenv import load_dotenv

_backend_env = Path(__file__).resolve().parent.parent / "App" / "backend" / ".env"
load_dotenv(_backend_env)
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY not found in .env")
else:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    print("Checking model gpt-5-nano is retrievable...")
    try:
        m = client.models.retrieve("gpt-5-nano")
        print(f"OK: {m.id}")
    except Exception as e:
        print(f"Error: {e}")
