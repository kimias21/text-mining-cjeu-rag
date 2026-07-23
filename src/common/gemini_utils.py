"""
Shared helpers for the Gemini-based agents:
  - to_gemini_tool: convert OpenAI-style tool schemas (nested
    {"type": "function", "function": {...}}, as defined in
    single_agent/tools.py) into a Gemini `types.Tool`, so schema
    definitions aren't duplicated between the single-agent (Task A) and
    multi-agent (Task B) systems.
  - generate_with_retry: wraps client.models.generate_content with
    retry-on-rate-limit, since Gemini's free tier is capped at 15
    requests/minute per model, and one question can trigger several LLM
    calls (search, more searches, final answer) -- easy to exceed in a
    batch run of 20 questions. Parses the API's suggested retry delay when
    present, falls back to exponential backoff otherwise.
"""
import re
import time

from google.genai import types
from google.genai import errors as genai_errors


def to_gemini_tool(openai_style_schemas):
    declarations = []
    for schema in openai_style_schemas:
        fn = schema["function"]
        declarations.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn["description"],
            parameters=fn["parameters"],
        ))
    return types.Tool(function_declarations=declarations)


def generate_with_retry(client, max_retries=8, base_delay=10, **kwargs):
    """Drop-in replacement for client.models.generate_content(**kwargs)
    that retries on 429 RESOURCE_EXHAUSTED (rate limit) errors instead of
    crashing, waiting either the API's suggested retry delay or an
    exponentially increasing fallback."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.ClientError as e:
            msg = str(e)
            if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
                raise
            m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"'retryDelay':\s*'(\d+)s'", msg)
            delay = float(m.group(1)) + 2 if m else base_delay * (2 ** attempt)
            print(f"  [rate limited -- waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}]")
            time.sleep(delay)
    raise RuntimeError(f"Exceeded {max_retries} retries due to persistent rate limiting.")
