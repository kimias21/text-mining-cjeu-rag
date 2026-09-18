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
    that retries instead of crashing on:
      - 429 RESOURCE_EXHAUSTED (rate limit) -- a ClientError; waits either
        the API's suggested retry delay or an exponentially increasing
        fallback.
      - 503 UNAVAILABLE ("model is currently experiencing high demand") --
        a ServerError, NOT a ClientError, so it needs its own except
        clause. Bug fix: this used to only catch ClientError, so a
        transient 503 (common on the free tier, especially for
        gemini-3.1-flash-lite under load) crashed the whole batch/agent
        run outright instead of just costing a short wait -- found while
        running the KG ablation, where two separate 20-question batches
        both died on an ordinary mid-run 503. This matters beyond the
        ablation: the official hidden-query evaluation calls this same
        code path, so an unhandled 503 there would fail a live graded
        question, not just a local script.
    """
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.ServerError as e:
            msg = str(e)
            if "503" not in msg and "UNAVAILABLE" not in msg:
                raise
            delay = min(base_delay * (2 ** attempt), 60)
            print(f"  [model overloaded (503) -- waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}]")
            time.sleep(delay)
        except genai_errors.ClientError as e:
            msg = str(e)
            if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
                raise
            m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"'retryDelay':\s*'(\d+)s'", msg)
            delay = float(m.group(1)) + 2 if m else base_delay * (2 ** attempt)
            print(f"  [rate limited -- waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}]")
            time.sleep(delay)
    raise RuntimeError(f"Exceeded {max_retries} retries due to persistent rate limiting / server errors.")
