"""
Step 3 - Single-agent ReAct system (Task A) -- Google Gemini backend.

A single LLM-driven agent that can call the tools in tools.py in a loop:
  Thought (implicit in the model's reasoning) -> Action (tool call) ->
  Observation (tool result) -> ... -> Final Answer.

Uses Google's Gemini API (free tier: no credit card required, Flash models,
~1,500 requests/day -- see https://ai.google.dev), via the official
`google-genai` SDK, with automatic function calling DISABLED so we can drive
the loop ourselves and log every (thought/action/observation) step -- both
for the evaluation dashboard (Step 6) and for turning the loop into a
flowchart for the write-up.

Get a free API key (no credit card): https://aistudio.google.com/apikey

Usage:
    from single_agent.agent import SingleAgent
    agent = SingleAgent()
    result = agent.answer("Can mitigation measures be taken into account in a Habitats Directive assessment?")
    print(result["answer"])
    print(result["trace"])       # step-by-step Thought/Action/Observation log
    print(result["sources"])     # case numbers actually used as context
"""
import json
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from gemini_utils import to_gemini_tool, generate_with_retry  # noqa: E402

from tools import TOOLS, TOOL_SCHEMAS

SYSTEM_PROMPT = """You are a legal research assistant specialized in EU environmental \
and agricultural law, answering questions about CJEU (Court of Justice of the EU) \
preliminary rulings under Article 267 TFEU.

You have tools to search and retrieve judgments from a corpus of 279 such rulings. \
Always ground your answers in retrieved judgments -- never answer from memory alone. \
Use search_corpus to find relevant judgments, then get_case_by_number to pull the full \
text/operative part of the most relevant one(s) before answering. Use list_cases_by_filter \
for pure counting/browsing questions.

When you have enough evidence, give a final answer with NO further tool calls. Your final \
answer must:
- directly answer the question,
- cite the specific case number(s) (e.g. "C-116/20") your answer relies on,
- state briefly if the corpus does not contain a clear answer, rather than guessing.
"""

# Free-tier-eligible model. As of July 2026, Google's Flash-class models are
# free-tier eligible while Pro is not; gemini-2.5-flash was recently retired
# for new users, so gemini-3.1-flash-lite (GA, cost-effective) is used here.
# If this stops working by the time you run it (Google iterates on model
# names frequently), check https://ai.google.dev/gemini-api/docs/models for
# the current free-tier model list and set SINGLE_AGENT_MODEL accordingly.
MODEL_NAME = os.environ.get("SINGLE_AGENT_MODEL", "gemini-3.1-flash-lite")
MAX_STEPS = 6


class SingleAgent:
    def __init__(self, model_name=MODEL_NAME, client=None, api_key=None):
        self.model_name = model_name
        self.client = client or genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.tool = to_gemini_tool(TOOL_SCHEMAS)
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[self.tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def answer(self, question: str, max_steps: int = MAX_STEPS) -> dict:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        trace = []
        sources = set()

        for step in range(max_steps):
            response = generate_with_retry(
                self.client, model=self.model_name, contents=contents, config=self.config,
            )
            calls = response.function_calls or []

            if not calls:
                trace.append({"step": step, "type": "final_answer", "content": response.text})
                return {"answer": response.text, "trace": trace, "sources": sorted(sources)}

            contents.append(response.candidates[0].content)  # model's turn (the function call(s))

            response_parts = []
            for call in calls:
                fn_name = call.name
                args = dict(call.args or {})

                if fn_name not in TOOLS:
                    result = {"error": f"Unknown tool '{fn_name}'"}
                else:
                    result = TOOLS[fn_name](**args)

                for hit in (result if isinstance(result, list) else [result]):
                    if isinstance(hit, dict) and hit.get("case_number"):
                        sources.add(hit["case_number"])
                    elif isinstance(hit, dict) and hit.get("metadata", {}).get("case_number"):
                        sources.add(hit["metadata"]["case_number"])

                trace.append({
                    "step": step, "type": "action", "tool": fn_name, "args": args,
                    "observation_preview": json.dumps(result, ensure_ascii=False)[:500],
                })

                response_parts.append(types.Part.from_function_response(
                    name=fn_name, response={"result": result},
                ))

            contents.append(types.Content(role="tool", parts=response_parts))

        # Ran out of steps without a final answer -- force one.
        contents.append(types.Content(role="user", parts=[types.Part.from_text(
            text="Please give your final answer now, based on the evidence gathered so far.")]))
        response = generate_with_retry(
            self.client, model=self.model_name, contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        trace.append({"step": max_steps, "type": "final_answer_forced", "content": response.text})
        return {"answer": response.text, "trace": trace, "sources": sorted(sources)}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Can mitigation measures be taken into account in an assessment under Article 6(3) of the Habitats Directive?"
    agent = SingleAgent()
    result = agent.answer(q)
    print("ANSWER:\n", result["answer"])
    print("\nSOURCES:", result["sources"])
    print("\nTRACE:")
    for step in result["trace"]:
        print(" ", step)
