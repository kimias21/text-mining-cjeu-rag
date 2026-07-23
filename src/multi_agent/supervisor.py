"""
Step 4 - Supervisor (Task B, multi-agent system).

The supervisor never touches the corpus directly. Its only two tools are
"consult" calls to the two DomainAgents (agricultural / environmental); each
one runs its own full ReAct loop (search -> retrieve -> reason) internally
and returns a grounded, cited answer. The supervisor's job is purely routing
and synthesis:
  1. decide whether the question needs the agricultural specialist, the
     environmental specialist, or both (many questions plausibly touch both,
     e.g. organic-farming pesticide residue limits, so the supervisor is
     encouraged to consult both when unsure rather than guess a single domain),
  2. if both were consulted, merge the two answers into one coherent response
     (noting where they agree/differ), citing every case number used.

This maps onto Task B's required "multi-agent system with a supervisor,"
covers the FULL corpus (both domain indices, six cross-domain judgments
present in each), and stays fully comparable to Task A's `SingleAgent`:
`answer()` returns the same {"answer", "trace", "sources"} contract, with
`trace` additionally nesting each consulted specialist's own trace.
"""
import json
import sys
from pathlib import Path

from google import genai
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "common"))
from gemini_utils import to_gemini_tool, generate_with_retry  # noqa: E402
from domain_agent import DomainAgent  # noqa: E402

SUPERVISOR_SYSTEM_PROMPT = """You are the supervisor of a legal research system covering \
CJEU (Court of Justice of the EU) preliminary rulings under Article 267 TFEU, split into two \
specialist agents: one for EU agricultural law, one for EU environmental law. A small number \
of judgments are relevant to both domains.

You do not have direct access to the judgment corpus. For every question, consult at least \
one specialist using consult_agricultural_agent and/or consult_environmental_agent. If the \
question is clearly and exclusively about one domain, consult only that specialist. If it is \
ambiguous, could plausibly involve both domains (e.g. pesticide residues in organic farming, \
environmental conditionality of CAP payments), or you are not sure, consult BOTH specialists \
before answering -- do not guess a single domain when unsure.

Once you have the specialist response(s), give a final answer with NO further tool calls. \
Your final answer must:
- directly answer the question, synthesizing both specialists' input if you consulted both \
  (note if they overlap, agree, or address different aspects),
- cite the specific case number(s) the specialist(s) relied on,
- state briefly if neither specialist found a clear answer, rather than guessing.
"""

SUPERVISOR_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "consult_agricultural_agent",
            "description": "Ask the EU agricultural law specialist (with access to the full agricultural sub-corpus, including cross-domain judgments) to research and answer a question.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "The question to forward, verbatim or refined for this specialist's domain."}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consult_environmental_agent",
            "description": "Ask the EU environmental law specialist (with access to the full environmental sub-corpus, including cross-domain judgments) to research and answer a question.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "The question to forward, verbatim or refined for this specialist's domain."}},
                "required": ["question"],
            },
        },
    },
]

MODEL_NAME = "gemini-3.1-flash-lite"
MAX_STEPS = 4


class Supervisor:
    def __init__(self, model_name=MODEL_NAME, client=None, api_key=None):
        self.model_name = model_name
        self.client = client or genai.Client(api_key=api_key)
        self.agents = {
            "agricultural": DomainAgent("agricultural", model_name=model_name, client=self.client),
            "environmental": DomainAgent("environmental", model_name=model_name, client=self.client),
        }
        self.tool = to_gemini_tool(SUPERVISOR_TOOL_SCHEMAS)
        self.config = types.GenerateContentConfig(
            system_instruction=SUPERVISOR_SYSTEM_PROMPT,
            tools=[self.tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _consult(self, domain: str, question: str) -> dict:
        return self.agents[domain].answer(question)

    def answer(self, question: str, max_steps: int = MAX_STEPS) -> dict:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        trace = []
        sources = set()
        consulted_domains = []

        for step in range(max_steps):
            response = generate_with_retry(
                self.client, model=self.model_name, contents=contents, config=self.config,
            )
            calls = response.function_calls or []

            if not calls:
                trace.append({"step": step, "type": "final_answer", "content": response.text})
                return {
                    "answer": response.text, "trace": trace, "sources": sorted(sources),
                    "consulted_domains": consulted_domains,
                }

            contents.append(response.candidates[0].content)

            response_parts = []
            for call in calls:
                fn_name = call.name
                args = dict(call.args or {})
                domain = "agricultural" if fn_name == "consult_agricultural_agent" else "environmental"

                if fn_name not in ("consult_agricultural_agent", "consult_environmental_agent"):
                    result = {"error": f"Unknown tool '{fn_name}'"}
                else:
                    sub_question = args.get("question", question)
                    result = self._consult(domain, sub_question)
                    consulted_domains.append(domain)
                    sources.update(result.get("sources", []))
                    trace.append({
                        "step": step, "type": "delegate", "domain": domain,
                        "sub_question": sub_question, "sub_trace": result.get("trace"),
                    })

                trace.append({
                    "step": step, "type": "action", "tool": fn_name, "args": args,
                    "observation_preview": json.dumps(
                        {"answer": result.get("answer"), "sources": result.get("sources")},
                        ensure_ascii=False)[:500],
                })
                response_parts.append(types.Part.from_function_response(name=fn_name, response={"result": {
                    "answer": result.get("answer"), "sources": result.get("sources"),
                }}))

            contents.append(types.Content(role="tool", parts=response_parts))

        contents.append(types.Content(role="user", parts=[types.Part.from_text(
            text="Please give your final synthesized answer now, based on the specialist responses gathered so far.")]))
        response = generate_with_retry(
            self.client, model=self.model_name, contents=contents,
            config=types.GenerateContentConfig(system_instruction=SUPERVISOR_SYSTEM_PROMPT),
        )
        trace.append({"step": max_steps, "type": "final_answer_forced", "content": response.text})
        return {
            "answer": response.text, "trace": trace, "sources": sorted(sources),
            "consulted_domains": consulted_domains,
        }


if __name__ == "__main__":
    import sys as _sys
    q = _sys.argv[1] if len(_sys.argv) > 1 else "What residue limits apply to pesticides used in organic farming, and are they subject to environmental impact assessment?"
    supervisor = Supervisor()
    result = supervisor.answer(q)
    print("ANSWER:\n", result["answer"])
    print("\nCONSULTED:", result["consulted_domains"])
    print("SOURCES:", result["sources"])
    print("\nTRACE:")
    for step in result["trace"]:
        print(" ", step)
