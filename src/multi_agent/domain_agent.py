"""
Step 4 - Domain-specialized agent (used by the multi-agent system, Task B).

Almost identical to single_agent.agent.SingleAgent, except every tool call is
pinned to one legal domain ("agricultural" or "environmental") so this agent
only ever sees judgments from its own domain -- plus the 6 cross-domain
judgments, which are included in both domain-specific FAISS indices (see
Step 2's build_index.py), so neither specialist misses them.

The supervisor (supervisor.py) treats this class as a tool it can consult;
it is not talked to directly by the end user.
"""
import json
import sys
from pathlib import Path

from google import genai
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "single_agent"))
sys.path.insert(0, str(REPO_ROOT / "src" / "common"))
import tools as base_tools  # noqa: E402
from gemini_utils import to_gemini_tool  # noqa: E402

DOMAIN_SYSTEM_PROMPT = """You are a legal research specialist in EU {domain} law, \
answering questions about CJEU (Court of Justice of the EU) preliminary rulings under \
Article 267 TFEU, restricted to the {domain} sub-corpus (a supervisor has already decided \
your domain is relevant to this question).

Always ground your answers in retrieved judgments -- never answer from memory alone. \
Use search_corpus to find relevant judgments, then get_case_by_number to pull the full \
text/operative part of the most relevant one(s) before answering. Use list_cases_by_filter \
for pure counting/browsing questions.

When you have enough evidence, give a final answer with NO further tool calls. Your final \
answer must cite the specific case number(s) it relies on, and state briefly if your domain's \
corpus does not contain a clear answer, rather than guessing."""

MAX_STEPS = 5


def _domain_scoped_tools(domain: str):
    """Wrap the shared tools so legal_domain is always forced to `domain`,
    and drop legal_domain from the exposed schema (the agent shouldn't be
    able to escape its own domain)."""

    def search_corpus(query, k=5, thematic_area=None, eu_instrument=None, member_state=None):
        return base_tools.search_corpus(query, k=k, legal_domain=domain,
                                         thematic_area=thematic_area,
                                         eu_instrument=eu_instrument,
                                         member_state=member_state)

    def list_cases_by_filter(thematic_area=None, eu_instrument=None, member_state=None, limit=20):
        return base_tools.list_cases_by_filter(legal_domain=domain, thematic_area=thematic_area,
                                                eu_instrument=eu_instrument, member_state=member_state,
                                                limit=limit)

    tools = {
        "search_corpus": search_corpus,
        "get_case_by_number": base_tools.get_case_by_number,
        "list_cases_by_filter": list_cases_by_filter,
    }
    # Bonus: graph expansion is deliberately NOT domain-restricted -- its value
    # (per the exam guide) is precisely in surfacing related judgments across
    # the environmental/agricultural boundary that a domain-scoped search would miss.
    if getattr(base_tools, "_GRAPH_AVAILABLE", False):
        tools["expand_via_graph"] = base_tools.TOOLS["expand_via_graph"]

    schemas = []
    for schema in base_tools.TOOL_SCHEMAS:
        fn = dict(schema["function"])
        params = dict(fn["parameters"])
        props = dict(params["properties"])
        props.pop("legal_domain", None)
        params = {**params, "properties": props}
        fn = {**fn, "parameters": params}
        schemas.append({"type": "function", "function": fn})
    return tools, schemas


class DomainAgent:
    def __init__(self, domain: str, model_name="gemini-3.1-flash-lite", client=None, api_key=None):
        assert domain in ("agricultural", "environmental")
        self.domain = domain
        self.model_name = model_name
        self.client = client or genai.Client(api_key=api_key)
        self.tools, schemas = _domain_scoped_tools(domain)
        self.tool = to_gemini_tool(schemas)
        self.config = types.GenerateContentConfig(
            system_instruction=DOMAIN_SYSTEM_PROMPT.format(domain=domain),
            tools=[self.tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def answer(self, question: str, max_steps: int = MAX_STEPS) -> dict:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        trace = []
        sources = set()

        for step in range(max_steps):
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=self.config,
            )
            calls = response.function_calls or []

            if not calls:
                trace.append({"step": step, "type": "final_answer", "content": response.text})
                return {"domain": self.domain, "answer": response.text, "trace": trace, "sources": sorted(sources)}

            contents.append(response.candidates[0].content)

            response_parts = []
            for call in calls:
                fn_name = call.name
                args = dict(call.args or {})
                result = self.tools[fn_name](**args) if fn_name in self.tools else {"error": f"Unknown tool '{fn_name}'"}

                for hit in (result if isinstance(result, list) else [result]):
                    if isinstance(hit, dict) and hit.get("case_number"):
                        sources.add(hit["case_number"])
                    elif isinstance(hit, dict) and hit.get("metadata", {}).get("case_number"):
                        sources.add(hit["metadata"]["case_number"])

                trace.append({
                    "step": step, "type": "action", "tool": fn_name, "args": args,
                    "observation_preview": json.dumps(result, ensure_ascii=False)[:500],
                })
                response_parts.append(types.Part.from_function_response(name=fn_name, response={"result": result}))

            contents.append(types.Content(role="tool", parts=response_parts))

        contents.append(types.Content(role="user", parts=[types.Part.from_text(
            text="Please give your final answer now, based on the evidence gathered so far.")]))
        response = self.client.models.generate_content(
            model=self.model_name, contents=contents,
            config=types.GenerateContentConfig(system_instruction=DOMAIN_SYSTEM_PROMPT.format(domain=self.domain)),
        )
        trace.append({"step": max_steps, "type": "final_answer_forced", "content": response.text})
        return {"domain": self.domain, "answer": response.text, "trace": trace, "sources": sorted(sources)}
