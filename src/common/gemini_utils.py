"""
Shared helper: convert OpenAI-style tool schemas (nested
{"type": "function", "function": {...}}, as defined in single_agent/tools.py)
into a Gemini `types.Tool`, so schema definitions aren't duplicated between
the single-agent (Task A) and multi-agent (Task B) systems.
"""
from google.genai import types


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
