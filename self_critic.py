"""
self_critic.py
Stage 3: Self-Critique Pass — using xAI Grok via OpenAI-compatible API.
"""

import json
import re
from openai import OpenAI


def build_critique_prompt(query: str, intent: dict, code_output: dict) -> str:
    r = code_output.get("result")
    if r and isinstance(r, dict):
        if r.get("type") == "dataframe":
            preview = f"DataFrame {r['shape']}: columns={r['columns']}, first rows={r['data'][:3]}"
        elif r.get("type") == "scalar":
            preview = f"Scalar value: {r['value']}"
        elif r.get("type") == "series":
            preview = f"Series: {dict(zip(r['index'][:5], r['values'][:5]))}"
        else:
            preview = str(r)
    else:
        preview = "No result (execution failed)"

    error_note = f"\nEXECUTION ERROR: {code_output.get('execution_error')}" if code_output.get("execution_error") else ""

    return f"""You are a senior data analyst performing a quality review.

ORIGINAL USER QUERY: "{query}"

WHAT THE SYSTEM UNDERSTOOD:
{json.dumps(intent, indent=2)}

GENERATED PANDAS CODE:
{code_output.get('pandas_code', 'N/A')}
{error_note}

ACTUAL RESULT PRODUCED:
{preview}

APPROACH USED: {code_output.get('approach_summary', 'N/A')}
ANOMALIES DETECTED: {code_output.get('anomalies', [])}

Critically evaluate whether the result correctly answers the user query.

Return ONLY valid JSON:
{{
  "confidence_score": <float 0.0 to 1.0>,
  "confidence_label": "High | Medium | Low",
  "what_was_understood": "<1 sentence>",
  "how_result_was_derived": "<1-2 sentences>",
  "caveats": ["<concerns or limitations>"],
  "result_seems_correct": <true | false>,
  "suggested_improvement": "<optional suggestion or null>"
}}

Scoring: 0.9-1.0=clearly correct, 0.7-0.9=likely correct, 0.5-0.7=partially correct, 0.0-0.5=likely wrong.
"""


def self_critique(
    query: str, intent: dict, code_output: dict,
    client: OpenAI, model_name: str,
) -> dict:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": build_critique_prompt(query, intent, code_output)}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)

    try:
        critique = json.loads(raw)
    except json.JSONDecodeError:
        critique = {
            "confidence_score": 0.5, "confidence_label": "Medium",
            "what_was_understood": "Unable to parse critique",
            "how_result_was_derived": code_output.get("approach_summary", ""),
            "caveats": ["Self-critique parsing failed"],
            "result_seems_correct": False, "suggested_improvement": None,
        }

    critique["confidence_score"] = max(0.0, min(1.0, float(critique.get("confidence_score", 0.5))))
    return critique
