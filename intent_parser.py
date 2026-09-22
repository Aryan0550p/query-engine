"""
intent_parser.py
Stage 1: Semantic Intent Graph (Query DNA) — using xAI Grok via OpenAI-compatible API.
"""

import json
import re
from openai import OpenAI


INTENT_SCHEMA = {
    "intent_type": "aggregation | ranking | comparison | contribution | trend | nested",
    "metrics": ["list of metrics e.g. revenue, profit, orders, avg_order_value"],
    "dimensions": ["list of group-by columns e.g. region, city, product_category"],
    "filters": [{"column": "...", "operator": "=|>|<|in|between", "value": "..."}],
    "temporal": {
        "column": "month|year|quarter",
        "period": "specific period string or null",
        "comparison": "yoy|mom|none"
    },
    "ranking": {
        "top_n": "integer or null",
        "order": "desc|asc",
        "partition_by": "column or null"
    },
    "join_targets": "bool — whether targets table is needed",
    "complexity": "simple|medium|complex"
}


def build_intent_prompt(query: str, schema: str) -> str:
    return f"""You are a data analytics expert. Parse this natural language query into a structured Query DNA JSON object.

DATASET SCHEMA:
{schema}

USER QUERY: "{query}"

Return ONLY valid JSON matching this structure (fill nulls where not applicable):
{json.dumps(INTENT_SCHEMA, indent=2)}

Rules:
- Map synonyms: sales->revenue, income->revenue, earnings->profit, aov->avg_order_value
- Detect temporal: "March" -> filter month='2024-03', "YoY" -> trend comparison
- Detect ranking: "Top 2", "top product" -> set ranking.top_n
- If query needs targets comparison -> join_targets: true
- complexity: simple=single agg, medium=grouped/filtered, complex=nested/window/yoy
"""


def parse_intent(query: str, schema: str, client: OpenAI, model_name: str) -> dict:
    """Call LLM to extract Query DNA from the natural language query."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": build_intent_prompt(query, schema)}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)

    return json.loads(raw)


def intent_to_human_explanation(intent: dict, query: str) -> str:
    """Convert intent dict to a readable explanation string."""
    parts = []
    parts.append(f"**Query type**: {intent.get('intent_type', 'unknown')}")

    if intent.get("metrics"):
        parts.append(f"**Measuring**: {', '.join(intent['metrics'])}")
    if intent.get("dimensions"):
        parts.append(f"**Grouped by**: {', '.join(intent['dimensions'])}")

    filters = intent.get("filters", [])
    if filters:
        filter_strs = [f"{f['column']} {f['operator']} {f['value']}" for f in filters]
        parts.append(f"**Filtered by**: {', '.join(filter_strs)}")

    temporal = intent.get("temporal", {}) or {}
    if temporal.get("period"):
        parts.append(f"**Time period**: {temporal['period']}")
    if temporal.get("comparison") not in (None, "none", "null", ""):
        parts.append(f"**Comparison**: {temporal['comparison'].upper()}")

    ranking = intent.get("ranking", {}) or {}
    if ranking.get("top_n"):
        s = f"**Ranking**: Top {ranking['top_n']}"
        if ranking.get("partition_by"):
            s += f" per {ranking['partition_by']}"
        parts.append(s)

    if intent.get("join_targets"):
        parts.append("**Uses targets table**: Yes")

    parts.append(f"**Complexity**: {intent.get('complexity', 'unknown')}")
    return " | ".join(parts)
