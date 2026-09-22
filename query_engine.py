"""
query_engine.py
Orchestrates the full 4-stage pipeline using xAI Grok via OpenAI-compatible API.
"""

import os
import time
import json
import re
from openai import OpenAI
from typing import Optional

from data_loader import load_data, get_schema_description
from intent_parser import parse_intent, intent_to_human_explanation
from code_generator import generate_and_execute
from self_critic import self_critique
from feedback_store import FeedbackStore


MODEL_NAME = "openai/gpt-oss-120b"        # best available on this Groq account
XAI_BASE_URL = "https://api.groq.com/openai/v1"


def init_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=XAI_BASE_URL)


class ChatContext:
    """Maintains conversation history for contextual follow-up queries."""
    def __init__(self, max_turns: int = 10):
        self.history = []
        self.max_turns = max_turns

    def add(self, query: str, result_summary: str):
        self.history.append({"query": query, "summary": result_summary})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context_string(self) -> str:
        if not self.history:
            return ""
        lines = ["PREVIOUS QUERIES IN THIS SESSION:"]
        for i, turn in enumerate(self.history[-3:], 1):
            lines.append(f"  {i}. Q: {turn['query']} -> {turn['summary']}")
        return "\n".join(lines)

    def clear(self):
        self.history = []


def decompose_query(query: str, schema: str, client: OpenAI) -> list:
    """For complex queries, decompose into ordered sub-queries."""
    prompt = f"""You are a data analyst. Determine if this query requires multi-hop reasoning.

SCHEMA SUMMARY:
{schema[:500]}

QUERY: "{query}"

If the query can be answered in ONE step, return: {{"multi_hop": false, "sub_queries": []}}

If it needs multiple steps (e.g., rank within groups then aggregate), return:
{{
  "multi_hop": true,
  "sub_queries": ["Step 1: ...", "Step 2: ... using results from step 1"],
  "reasoning": "why multi-hop is needed"
}}

Return ONLY valid JSON."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    try:
        plan = json.loads(response.choices[0].message.content)
        if plan.get("multi_hop") and plan.get("sub_queries"):
            return plan["sub_queries"]
    except Exception:
        pass
    return []


class QueryEngine:
    def __init__(self, api_key: str):
        self.client = init_client(api_key)
        self.data = load_data()
        self.schema = get_schema_description(self.data)
        self.feedback_store = FeedbackStore()
        self.chat_context = ChatContext()

    def run(self, query: str, session_id: str = "default") -> dict:
        """Execute the full 4-stage pipeline and return a complete result dict."""
        start_time = time.time()

        # Context enrichment for follow-ups
        context = self.chat_context.get_context_string()
        enriched_query = f"{query}\n\n{context}" if context else query

        # ── Stage 1: Query DNA ─────────────────────────────────
        intent = parse_intent(enriched_query, self.schema, self.client, MODEL_NAME)
        intent_explanation = intent_to_human_explanation(intent, query)

        # Multi-hop check for complex queries
        sub_queries = []
        if intent.get("complexity") == "complex":
            sub_queries = decompose_query(query, self.schema, self.client)

        # ── Stage 2: Code Gen + Execute ────────────────────────
        code_output = generate_and_execute(
            query=enriched_query,
            intent=intent,
            schema=self.schema,
            df=self.data["sales"],
            targets=self.data["targets"],
            client=self.client,
            model_name=MODEL_NAME,
        )

        # ── Stage 3: Self-Critique ─────────────────────────────
        critique = self_critique(
            query=query,
            intent=intent,
            code_output=code_output,
            client=self.client,
            model_name=MODEL_NAME,
        )

        # ── Stage 4: Feedback Bias ─────────────────────────────
        feedback_bias = self.feedback_store.get_feedback_bias(query)
        raw_confidence = critique["confidence_score"]
        adjusted_confidence = max(0.0, min(1.0, raw_confidence + feedback_bias["score_adjustment"]))

        self.chat_context.add(query, critique.get("what_was_understood", query[:60]))

        elapsed = round(time.time() - start_time, 2)

        return {
            "query": query,
            "session_id": session_id,
            "intent": intent,
            "intent_explanation": intent_explanation,
            "sub_queries": sub_queries,
            "generated_logic": {
                "pandas": code_output["pandas_code"],
                "sql": code_output["sql_query"],
                "approach": code_output["approach_summary"],
                "attempts": code_output["attempts"],
            },
            "result": code_output["result"],
            "anomalies": code_output.get("anomalies", []),
            "execution_error": code_output.get("execution_error"),
            "confidence_score": adjusted_confidence,
            "confidence_raw": raw_confidence,
            "confidence_label": critique["confidence_label"],
            "what_was_understood": critique["what_was_understood"],
            "how_result_was_derived": critique["how_result_was_derived"],
            "caveats": critique.get("caveats", []),
            "result_seems_correct": critique.get("result_seems_correct", False),
            "suggested_improvement": critique.get("suggested_improvement"),
            "feedback_bias": feedback_bias,
            "elapsed_seconds": elapsed,
        }

    def submit_feedback(self, query: str, is_correct: bool, notes: str = None) -> dict:
        return self.feedback_store.add_feedback(query, is_correct, notes=notes)

    def get_feedback_stats(self) -> dict:
        return self.feedback_store.get_stats()

    def clear_context(self):
        self.chat_context.clear()

    def get_sample_queries(self) -> list:
        return [q["query"] for q in self.data["nl_queries"]]

    def run_all_benchmarks(self) -> list:
        results = []
        for item in self.data["nl_queries"]:
            result = self.run(item["query"])
            result["expected_logic"] = item.get("expected_logic", "")
            results.append(result)
        return results
