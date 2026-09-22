"""
code_generator.py
Stage 2: Dual-Path Code Generation — using xAI Grok via OpenAI-compatible API.
"""

import json
import re
import traceback
import pandas as pd
import numpy as np
from openai import OpenAI


MAX_RETRIES = 2


def build_codegen_prompt(query: str, intent: dict, schema: str, error: str = None) -> str:
    retry_block = f"\nPREVIOUS ATTEMPT FAILED:\n{error}\nFix the Pandas code.\n" if error else ""
    return f"""You are a Python/SQL data analyst. Generate BOTH Pandas code and SQL for this query.

SCHEMA:
{schema}

USER QUERY: "{query}"

PARSED INTENT:
{json.dumps(intent, indent=2)}
{retry_block}
Return ONLY valid JSON with these keys:
{{
  "pandas_code": "multi-line Python code that assigns final output to variable `result`. Use `df` for sales_data, `targets` for targets. pandas=`pd`, numpy=`np`. No imports.",
  "sql_query": "equivalent readable SQL with CTEs if needed",
  "approach_summary": "1-2 sentence description of the analytical approach"
}}

PANDAS RULES:
- Assign final output to `result` (DataFrame or scalar)
- revenue column already exists: quantity * unit_price * (1 - discount)
- month column format: '2024-03'
- For YoY: group by year, sum revenue, then pct_change()
- For top-N per group: use groupby + rank() or nlargest()
- For target comparison: merge df with targets on ['region','month']
- Use .round(2) on floats, .reset_index() for clean DataFrames

SQL RULES:
- Table names: sales_data, targets
- revenue = quantity * unit_price * (1 - discount)
- Use CTEs for complex logic, indent properly
"""


def execute_pandas(code: str, df: pd.DataFrame, targets: pd.DataFrame):
    namespace = {"df": df.copy(), "targets": targets.copy(), "pd": pd, "np": np}
    try:
        exec(compile(code, "<query>", "exec"), namespace)
        return namespace.get("result"), None
    except Exception:
        return None, traceback.format_exc()


def detect_anomalies(result) -> list:
    anomalies = []
    if not isinstance(result, pd.DataFrame):
        return anomalies
    for col in result.select_dtypes(include=[np.number]).columns:
        series = result[col].dropna()
        if len(series) < 3:
            continue
        mean, std = series.mean(), series.std()
        if std == 0:
            continue
        for idx in series[(series - mean).abs() > 2 * std].index:
            anomalies.append({
                "column": col,
                "value": round(float(series[idx]), 2),
                "mean": round(float(mean), 2),
                "std": round(float(std), 2),
                "row": str(result.index.get_loc(idx) if idx in result.index else idx),
            })
    return anomalies


def result_to_serializable(result):
    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return {"type": "dataframe", "columns": result.columns.tolist(),
                "data": result.round(2).fillna("N/A").values.tolist(), "shape": list(result.shape)}
    if isinstance(result, pd.Series):
        return {"type": "series", "index": [str(i) for i in result.index.tolist()],
                "values": [round(v, 2) if isinstance(v, float) else v for v in result.tolist()]}
    if isinstance(result, (int, float, np.integer, np.floating)):
        return {"type": "scalar", "value": round(float(result), 4)}
    return {"type": "text", "value": str(result)}


def generate_and_execute(
    query: str, intent: dict, schema: str,
    df: pd.DataFrame, targets: pd.DataFrame,
    client: OpenAI, model_name: str,
) -> dict:
    error_context = None
    for attempt in range(MAX_RETRIES + 1):
        prompt = build_codegen_prompt(query, intent, schema, error=error_context)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw)

        try:
            generated = json.loads(raw)
        except json.JSONDecodeError:
            error_context = f"Response was not valid JSON: {raw[:200]}"
            continue

        pandas_code = generated.get("pandas_code", "")
        result, exec_error = execute_pandas(pandas_code, df, targets)

        if exec_error and attempt < MAX_RETRIES:
            error_context = exec_error
            continue

        return {
            "pandas_code": pandas_code,
            "sql_query": generated.get("sql_query", ""),
            "approach_summary": generated.get("approach_summary", ""),
            "result": result_to_serializable(result),
            "anomalies": detect_anomalies(result),
            "execution_error": exec_error,
            "attempts": attempt + 1,
        }

    return {
        "pandas_code": "", "sql_query": "",
        "approach_summary": "Failed to generate executable code.",
        "result": None, "anomalies": [],
        "execution_error": error_context,
        "attempts": MAX_RETRIES + 1,
    }
