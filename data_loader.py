"""
data_loader.py
Loads and prepares all datasets for the query engine.
"""

import pandas as pd
import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_data():
    """Load all datasets and return as a dict."""
    sales = pd.read_csv(DATA_DIR / "sales_data.csv")
    targets = pd.read_csv(DATA_DIR / "targets.csv")

    with open(DATA_DIR / "data_dictionary.json", encoding="utf-8-sig") as f:
        data_dict = json.load(f)

    with open(DATA_DIR / "nl_queries.json", encoding="utf-8-sig") as f:
        nl_queries = json.load(f)

    # Preprocess
    sales["order_date"] = pd.to_datetime(sales["order_date"])
    sales["year"] = sales["order_date"].dt.year
    sales["month"] = sales["order_date"].dt.to_period("M").astype(str)
    sales["quarter"] = sales["order_date"].dt.to_period("Q").astype(str)
    sales["revenue"] = (
        sales["quantity"] * sales["unit_price"] * (1 - sales["discount"])
    ).round(2)

    return {
        "sales": sales,
        "targets": targets,
        "data_dict": data_dict,
        "nl_queries": nl_queries,
    }


def get_schema_description(data: dict) -> str:
    """Generate a concise schema description for LLM prompts."""
    sales = data["sales"]
    dd = data["data_dict"]

    columns = sales.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in sales.dtypes.items()}
    sample = sales.head(2).to_dict(orient="records")

    schema = f"""
SALES DATA (DataFrame name: `df`):
Columns: {columns}
Dtypes: {dtypes}
Sample rows: {sample}

TARGETS DATA (DataFrame name: `targets`):
Columns: {data['targets'].columns.tolist()}
Sample: {data['targets'].head(3).to_dict(orient='records')}

COMPUTED COLUMNS:
- revenue = quantity * unit_price * (1 - discount)
- month = order_date formatted as 'YYYY-MM'
- year = order_date year
- quarter = order_date quarter as 'YYYY-QN'

METRICS:
{json.dumps(dd.get('metrics', {}), indent=2)}

DIMENSIONS:
{dd.get('dimensions', [])}

SYNONYMS (map these to actual column/metric names):
{json.dumps(dd.get('synonyms', {}), indent=2)}
"""
    return schema.strip()
