# Intelligent Analytics Query Engine

A robust web application that converts natural language into executable analytics using a novel 4-stage Generative AI pipeline.

## 1. Approach
Our approach breaks the complex problem of Text-to-SQL/Pandas into a highly reliable **4-stage pipeline**:
1. **Stage 1 (Query DNA):** Instead of immediately generating code, the LLM first parses the user's query into a structured JSON intent (extracting metrics, dimensions, filters, temporal logic, and complexity).
2. **Stage 2 (Code Generation):** Using the Query DNA, the LLM generates both executable Pandas code and human-readable SQL. The Pandas code is executed safely in a local sandbox. If it fails, the error is injected back into the LLM for an auto-correction retry. Anomaly detection automatically flags statistical outliers (>2σ) in the results.
3. **Stage 3 (Self-Critique):** The LLM reviews its own output table against the original query to assign a confidence score (0.0 to 1.0), explain its derivation, and flag caveats.
4. **Stage 4 (Feedback Loop):** A TF-IDF similarity engine checks past user feedback (👍/👎) on similar queries and adjusts the final confidence score dynamically.

## 2. Architecture
- **Backend:** FastAPI (Python), providing both REST endpoints and a WebSocket endpoint for real-time pipeline streaming.
- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`) for blazing fast inference via OpenAI-compatible endpoints.
- **Frontend:** A vanilla HTML/JS/CSS Single Page Application (SPA). It uses no build tools for maximum portability while featuring a modern dark theme, animated progress, and tabbed results.
- **Data Layer:** Pandas for in-memory data manipulation and anomaly detection.

## 3. Tradeoffs
- **Vanilla JS/HTML vs. React/Vue:** We chose a zero-build vanilla JS frontend for maximum simplicity and ease of review, trading off the component reusability that a framework like React provides.
- **TF-IDF vs. Vector Database:** We implemented a pure-Python TF-IDF system for the feedback loop to avoid heavy dependencies (like PyTorch or ChromaDB). This is incredibly fast and portable, but trades off the deep semantic matching a true embedding-based Vector DB would offer.
- **`exec()` Sandbox vs. Docker Isolation:** The Pandas code runs inside a Python `exec()` block with a restricted namespace. This trades total security (which would require an isolated Docker container) for speed and architectural simplicity.

## 4. Sample Outputs
**Query:** `"Top 2 cities by profit"`
- **Result Tab:** Shows a tabular result of `[{"city": "New York", "profit": 200}, {"city": "San Francisco", "profit": 180}]`.
- **Query DNA Tab:** Extracted JSON with `metrics: ["profit"]`, `dimensions: ["city"]`, `ranking: {"top_n": 2, "order": "desc"}`.
- **Code Tab:** Generates Pandas: `result = df.groupby('city')['profit'].sum().nlargest(2).reset_index()` and equivalent SQL.
- **Explanation Tab:** High Confidence (0.9). "Understood as ranking top 2 cities by total profit. Computed by grouping by city, summing profit, and taking the top 2."

## 5. Improvements (Given More Time)
If we had more time to expand the project, we would implement the following novel features and refinements:
1. **UI Improvements (Automated Data Viz):** Upgrade the tabular results to render interactive data visualizations (e.g., using Plotly.js, D3.js, or Chart.js). The UI would intelligently choose to render a bar chart, line graph, or pie chart depending on the extracted Query DNA.
2. **Vector Database Integration (Semantic Search):** Replace the current TF-IDF feedback similarity with a local vector database (like ChromaDB or FAISS) using sentence-transformers to capture true semantic meaning for the feedback loop.
3. **WebAssembly / Secure Sandboxing:** Run the generated analytical Python code entirely inside the browser using Pyodide (WebAssembly), or in an isolated ephemeral Docker container, eliminating all security risks associated with server-side code execution.
