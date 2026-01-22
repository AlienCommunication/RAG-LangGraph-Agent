# Agentic, Context-Aware RAG System (Production-Grade)

This repository implements a **production-ready, agentic Retrieval-Augmented Generation (RAG) system** designed to answer factual questions **strictly grounded in source documents**, with **explicit hallucination control, evaluation loops, and retry intelligence**.

The system is intentionally designed to mirror **real-world enterprise RAG systems**, not demo-level pipelines.

---

## 1. High-Level Architecture

At a high level, the system is composed of **four clearly separated layers**:

```
Ingestion  →  Retrieval  →  Agentic Reasoning  →  Evaluation
```

### Architecture Diagram (Conceptual)

```
User Query
   ↓
Hybrid Retrieval (Keyword + Vector)
   ↓
Re-Ranking (Cross-Encoder)
   ↓
LangGraph Agent
   ├─ Generate Answer
   ├─ Judge Faithfulness
   └─ Retry if Confidence < Threshold
   ↓
Final Answer + Confidence
   ↓
Evaluation Metrics + Artifacts
```

---

## 2. Why an *Agentic* RAG System (Not a Simple Pipeline)

A traditional RAG pipeline follows a **single-pass flow**:

```
retrieve → generate → return
```

This project intentionally **does NOT** do that.

Instead, it uses a **stateful agent** implemented with **LangGraph**, enabling:

* Explicit state tracking
* Conditional branching
* Retry logic based on answer confidence
* Clear separation of reasoning steps

This design is critical for:

* Reducing hallucinations
* Handling ambiguous queries
* Building systems that can *reason about their own failures*

---

## 3. Agent Orchestration (`agent/graph.py`)

### What the Agent Does

The agent is a **state machine**, not a function.

It executes the following steps:

1. **Retrieve** candidate documents
2. **Re-rank** them for precision
3. **Generate** an answer strictly from context
4. **Judge** whether the answer is fully supported
5. **Retry** retrieval if confidence is too low

### Why LangGraph Was Chosen

LangGraph was chosen because it:

* Makes control-flow explicit
* Supports conditional loops
* Forces correct state handling
* Matches how production agents are built

The retry loop is defined as:

```python
if confidence < 0.7 and retries < max_retries:
    retry retrieval
else:
    end execution
```

This is **not prompt engineering** — it is **system-level reasoning**.

---

## 4. Explicit Agent State (`agent/state.py`)

The agent state is represented as a structured object:

```python
RAGState(
  query,
  retrieved_docs,
  ranked_docs,
  answer,
  confidence,
  retries
)
```

### Why Explicit State Matters

Without explicit state:

* The system cannot reason over failures
* Retries become blind repetition
* Evaluation is impossible to trace

With explicit state:

* Every decision is explainable
* Failures are inspectable
* Behavior is deterministic and debuggable

---

## 5. Retrieval Strategy (Why Hybrid Retrieval)

### Problem with Vector-Only Retrieval

Pure vector search often fails on:

* Numbers (e.g., voltages, IDs)
* Exact terms (protocol names, part numbers)
* Regulatory or technical language

### Hybrid Retrieval (`retrievers/hybrid.py`)

This system combines:

1. **Keyword retrieval (TF-IDF)**
2. **Vector similarity search (embeddings)**

The final score is a **weighted fusion**:

```
final_score = α · keyword_score + (1 − α) · vector_score
```

### Why This Matters

* Keyword search ensures **exact term recall**
* Vector search ensures **semantic recall**
* Together, they significantly reduce false negatives

This pattern is standard in **enterprise search systems**.

---

## 6. Re-Ranking (Why It Is Essential)

### What Re-Ranking Does (`retrievers/reranker.py`)

Initial retrieval returns **candidates**, not the best answers.

Re-ranking uses a **cross-encoder** to score:

```
(query, document) → relevance score
```

### Why a Cross-Encoder Was Used

* Evaluates query and document *together*
* Much more precise than embedding similarity
* Dramatically improves top-K quality

Re-ranking is one of the **most effective hallucination-reduction techniques** in RAG systems.

---

## 7. Context-Aware Chunking (`ingest.py`)

### Chunking Strategy

* **Chunk size:** 500 tokens
* **Overlap:** 50 tokens
* **Separators:** paragraph → sentence → word

### Why This Chunking Was Chosen

* Large enough to preserve semantic meaning
* Small enough to fit multiple chunks in context
* Overlap prevents boundary information loss

This balances:

* Recall
* Precision
* Context window efficiency

---

## 8. Embedding Model Choice

**Embedding model:** `all-MiniLM-L6-v2`

### Why This Model

* Strong semantic performance
* Fast inference
* Low memory footprint
* Industry-standard for dense retrieval

This model is widely used in **production RAG systems** where latency and cost matter.

---

## 9. Idempotent Ingestion (Why We Reset the Vector DB)

During ingestion:

```python
if vector_db_exists:
    delete_it
rebuild_from_source()
```

### Why Idempotency Matters

* Guarantees reproducibility
* Prevents duplicate embeddings
* Ensures ingestion is deterministic

This is critical for:

* CI/CD pipelines
* Data refresh jobs
* Debugging retrieval issues

---

## 10. Hallucination Control Strategy

Hallucination control is implemented at **three layers**:

### 1️⃣ Prompt Constraints

The generation prompt explicitly states:

> “Answer strictly from context.”

### 2️⃣ Agent Judge (`judge_node`)

A second LLM evaluates:

* Whether **every claim** is supported by context
* Returns a **confidence score (0–1)**

### 3️⃣ Retry Logic

If confidence < 0.7:

* Retrieval is re-attempted
* Re-ranking is re-applied
* Answer is regenerated

This creates a **closed feedback loop**, not blind generation.

---

## 11. Citations (What They Are and Why They Matter)

### What Are Citations

Citations link an answer back to:

* Specific chunks
* Specific sections of the source document

### Why Citations Matter

* Increase trust
* Enable auditing
* Prevent silent hallucinations

The ingestion pipeline enriches metadata (`chunk_id`, `source`) explicitly to support citations.

---

## 12. Evaluation Framework (`evaluation/evaluate.py`)

Evaluation is **first-class**, not an afterthought.

### Metrics Used

#### 1️⃣ Retrieval Recall@K

**What it means:**
Did we retrieve the documents needed to answer the question?

**How it’s calculated:**

```
Recall@K = retrieved_relevant_docs / total_relevant_docs
```

**How to interpret:**

* `1.0` → perfect retrieval
* `< 1.0` → retrieval gap
* `None` → no ground truth available

---

#### 2️⃣ Context Relevance

**Question:**
Is the retrieved context useful, or just noise?

Evaluated by an LLM judge (PASS / FAIL).

---

#### 3️⃣ Faithfulness (Groundedness)

**Question:**
Is every factual claim supported by the context?

This is the **primary hallucination metric**.

---

#### 4️⃣ Answer Relevance

**Question:**
Did the system actually answer the user’s intent?

Prevents evasive or irrelevant answers from scoring well.

---

## 13. How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Set Environment Variables

```bash
export GOOGLE_API_KEY=your_key_here
```

### 3️⃣ Ingest Data

```bash
python ingest.py data/technical_manual.txt
```

### 4️⃣ Run the Agent

```bash
python run_agent.py "What is the voltage requirement?"
```

### 5️⃣ Run Evaluation

```bash
python -m evaluation.evaluate
```

Evaluation results are saved to:

```
artifacts/evaluation_results.csv
```

---

## 14. Design Philosophy Summary

This system is built around **five core principles**:

1. **Explicit state over implicit behavior**
2. **Hybrid retrieval over single-strategy search**
3. **Re-ranking for precision, not speed alone**
4. **Judged answers, not blind generation**
5. **Evaluation as a first-class citizen**

---

## 15. What This Project Demonstrates

* Production-grade RAG design
* Agentic reasoning workflows
* Hallucination-aware answer generation
* Enterprise-style evaluation metrics
* Clean, scalable project structure

