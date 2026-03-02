# Golden Set Guide

A **golden set** is a labeled dataset used to evaluate retrieval quality: each item is a **(query, expected document ids)** pair. The evaluator runs your retrieval pipeline (embedding + search) and checks whether the expected documents appear in the top‑k results.

## Format

### Corpus (documents)

- **JSONL**: one JSON object per line.
- Each object must have:
  - `id`: unique document id (string).
  - `text`: document content to embed and search (or `content`, which is normalized to `text`).
- Example:

```jsonl
{"id": "doc1", "text": "Python is a programming language."}
{"id": "doc2", "text": "RAG combines retrieval with language models."}
```

- You can also use a single JSON array of such objects (`.json`).

### Queries (golden set)

- **JSONL** or **JSON**.
- Each query object must have:
  - `query_id`: unique id for the query (for reporting).
  - `query`: the search query string.
  - `expected_ids`: list of document ids that are considered relevant (at least one should appear in top‑k for a “success”).
- Example:

```jsonl
{"query_id": "q1", "query": "What is RAG?", "expected_ids": ["doc2"]}
{"query_id": "q2", "query": "programming language", "expected_ids": ["doc1"]}
```

- If you have multiple relevant docs per query, list them all in `expected_ids`; recall@k will count how many appear in top‑k.

## Config (YAML)

Point the eval suite at your corpus and query files, and choose metrics:

```yaml
name: my-eval

corpus:
  path: data/corpus.jsonl   # relative to config file dir or use ../data/...

queries:
  path: data/queries.jsonl

metrics:
  k: [1, 5, 10]   # recall@1, recall@5, recall@10
  mrr: true       # mean reciprocal rank
```

Paths in `corpus.path` and `queries.path` are resolved relative to the **directory of the config file**. Use `../data/corpus.jsonl` if your data lives next to the folder that contains the config.

## How to build a good golden set

1. **Representative queries**: Cover the main use cases and query types (keywords, natural language, long/short).
2. **Stable relevance**: Expected ids should be clearly relevant so that small embedding changes don’t flip labels.
3. **Size**: Start with 20–50 queries; add more for regression confidence.
4. **No train/test leakage**: Queries and labels are for **eval only**. Don’t use the same queries (or doc content) to train or tune your embedding model or index. See [Avoiding leakage](avoiding_leakage.md).

## Running the eval

```bash
retrieval-eval run config/example_suite.yaml --embedding openai --retriever memory -o report.json --md report.md
```

Use `--embedding sentence-transformers` for a local model (install with `pip install -e ".[local]"`).
