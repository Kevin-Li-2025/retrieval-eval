# Retrieval Eval

**Industry-grade RAG retrieval quality evaluation and regression testing.**

Measure recall@k, MRR, and custom metrics over your retrieval pipeline. Compare runs (A/B or regression) and get a clear report with failure cases for CI or manual review.

## Features

- **Golden-set driven**: Define (query, expected doc ids) once; run against any embedding + vector store combo.
- **Multiple backends**: OpenAI embeddings, Qdrant, and an in-memory retriever out of the box; plug in your own.
- **Metrics**: Recall@1 / @5 / @10, MRR; extend with custom metrics.
- **Regression**: Diff two runs (e.g. after changing model or index) to catch regressions.
- **Reports**: JSON (machine-readable) + Markdown summary with failure cases for triage.
- **No leakage**: Golden set is query-only; corpus is separate so eval doesn’t train on test data.

## Install

```bash
pip install -e ".[local]"   # with sentence-transformers for local embeddings
# or
pip install -e .             # OpenAI + Qdrant only
```

## Quick start

1. **Define a golden set** (see [Golden set guide](docs/golden_set_guide.md)):

```yaml
# config/golden.yaml
name: my-rag-eval
corpus:
  path: data/corpus.jsonl   # one JSON object per line: {"id": "...", "text": "..."}
queries:
  path: data/queries.jsonl  # {"query_id": "...", "query": "...", "expected_ids": ["id1", "id2"]}
metrics:
  k: [1, 5, 10]
  mrr: true
```

2. **Run evaluation** (in-memory retriever with OpenAI embeddings):

```bash
retrieval-eval run config/golden.yaml --embedding openai --retriever memory --output report.json
```

3. **View report and failures**:

```bash
retrieval-eval report report.json --md report.md
```

4. **Regression**: run again after a change, then diff:

```bash
retrieval-eval run config/golden.yaml --embedding openai --retriever memory -o run2.json
retrieval-eval diff report.json run2.json --md regression.md
```

## Configuration

- **Embeddings**: `openai` (default), `sentence-transformers` (install with `[local]`).
- **Retrievers**: `memory` (builds in-memory index from corpus), `qdrant` (existing Qdrant collection).
- **Golden set**: JSONL or JSON; expected_ids define relevance. See [docs/golden_set_guide.md](docs/golden_set_guide.md) and [docs/avoiding_leakage.md](docs/avoiding_leakage.md).

## License

MIT.
