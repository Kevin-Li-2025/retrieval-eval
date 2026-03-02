# Avoiding Data Leakage in Retrieval Eval

**Data leakage** here means any situation where the eval set (golden set) influences model training or index construction, so that reported metrics overestimate real-world performance.

## Rules of thumb

1. **Golden set is query-only**  
   The eval uses **only** the query text and the list of expected document ids. The corpus (documents) is built once from your real index or a fixed snapshot. The eval does not “train” on the golden set.

2. **Corpus is fixed**  
   Use the same corpus (or a dedicated eval index) for all runs you compare. Don’t add documents that were created from the golden-set queries (e.g. synthetic docs derived from the query text).

3. **Don’t tune on the same queries**  
   If you use the golden set to choose hyperparameters (e.g. chunk size, top_k, reranker), you are effectively “training” on the eval set. Prefer:
   - a **separate dev set** for tuning, and
   - the **golden set** only for final reporting and regression.

4. **Separate train vs eval data for embeddings**  
   If you fine-tune an embedding model, its training data must not include the golden-set queries or the exact text of the eval documents. Keep a clear split: train corpus + train labels vs eval corpus + golden set.

## What this tool does

- **Input**: A corpus (id + text) and a golden set (query_id, query, expected_ids).
- **Process**: Embeds the corpus (if using in-memory retriever), embeds each query, runs retrieval, then computes recall@k and MRR.
- **No training**: No gradient updates; no use of golden-set labels to change the model or the index. So within this eval, the only “leakage” risk is if you yourself build the corpus or train the embedder using the golden set or the eval docs.

## Recommended workflow

1. **Define the golden set** from a sample of real (or realistic) user queries and label expected doc ids.
2. **Lock the golden set** and the eval corpus snapshot (no edits for reporting).
3. **Tune** retrieval (model, index, k, etc.) on a **separate dev set** or via A/B in production.
4. **Run this eval** for regression: after any change, run the same suite and diff (e.g. `retrieval-eval diff baseline.json current.json`) to ensure metrics don’t drop.

This keeps your reported recall@k and MRR trustworthy and comparable over time.
