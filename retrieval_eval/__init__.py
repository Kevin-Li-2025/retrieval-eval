"""
Retrieval Eval: RAG retrieval quality evaluation and regression testing.
"""

__version__ = "0.1.0"

from retrieval_eval.core import run_eval, load_golden_set, EvalResult
from retrieval_eval.metrics import compute_metrics, recall_at_k, mrr

__all__ = [
    "__version__",
    "run_eval",
    "load_golden_set",
    "EvalResult",
    "compute_metrics",
    "recall_at_k",
    "mrr",
]
