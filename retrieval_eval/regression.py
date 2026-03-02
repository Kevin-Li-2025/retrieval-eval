"""
Regression / A/B diff: compare two eval result JSONs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_results(path_a: str | Path, path_b: str | Path) -> Tuple[Dict, Dict]:
    from .report import load_result
    return load_result(path_a), load_result(path_b)


def diff_metrics(metrics_a: Dict[str, float], metrics_b: Dict[str, float]) -> List[Dict[str, Any]]:
    """Compare metrics; return list of {metric, run_a, run_b, diff, improved}."""
    all_keys = sorted(set(metrics_a) | set(metrics_b))
    rows = []
    for k in all_keys:
        if k == "num_queries":
            continue
        va = metrics_a.get(k)
        vb = metrics_b.get(k)
        if va is None or vb is None:
            continue
        try:
            diff = float(vb) - float(va)
            # higher is better for recall, mrr
            improved = diff > 0
            rows.append({
                "metric": k,
                "run_a": va,
                "run_b": vb,
                "diff": diff,
                "improved": improved,
            })
        except (TypeError, ValueError):
            pass
    return rows


def format_diff_md(rows: List[Dict[str, Any]], run_a_name: str = "baseline", run_b_name: str = "current") -> str:
    """Format diff rows as Markdown table."""
    lines = [
        "## Regression diff",
        "",
        f"| Metric | {run_a_name} | {run_b_name} | Diff | Status |",
        "|--------|-------------|-------------|------|--------|",
    ]
    for r in rows:
        status = "✅" if r["improved"] else ("⚠️ regressed" if r["diff"] < 0 else "—")
        diff_str = f"{r['diff']:+.4f}" if isinstance(r["run_a"], (int, float)) and isinstance(r["run_b"], (int, float)) else str(r["diff"])
        va = r["run_a"]
        vb = r["run_b"]
        if isinstance(va, float) and 0 < va < 1:
            va = f"{va:.4f}"
        if isinstance(vb, float) and 0 < vb < 1:
            vb = f"{vb:.4f}"
        lines.append(f"| {r['metric']} | {va} | {vb} | {diff_str} | {status} |")
    return "\n".join(lines)


def regression_report(path_a: str | Path, path_b: str | Path) -> Dict[str, Any]:
    """Build full regression report dict (metrics diff + summary)."""
    ra, rb = load_results(path_a, path_b)
    ma = ra.get("metrics") or {}
    mb = rb.get("metrics") or {}
    rows = diff_metrics(ma, mb)
    regressions = [r for r in rows if not r["improved"] and r["diff"] < 0]
    return {
        "run_a": str(path_a),
        "run_b": str(path_b),
        "metrics_diff": rows,
        "regressions": regressions,
        "summary": {
            "num_regressions": len(regressions),
            "all_improved": len(regressions) == 0,
        },
    }
