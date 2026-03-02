"""
CLI: run, report, diff.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .core import run_eval_from_config, run_eval, load_golden_set, EvalResult
from .schema import load_config, load_golden_set_from_config
from .report import write_json_report, write_markdown_report, load_result
from .regression import regression_report, format_diff_md


@click.group()
@click.version_option(version="0.1.0", prog_name="retrieval-eval")
def main() -> None:
    """RAG retrieval quality evaluation and regression testing."""
    pass


@main.command("run")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--embedding", "-e", default="openai", help="Embedding provider: openai | sentence-transformers")
@click.option("--retriever", "-r", default="memory", help="Retriever: memory | qdrant")
@click.option("--output", "-o", "output_path", type=click.Path(path_type=Path), help="Write result JSON here")
@click.option("--md", "md_path", type=click.Path(path_type=Path), help="Also write Markdown report here")
def run(config_path: Path, embedding: str, retriever: str, output_path: Path | None, md_path: Path | None) -> None:
    """Run evaluation from a YAML config (golden set + metrics)."""
    result = run_eval_from_config(
        config_path,
        embedding=embedding,
        retriever=retriever,
        output_path=output_path,
    )
    d = result.to_dict()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_report(d, output_path)
        click.echo(f"Wrote JSON: {output_path}")
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown_report(d, md_path, title="Retrieval Eval Report")
        click.echo(f"Wrote Markdown: {md_path}")
    click.echo("Metrics:")
    for k, v in result.metrics.items():
        if k != "num_queries":
            click.echo(f"  {k}: {v}")
    click.echo(f"Failure cases: {len(result.failure_cases)}")


@main.command("report")
@click.argument("result_json", type=click.Path(exists=True, path_type=Path))
@click.option("--md", "md_path", type=click.Path(path_type=Path), required=True, help="Output Markdown path")
@click.option("--title", default="Retrieval Eval Report", help="Report title")
def report(result_json: Path, md_path: Path, title: str) -> None:
    """Generate Markdown report from a result JSON (metrics + failure cases)."""
    data = load_result(result_json)
    write_markdown_report(data, md_path, title=title)
    click.echo(f"Wrote: {md_path}")


@main.command("diff")
@click.argument("result_a", type=click.Path(exists=True, path_type=Path))
@click.argument("result_b", type=click.Path(exists=True, path_type=Path))
@click.option("--md", "md_path", type=click.Path(path_type=Path), help="Write regression diff Markdown here")
@click.option("--json-out", "json_path", type=click.Path(path_type=Path), help="Write full diff JSON here")
def diff(result_a: Path, result_b: Path, md_path: Path | None, json_path: Path | None) -> None:
    """Compare two result JSONs (regression / A-B)."""
    reg = regression_report(result_a, result_b)
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Regression Diff", "", f"- **Baseline:** `{result_a}`", f"- **Current:** `{result_b}`", ""]
        lines.append(format_diff_md(reg["metrics_diff"]))
        if reg["regressions"]:
            lines.extend(["", "## Regressions", ""])
            for r in reg["regressions"]:
                lines.append(f"- {r['metric']}: {r['run_a']} → {r['run_b']} ({r['diff']:+.4f})")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        click.echo(f"Wrote: {md_path}")
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)
        click.echo(f"Wrote: {json_path}")
    for row in reg["metrics_diff"]:
        status = "✅" if row["improved"] else ("⚠️" if row["diff"] < 0 else "—")
        click.echo(f"  {row['metric']}: {row['run_a']} → {row['run_b']} ({row['diff']:+.4f}) {status}")
    click.echo(f"Regressions: {reg['summary']['num_regressions']}")


if __name__ == "__main__":
    main()
