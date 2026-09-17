#!/usr/bin/env python3
"""Single driver: keyword/theme-lexicon analysis + qualitative excerpt-coding
stats + all figures, in one run, into one versioned output folder, with one
combined report. This is the only script meant to be run directly; the
engine modules it calls live in src/.

    python run_pipeline.py
    python run_pipeline.py --input new_transcript --output analysis_output
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from thematic_analysis import (
    DEFAULT_BATCH_DIR, DEFAULT_EXCLUDED_SPEAKERS, QUALITATIVE_THEMES, STOPWORDS,
    _normalize_speaker, load_themes, run_keyword_pipeline,
)
from compute_theme_stats import compute_stats, write_summary_table
from visualize_theme_stats import render_qualitative_figures

DEFAULT_THEMES_PATH = ROOT / "src" / "themes.qualitative.json"


def _next_version_dir(base: Path) -> Path:
    """Pick base/v1, base/v2, ... -- the next unused version, so a new run
    never overwrites a previous one's output."""
    base.mkdir(parents=True, exist_ok=True)
    used = [int(m.group(1)) for p in base.iterdir() if p.is_dir() and (m := re.fullmatch(r"v(\d+)", p.name))]
    return base / f"v{max(used, default=0) + 1}"


def write_combined_report(out_dir: Path, kw: dict, stats: dict, qual_figs: list) -> Path:
    a = kw["analysis"]
    lines = ["# Combined Thematic Analysis Report\n"]

    lines.append("## 1. Batch overview\n")
    lines.append(f"- Transcripts processed: {len(kw['transcript_summaries'])} (from `new_transcript/`)")
    lines.append(f"- Excluded speaker(s): {', '.join(kw['excluded_display']) or '(none)'}")
    lines.append(f"- Aggregated participant turns analysed: {len(a.turns)}")
    lines.append(f"- Theme lexicon: the 6 qualitative-coding themes (`{', '.join(QUALITATIVE_THEMES)}`)\n")

    lines.append("## 2. Keyword-lexicon analysis (src/thematic_analysis.py)\n")
    for c in kw["charts"]:
        lines.append(f"![{c.name}]({c.name})")
    lines.append("")

    lines.append("## 3. Qualitative excerpt coding (src/compute_theme_stats.py)\n")
    lines.append("Theme x frequency x prevalence x dominant stance, cross-referenced with "
                  "keyword-lexicon hit counts from the same run:\n")
    lines.append("| Theme | Excerpts | Prevalence | Dominant stance | Keyword hits |")
    lines.append("|---|---|---|---|---|")
    stance_dist = stats["stance_distribution"]
    overall = stats["theme_frequency"]["overall"]
    kw_hits = stats["theme_frequency"]["keyword_hits"]
    prevalence_pct = stats["theme_prevalence"]["percentages"]
    for row in stats["salience_ranking"]:
        th = row["theme"]
        dom = max(("Positive", "Negative", "Mixed", "Neutral"), key=lambda s: stance_dist[th][s]["count"])
        lines.append(f"| {th} | {overall[th]} | {prevalence_pct[th]:.0f}% | {dom} | {kw_hits.get(th, 0)} |")
    pf1 = stats["salience_ranking"][0]["theme"] == "Performance First"
    lines.append(f"\n**Performance First ranks #1 by composite salience score: {pf1}**\n")

    lines.append("## 4. Figures\n")
    for p in qual_figs:
        lines.append(f"![{p.name}]({p.name})")
    lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_pipeline(
    input_dir: str = DEFAULT_BATCH_DIR,
    coded_csv: str = "coded_themes.csv",
    themes_path: str = str(DEFAULT_THEMES_PATH),
    output_base: str = "analysis_output",
    exclude_speakers=None,
) -> dict:
    out_dir = _next_version_dir(Path(output_base))
    excluded_display = [s.strip() for s in (exclude_speakers or DEFAULT_EXCLUDED_SPEAKERS) if s.strip()]
    excluded_speakers = {_normalize_speaker(s) for s in excluded_display}

    themes = load_themes(Path(themes_path))
    stopwords = set(STOPWORDS)

    # 1. Keyword/theme-lexicon analysis -- writes its own charts/CSVs/report into out_dir.
    kw = run_keyword_pipeline(Path(input_dir), out_dir, themes, stopwords, excluded_speakers, excluded_display)

    # 2. Qualitative excerpt-coding stats, cross-referenced with this run's keyword hits.
    stats = compute_stats(Path(coded_csv), keyword_hits=kw["analysis"].theme_totals)
    (out_dir / "theme_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_summary_table(stats, out_dir / "summary_table.md")
    shutil.copy(coded_csv, out_dir / "coded_themes.csv")

    # 3. All 6 qualitative figures, same output folder as the keyword charts.
    qual_figs = render_qualitative_figures(stats, out_dir)

    # 4. One combined report referencing every chart and both datasets.
    report = write_combined_report(out_dir, kw, stats, qual_figs)
    print(f"\nPipeline complete. Everything is in {out_dir}/ -- see {report.name}")

    return {"keyword": kw, "stats": stats, "qual_figures": qual_figs, "report": report, "out_dir": out_dir}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the full keyword + qualitative-coding pipeline in one go.")
    p.add_argument("-i", "--input", default=DEFAULT_BATCH_DIR)
    p.add_argument("-c", "--coded-csv", default="coded_themes.csv")
    p.add_argument("-t", "--themes", default=str(DEFAULT_THEMES_PATH))
    p.add_argument("-o", "--output", default="analysis_output",
                   help="Base output folder; each run writes into a new output/vN subfolder.")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args.input, args.coded_csv, args.themes, args.output)
