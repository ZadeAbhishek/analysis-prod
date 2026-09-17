#!/usr/bin/env python3
"""Stage 1+2: compute theme frequency/prevalence/stance/co-occurrence/salience
from coded_themes.csv, optionally merged with keyword-hit counts from
thematic_analysis.py's Analysis object, into the Stage 2 JSON schema.
"""
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

from thematic_analysis import QUALITATIVE_THEMES as THEMES

STANCES = ["Positive", "Negative", "Mixed", "Neutral"]
CANDIDATES = [f"Candidate {i}" for i in range(1, 13)]


def compute_stats(coded_csv: Path, keyword_hits: Optional[Dict[str, int]] = None) -> dict:
    """Compute the Stage 2 JSON from ``coded_csv`` (candidate_id,theme,quote,
    paraphrase,stance rows). ``keyword_hits`` -- theme -> lexicon hit count
    from thematic_analysis.py's Analysis.theme_totals -- is folded in as
    ``theme_frequency.keyword_hits`` so this stays cross-referenced with the
    main keyword-engine run instead of being a fully separate dataset.
    """
    with open(coded_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    overall_freq = Counter(r["theme"] for r in rows)
    by_candidate = {th: {c: 0 for c in CANDIDATES} for th in THEMES}
    for r in rows:
        by_candidate[r["theme"]][r["candidate_id"]] += 1

    theme_candidates = defaultdict(set)
    for r in rows:
        theme_candidates[r["theme"]].add(r["candidate_id"])
    prevalence_counts = {th: len(theme_candidates[th]) for th in THEMES}
    prevalence_pct = {th: round(100 * prevalence_counts[th] / 12, 1) for th in THEMES}

    stance_distribution = {}
    for th in THEMES:
        sub = [r for r in rows if r["theme"] == th]
        total = len(sub) or 1
        counts = Counter(r["stance"] for r in sub)
        stance_distribution[th] = {
            s: {"count": counts.get(s, 0), "pct": round(100 * counts.get(s, 0) / total, 1)} for s in STANCES
        }

    co_occurrence = {th_i: {th_j: 0 for th_j in THEMES} for th_i in THEMES}
    for th_i, th_j in combinations(THEMES, 2):
        n = len(theme_candidates[th_i] & theme_candidates[th_j])
        co_occurrence[th_i][th_j] = co_occurrence[th_j][th_i] = n
    for th in THEMES:
        co_occurrence[th][th] = len(theme_candidates[th])

    total_excerpts = sum(overall_freq.values())
    salience = []
    for th in THEMES:
        norm_freq = 100 * overall_freq[th] / total_excerpts
        score = (norm_freq + prevalence_pct[th]) / 2
        salience.append((th, score))
    salience.sort(key=lambda x: x[1], reverse=True)

    salience_ranking = [
        {
            "rank": i + 1, "theme": th, "score": round(score, 1),
            "excerpt_count": overall_freq[th], "prevalence_pct": prevalence_pct[th],
            "is_performance_first": th == "Performance First",
        }
        for i, (th, score) in enumerate(salience)
    ]

    return {
        "theme_frequency": {
            "overall": {th: overall_freq[th] for th in THEMES},
            "by_candidate": by_candidate,
            "keyword_hits": keyword_hits or {},
        },
        "theme_prevalence": {"counts": prevalence_counts, "percentages": prevalence_pct},
        "stance_distribution": stance_distribution,
        "co_occurrence_matrix": co_occurrence,
        "salience_ranking": salience_ranking,
    }


def write_summary_table(stats: dict, path: Path) -> None:
    stance_distribution = stats["stance_distribution"]
    overall_freq = stats["theme_frequency"]["overall"]
    prevalence_pct = stats["theme_prevalence"]["percentages"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("| Theme | Frequency (excerpts) | Prevalence (% of 12) | Dominant stance |\n")
        f.write("|---|---|---|---|\n")
        for row in stats["salience_ranking"]:
            th = row["theme"]
            dom_stance = max(STANCES, key=lambda s: stance_distribution[th][s]["count"])
            f.write(f"| {th} | {overall_freq[th]} | {prevalence_pct[th]:.0f}% | {dom_stance} |\n")


if __name__ == "__main__":
    stats = compute_stats(Path("coded_themes.csv"))
    Path("theme_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_summary_table(stats, Path("summary_table.md"))
    is_pf_1 = stats["salience_ranking"][0]["theme"] == "Performance First"
    print(f"Performance First ranks #1 by composite salience: {is_pf_1}")
    print("Wrote theme_stats.json, summary_table.md")
