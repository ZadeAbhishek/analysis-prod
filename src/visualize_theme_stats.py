#!/usr/bin/env python3
"""Stage 3: render the 6 qualitative-coding figures from a Stage-2 stats dict.

Uses Matplotlib + Seaborn for the static charts, NetworkX for the
co-occurrence graph layout, and Plotly (via kaleido) for that same network,
falling back to Matplotlib+NetworkX if kaleido isn't installed. Colors reuse
thematic_analysis.qualitative_theme_colors() so every theme has one fixed hex
color across this script AND the keyword-lexicon charts in thematic_analysis.py.
"""
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from thematic_analysis import QUALITATIVE_THEMES as THEMES, qualitative_theme_colors

DPI = 300
STANCES = ["Positive", "Negative", "Mixed", "Neutral"]
CANDIDATES = [f"Candidate {i}" for i in range(1, 13)]
STANCE_COLORS = {"Positive": "#2ca02c", "Negative": "#d62728", "Mixed": "#ff7f0e", "Neutral": "#7f7f7f"}


def _short(name: str, n: int = 28) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def render_qualitative_figures(stats: dict, out_dir: Path) -> List[Path]:
    """Draw all 6 required figures into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    palette = qualitative_theme_colors()
    written: List[Path] = []

    # Figure 1 — horizontal bar: prevalence (%) sorted descending.
    prev = stats["theme_prevalence"]["percentages"]
    order = sorted(THEMES, key=lambda t: prev[t])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([_short(t) for t in order], [prev[t] for t in order], color=[palette[t] for t in order])
    ax.set_xlabel("Prevalence (% of 12 candidates)")
    ax.set_title("Figure 1 — Theme Prevalence Across Candidates")
    ax.set_xlim(0, 105)
    for i, t in enumerate(order):
        ax.text(prev[t] + 1, i, f"{prev[t]:.0f}%", va="center", fontsize=9)
    fig.tight_layout()
    p = out_dir / "fig1_prevalence.png"
    fig.savefig(p, dpi=DPI); plt.close(fig); written.append(p)

    # Figure 2 — stacked bar: excerpts per theme, stacked by stance.
    stance_dist = stats["stance_distribution"]
    order2 = sorted(THEMES, key=lambda t: stats["theme_frequency"]["overall"][t], reverse=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bottoms = np.zeros(len(order2))
    for stance in STANCES:
        vals = np.array([stance_dist[t][stance]["count"] for t in order2])
        ax.bar([_short(t) for t in order2], vals, bottom=bottoms, label=stance, color=STANCE_COLORS[stance])
        bottoms += vals
    ax.set_ylabel("Excerpt count")
    ax.set_title("Figure 2 — Excerpts per Theme by Stance")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Stance", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    p = out_dir / "fig2_stance_stack.png"
    fig.savefig(p, dpi=DPI); plt.close(fig); written.append(p)

    # Figure 3 — heatmap: candidates (rows) x themes (columns), excerpt counts.
    by_cand = stats["theme_frequency"]["by_candidate"]
    mat = pd.DataFrame({_short(t): [by_cand[t][c] for c in CANDIDATES] for t in THEMES}, index=CANDIDATES)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Excerpt count"}, ax=ax, linewidths=0.5)
    ax.set_title("Figure 3 — Excerpt Counts: Candidates × Themes")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    p = out_dir / "fig3_candidate_theme_heatmap.png"
    fig.savefig(p, dpi=DPI); plt.close(fig); written.append(p)

    # Figure 4 — theme co-occurrence network (Plotly, falling back to Matplotlib+NetworkX).
    co = stats["co_occurrence_matrix"]
    G = nx.Graph()
    for t in THEMES:
        G.add_node(t, size=co[t][t])
    for i, ti in enumerate(THEMES):
        for tj in THEMES[i + 1:]:
            if co[ti][tj] > 0:
                G.add_edge(ti, tj, weight=co[ti][tj])
    pos = nx.circular_layout(G)
    p = out_dir / "fig4_cooccurrence_network.png"
    rendered = False
    try:
        import plotly.graph_objects as go

        edge_x, edge_y = [], []
        for u, v in G.edges:
            edge_x += [pos[u][0], pos[v][0], None]
            edge_y += [pos[u][1], pos[v][1], None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=2, color="#999999"), hoverinfo="none")
        node_trace = go.Scatter(
            x=[pos[t][0] for t in G.nodes], y=[pos[t][1] for t in G.nodes],
            mode="markers+text", text=[_short(t, 18) for t in G.nodes], textposition="top center",
            marker=dict(size=[18 + 3 * G.nodes[t]["size"] for t in G.nodes],
                        color=[palette[t] for t in G.nodes], line=dict(width=1, color="white")),
            hovertext=[f"{t}: mentioned by {G.nodes[t]['size']} candidates" for t in G.nodes], hoverinfo="text",
        )
        fig4 = go.Figure(data=[edge_trace, node_trace])
        fig4.update_layout(
            title="Figure 4 — Theme Co-occurrence Network (edge = candidates sharing both themes)",
            showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor="white", width=1400, height=1000, margin=dict(l=40, r=40, t=80, b=40),
        )
        fig4.write_image(str(p), scale=DPI / 96)
        rendered = True
    except Exception as e:  # pragma: no cover - environment-dependent
        print(f"  ! Plotly export unavailable ({e}); using Matplotlib+NetworkX for Figure 4")
    if not rendered:
        fig, ax = plt.subplots(figsize=(10, 8))
        weights = [G[u][v]["weight"] for u, v in G.edges]
        nx.draw_networkx_edges(G, pos, ax=ax, width=[0.6 + 0.5 * w for w in weights], edge_color="#999999")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[300 + 40 * G.nodes[t]["size"] for t in G.nodes],
                                node_color=[palette[t] for t in G.nodes], edgecolors="white", linewidths=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax, labels={t: _short(t, 20) for t in G.nodes}, font_size=8)
        for (u, v), w in zip(G.edges, weights):
            x, y = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            ax.text(x, y, str(w), fontsize=8, color="#333333", ha="center")
        ax.set_title("Figure 4 — Theme Co-occurrence Network (edge = candidates sharing both themes)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(p, dpi=DPI); plt.close(fig)
    written.append(p)

    # Figure 5 — radar chart: composite salience score per theme.
    rank = {r["theme"]: r["score"] for r in stats["salience_ranking"]}
    values = [rank[t] for t in THEMES]
    angles = np.linspace(0, 2 * np.pi, len(THEMES), endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="#1f77b4", linewidth=2)
    ax.fill(angles, values, color="#1f77b4", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_short(t, 22) for t in THEMES], fontsize=8)
    ax.set_title("Figure 5 — Composite Salience Score by Theme", pad=30)
    fig.tight_layout()
    p = out_dir / "fig5_salience_radar.png"
    fig.savefig(p, dpi=DPI); plt.close(fig); written.append(p)

    # Figure 6 — small multiples: one panel per theme, stance proportions.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, t in zip(axes.flat, THEMES):
        vals = [stance_dist[t][s]["count"] for s in STANCES]
        if sum(vals) == 0:
            ax.axis("off"); continue
        ax.pie(vals, labels=STANCES, colors=[STANCE_COLORS[s] for s in STANCES], autopct="%1.0f%%",
               textprops={"fontsize": 8})
        ax.set_title(_short(t, 34), fontsize=10)
    fig.suptitle("Figure 6 — Stance Proportions per Theme", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out_dir / "fig6_stance_small_multiples.png"
    fig.savefig(p, dpi=DPI); plt.close(fig); written.append(p)

    return written


if __name__ == "__main__":
    with open("theme_stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    paths = render_qualitative_figures(stats, Path("figures"))
    print(f"Wrote {len(paths)} figures (300 DPI) to figures/")
