# Thematic analysis pipeline

Turns the luxury-beauty interview transcripts in `new_transcript/` into a
single combined analysis: keyword/theme-lexicon charts, a 6-theme qualitative
coding breakdown, and a report tying both together. One command runs
everything.

## Prerequisites

- Python 3.9+
- macOS/Linux/Windows shell access
- No system packages beyond Python are required; all dependencies are pip-installable (see below).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs: `matplotlib`, `numpy`, `pandas`, `seaborn`,
`plotly`, `kaleido`, `networkx`, `wordcloud`.

## Run everything

```bash
python run_pipeline.py
```

This is the **only script you run directly**. It:

1. Loads every transcript in `new_transcript/`, excludes the recurring
   interviewer ("Bahade Yash"), and aggregates the remaining participant
   across all 12 transcripts into keyword/theme charts (`src/thematic_analysis.py`).
2. Computes frequency, prevalence, stance, co-occurrence, and salience
   ranking from `coded_themes.csv` (the manually/LLM-coded excerpts),
   cross-referenced with the keyword-hit counts from step 1
   (`src/compute_theme_stats.py`).
3. Renders all 6 required qualitative-coding figures using one fixed
   theme→color palette shared with step 1's charts (`src/visualize_theme_stats.py`).
4. Writes one combined `report.md` referencing every chart from steps 1–3.

Each run writes to a **new versioned folder** — `analysis_output/v1/`,
`analysis_output/v2/`, etc. — so re-running never overwrites a previous
result. Everything from a given run (charts, CSVs, JSON, the coded dataset,
and the report) lives together in that one `vN/` folder.

Optional flags:

```bash
python run_pipeline.py \
  --input new_transcript \        # transcript folder to batch-load
  --coded-csv coded_themes.csv \  # qualitative excerpt coding source
  --themes src/themes.qualitative.json \
  --output analysis_output        # base folder; writes into <output>/vN/
```

## Repository layout

```
run_pipeline.py          <- single entry point, run this
requirements.txt
coded_themes.csv          <- qualitative coding data (candidate_id, theme, quote, paraphrase, stance)
new_transcript/            <- 12 source interview transcripts (.docx)
src/                       <- analysis engine (imported by run_pipeline.py, not run directly)
  thematic_analysis.py       keyword/theme-lexicon engine + CLI (can still run standalone)
  compute_theme_stats.py     Stage 1/2: frequency/prevalence/stance/co-occurrence/salience
  visualize_theme_stats.py   Stage 3: the 6 required figures
  themes.qualitative.json    the 6 canonical theme names + keyword lexicon (default)
  themes.json, themes.beauty.json   alternate keyword lexicons
analysis_output/            <- generated, gitignored; one vN/ subfolder per run
transcripts/, transcript.docx  <- legacy sample/demo data for `thematic_analysis.py --demo`
```

## Notes

- `coded_themes.csv` is qualitative-coding output produced by reading the
  transcripts (manually or via an LLM pass) — no script in this repo
  regenerates it. If the transcripts change, that file needs to be re-coded
  by hand before re-running the pipeline.
- The 6 theme names (`Performance First`, `Sustainability as a Conditional
  Value-Add`, `Blurred Meanings of Clean and Sustainable Beauty`,
  `Credibility Through Evidence, Transparency and Consistency`,
  `Greenwashing and Communication Scepticism`, `Reconciling Luxury and
  Sustainability`) are defined once, as `QUALITATIVE_THEMES` in
  `src/thematic_analysis.py`, and imported everywhere else so naming and
  chart colors stay uniform across both the keyword and qualitative charts.
- Candidate identities are anonymized to "Candidate 1"–"12" throughout,
  consistent with the confidentiality promise made in the interviews.
