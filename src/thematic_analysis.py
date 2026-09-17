#!/usr/bin/env python3
"""Thematic analysis of conversation transcripts.

Batch-loads every transcript in a folder (default: ``new_transcript/``),
drops all lines from a recurring excluded speaker (default: "Bahade Yash"),
and analyses whichever single participant remains in each transcript --
aggregating the results across the whole batch rather than comparing
speakers against each other. Extracts keywords and phrases, groups them
into themes, and renders a set of graphs:

  * Top keywords overall (bar)
  * Per-speaker keyword comparison (grouped bars) -- only when 2+ speakers remain
  * Distinctive words per speaker (TF-IDF) -- only when 2+ speakers remain
  * Theme distribution (bar)
  * Theme x speaker emphasis (heatmap) -- only when 2+ speakers remain
  * How themes rise and fall across the conversation (timeline)
  * Talk share per speaker -- only when 2+ speakers remain
  * Word clouds (overall + per speaker) -- if `wordcloud` is installed
  * Keyword co-occurrence network -- if `networkx` is installed

It also writes CSV tables and a readable Markdown report.

Usage
-----
    python thematic_analysis.py --input new_transcript/ --output output/
    python thematic_analysis.py --exclude-speaker ""   # disable exclusion/aggregation
    python thematic_analysis.py --demo          # writes sample data, then runs
    python thematic_analysis.py --help

Transcript format
-----------------
Lines look like ``Speaker: what they said``. An optional leading timestamp is
allowed, e.g. ``[00:01:23] Alex: ...`` or ``00:01 Alex: ...``. A line with no
speaker label continues the previous speaker's turn. Blank lines are ignored.
A label is treated as a speaker only if it recurs, so prose like
``Note: remember to...`` is not mistaken for a speaker. VTT/Teams/Zoom-style
cues (glued name + speech, no colon) are also auto-detected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never try to open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Optional dependencies -- the tool degrades gracefully if they are absent.
try:  # word clouds
    from wordcloud import WordCloud  # type: ignore

    HAVE_WORDCLOUD = True
except Exception:  # pragma: no cover - optional
    HAVE_WORDCLOUD = False

try:  # co-occurrence network
    import networkx as nx  # type: ignore

    HAVE_NETWORKX = True
except Exception:  # pragma: no cover - optional
    HAVE_NETWORKX = False


# --------------------------------------------------------------------------- #
# Palette (validated categorical + sequential ramp from the data-viz method)
# --------------------------------------------------------------------------- #
# Categorical slots, assigned in fixed order (never cycled arbitrarily).
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
# Single-hue sequential ramp (blue) light -> dark, for magnitude encoding.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#1c5cab", "#184f95", "#104281",
]
INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}


def apply_chart_style() -> None:
    """Set recessive, neutral-ink matplotlib defaults matching the palette."""
    plt.rcParams.update(
        {
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "axes.edgecolor": INK["axis"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.8,
            "axes.titlecolor": INK["primary"],
            "axes.labelcolor": INK["secondary"],
            "text.color": INK["primary"],
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelcolor": INK["secondary"],
            "ytick.labelcolor": INK["secondary"],
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def sequential_colors(n: int) -> List[str]:
    """Pick ``n`` evenly spaced steps from the blue sequential ramp."""
    if n <= 0:
        return []
    if n == 1:
        return [SEQUENTIAL_BLUE[4]]
    idx = np.linspace(2, len(SEQUENTIAL_BLUE) - 1, n)
    return [SEQUENTIAL_BLUE[int(round(i))] for i in idx]


# --------------------------------------------------------------------------- #
# Stopwords
# --------------------------------------------------------------------------- #
# A compact but solid English stopword list plus conversational fillers, so the
# tool needs no NLTK download. Extend via --extra-stopwords.
STOPWORDS = set(
    """
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's
its itself let's me more most mustn't my myself no nor not of off on once only or
other ought our ours ourselves out over own same shan't she she'd she'll she's
should shouldn't so some such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those through
to too under until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves
    """.split()
)
# Conversational fillers, back-channels, and low-signal verbs common in speech.
STOPWORDS.update(
    """
um uh umm uhh er erm hmm mm mhm mmhm yeah yep yup nope okay ok oh ah aha right
well like just really actually basically literally kind sort mean know think
guess say said says saying go goes going went get gets getting got gonna wanna
gotta lot lots thing things stuff way ways let lets maybe perhaps sure fine
okay's gonna's alright also even still much many one two really thats im ive
youre dont cant didnt wasnt isnt theyre weve youve id youd hes shes theres
whats gonna yeah's mm-hmm uh-huh
    """.split()
)


# --------------------------------------------------------------------------- #
# Batch input / speaker exclusion configuration
# --------------------------------------------------------------------------- #
# Directory that the dynamic loader reads every transcript file from.
DEFAULT_BATCH_DIR = "new_transcript"

# Recurring speaker (e.g. an interviewer present in every transcript) whose
# lines, keywords, themes, and charts are excluded from the analysis. Override
# with --exclude-speaker, or pass an empty value to disable exclusion.
DEFAULT_EXCLUDED_SPEAKERS = ["Bahade Yash"]

# Canonical label that every remaining (non-excluded) speaker is folded into,
# so a whole batch of transcripts is analysed as one aggregated participant
# instead of comparing candidate vs. candidate across files.
SINGLE_USER_LABEL = "Participant"


def _normalize_speaker(name: str) -> str:
    """Lowercase + collapse whitespace, for robust speaker-name comparisons."""
    return re.sub(r"\s+", " ", name.strip()).lower()


# --------------------------------------------------------------------------- #
# Default theme lexicon
# The 6 qualitative themes used for excerpt-level coding (coded_themes.csv) in
# the new_transcript/ interview batch. Single source of truth: every script
# that needs these names (compute_theme_stats.py, visualize_theme_stats.py,
# run_pipeline.py) imports QUALITATIVE_THEMES from here instead of
# redeclaring the list, so theme names stay uniform everywhere. The matching
# keyword lexicon lives in themes.qualitative.json and is the default
# --themes file for this batch, so the keyword-hit charts use the same 6
# theme names as the qualitative coding.
QUALITATIVE_THEMES: List[str] = [
    "Performance First",
    "Sustainability as a Conditional Value-Add",
    "Blurred Meanings of Clean and Sustainable Beauty",
    "Credibility Through Evidence, Transparency and Consistency",
    "Greenwashing and Communication Scepticism",
    "Reconciling Luxury and Sustainability",
]


def qualitative_theme_colors() -> Dict[str, str]:
    """Fixed theme -> hex color map, reusing the CATEGORICAL palette so every
    chart (keyword-lexicon and qualitative-coding alike) colors a theme the
    same way."""
    return {th: CATEGORICAL[i % len(CATEGORICAL)] for i, th in enumerate(QUALITATIVE_THEMES)}


# --------------------------------------------------------------------------- #
# Generic, editable starting point. Replace with a themes.json tuned to your
# domain. Keys are theme names; values are keyword/phrase lists (matched as
# whole words, case-insensitive; multi-word phrases are matched as substrings).
DEFAULT_THEMES: Dict[str, List[str]] = {
    "Work & Career": [
        "work", "job", "career", "office", "boss", "manager", "meeting",
        "project", "deadline", "client", "team", "promotion", "salary",
        "colleague", "workload", "overtime", "interview", "company",
    ],
    "Health & Wellbeing": [
        "health", "sleep", "tired", "stress", "stressed", "anxiety",
        "exercise", "gym", "doctor", "sick", "energy", "rest", "burnout",
        "therapy", "wellbeing", "workout", "diet", "mental",
    ],
    "Family & Relationships": [
        "family", "kids", "children", "partner", "wife", "husband", "mom",
        "dad", "mother", "father", "friend", "friends", "relationship",
        "marriage", "parents", "son", "daughter", "home",
    ],
    "Money & Finance": [
        "money", "budget", "savings", "save", "spend", "cost", "expensive",
        "rent", "mortgage", "debt", "loan", "invest", "salary", "income",
        "afford", "bills", "financial", "price",
    ],
    "Education & Learning": [
        "school", "college", "university", "course", "study", "learn",
        "learning", "degree", "class", "teacher", "student", "exam",
        "skills", "training", "read", "book", "knowledge",
    ],
    "Technology": [
        "technology", "computer", "software", "app", "phone", "internet",
        "online", "data", "ai", "code", "digital", "device", "platform",
        "website", "email", "tech",
    ],
    "Future & Plans": [
        "future", "plan", "plans", "goal", "goals", "hope", "dream",
        "next", "change", "decision", "decide", "opportunity", "move",
        "moving", "start", "startup",
    ],
    "Emotions & Feelings": [
        "happy", "sad", "angry", "worried", "excited", "nervous", "afraid",
        "scared", "love", "hate", "frustrated", "proud", "guilty",
        "overwhelmed", "grateful", "lonely", "feel", "feeling", "emotion",
    ],
}


# --------------------------------------------------------------------------- #
# Transcript parsing
# --------------------------------------------------------------------------- #
# Optional leading timestamp like [00:01:23], (1:02), or 00:01:23
_TIMESTAMP = r"(?:[\[(]?\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*[\])]?\s*)?"
# A speaker label: starts with a letter/digit/quote, up to ~40 chars, no
# sentence-ending punctuation, then a colon.
_SPEAKER_LINE = re.compile(
    rf"^\s*{_TIMESTAMP}(?P<speaker>[A-Za-z0-9][^:?!.]{{0,39}}?):\s+(?P<text>\S.*)$"
)

# A cue timestamp range as produced by VTT / Teams / Zoom exports, e.g.
# "00:00:13.547 --> 00:00:29.387". These formats glue the speaker's display
# name directly onto the speech with no colon: "BAHADE YashMeeting record...".
_VTT_TS = re.compile(
    r"\d{1,2}:\d{2}(?::\d{2})?[.,]?\d{0,3}\s*-->\s*\d{1,2}:\d{2}(?::\d{2})?[.,]?\d{0,3}"
)
# Display-name patterns at the start of a cue, tried in order, each requiring
# an uppercase letter right after the name to mark where the glued speech
# begins. Real exports mix several conventions for the same field, so multiple
# shapes are needed: "BAHADE Yash" (first word ALLCAPS), "Kaustubh LAHOTY"
# (surname ALLCAPS), "isha bahade" (all lowercase), and email-style handles
# like "vipulkumar.career@gmail.com".
_NAME_HEAD_PATTERNS = [
    re.compile(r"^([A-Z][A-Za-z'’.\-]*(?: [A-Z][A-Za-z'’\-]*)+)(?=[A-Z])"),
    re.compile(r"^([a-z][a-z'’.\-]*(?: [a-z'’\-]+)+)(?=[A-Z])"),
    re.compile(r"^([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})(?=[A-Z])"),
]


def _match_name_head(seg: str) -> Optional[str]:
    """Try each display-name pattern in turn; return the first match, if any."""
    for pattern in _NAME_HEAD_PATTERNS:
        m = pattern.match(seg)
        if m:
            return m.group(1).strip()
    return None


def read_docx(path: Path) -> str:
    """Extract visible paragraph text from a .docx file (no external deps)."""
    import html
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
    except (zipfile.BadZipFile, KeyError):
        return ""
    paras: List[str] = []
    for chunk in xml.split("</w:p>"):
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", chunk, flags=re.S)
        line = html.unescape("".join(texts)).strip()
        if line:
            paras.append(line)
    return "\n".join(paras)


def parse_vtt_transcript(text: str, source: str = "") -> List[dict]:
    """Parse a VTT/Teams/Zoom transcript whose cues glue name onto speech.

    The cue layout is ``<timestamp> --> <timestamp><Speaker Name><speech>`` with
    no delimiter between the name and the speech. Speaker names are discovered
    from the cues where a name is cleanly followed by a capitalised speech word,
    then matched as a literal prefix on every cue (which also recovers cues where
    the speech continues in lower case, e.g. ``BAHADE Yashtheir purchase...``).
    Consecutive cues from the same speaker are merged into one turn.
    """
    matches = list(_VTT_TS.finditer(text))
    if not matches:
        return []

    segments: List[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg = text[m.end():end].strip().lstrip("0123456789").strip()
        if seg:
            segments.append(seg)

    # Discover recurring speaker names.
    name_counts: Counter = Counter()
    for seg in segments:
        nm = _match_name_head(seg)
        if nm:
            name_counts[nm] += 1
    candidates = [n for n, c in name_counts.items() if c >= 2] or list(name_counts)

    # Drop any candidate that is just a longer, rare mis-split of a shorter,
    # far more common candidate (e.g. an occasional "...YashOK." cue getting
    # captured as "BAHADE YashO") -- keep the frequent name and let it match.
    names: List[str] = []
    for n in sorted(candidates, key=lambda c: -name_counts[c]):
        if any(n != kept and n.startswith(kept) for kept in names):
            continue
        names.append(n)
    names.sort(key=len, reverse=True)  # match longest remaining name first

    turns: List[dict] = []
    for seg in segments:
        speaker = None
        for nm in names:
            if seg.startswith(nm):
                speaker, speech = nm, seg[len(nm):].strip()
                break
        if speaker is None:
            # No known name -> continuation of the previous speaker.
            if turns:
                turns[-1]["text"] += " " + seg
            continue
        if not speech:
            continue
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += " " + speech
        else:
            turns.append({"speaker": speaker, "text": speech, "source": source})
    return turns


def parse_transcript_text(text: str, source: str = "") -> List[dict]:
    """Parse raw transcript text into ordered turns.

    Returns a list of ``{"speaker", "text", "source"}`` dicts in reading order.
    A label is accepted as a speaker only if it recurs (>= 2 lines), which
    prevents prose like "Note: ..." from being read as a speaker. If no label
    recurs, every matched label is kept (handles short transcripts).
    """
    # VTT / Teams / Zoom cue format takes precedence when detected.
    if _VTT_TS.search(text):
        vtt = parse_vtt_transcript(text, source=source)
        if vtt:
            return vtt

    lines = text.splitlines()

    # First pass: count candidate speaker labels.
    label_counts: Counter = Counter()
    for line in lines:
        m = _SPEAKER_LINE.match(line)
        if m:
            label_counts[m.group("speaker").strip()] += 1

    if not label_counts:
        # No speaker structure at all -> treat the whole thing as one speaker.
        joined = " ".join(l.strip() for l in lines if l.strip())
        return [{"speaker": "Speaker", "text": joined, "source": source}] if joined else []

    recurring = {lbl for lbl, c in label_counts.items() if c >= 2}
    valid_labels = recurring or set(label_counts)

    turns: List[dict] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        m = _SPEAKER_LINE.match(line)
        if m and m.group("speaker").strip() in valid_labels:
            turns.append(
                {
                    "speaker": m.group("speaker").strip(),
                    "text": m.group("text").strip(),
                    "source": source,
                }
            )
        elif turns:
            # Continuation of the previous speaker's turn.
            turns[-1]["text"] += " " + stripped
        # else: leading noise before the first speaker -> ignore.
    return turns


def load_transcripts(path: Path) -> List[dict]:
    """Load turns from a file or from every transcript file in a directory."""
    exts = {".txt", ".vtt", ".md", ".text", ".docx"}
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
        if not files:
            raise SystemExit(f"No transcript files ({', '.join(sorted(exts))}) found in {path}")
    elif path.is_file():
        files = [path]
    else:
        raise SystemExit(f"Input path does not exist: {path}")

    turns: List[dict] = []
    for f in files:
        if f.suffix.lower() == ".docx":
            text = read_docx(f)
        else:
            text = f.read_text(encoding="utf-8", errors="replace")
        # Strip a WEBVTT header line if present.
        text = re.sub(r"^WEBVTT.*?\n", "", text, flags=re.IGNORECASE)
        turns.extend(parse_transcript_text(text, source=f.name))
    if not turns:
        raise SystemExit("No speaker turns could be parsed from the input.")
    return turns


def load_transcript_batch(directory: Path, excluded_speakers: set) -> List[dict]:
    """Dynamically load every transcript in ``directory``, drop lines spoken by
    ``excluded_speakers``, and fold whichever speaker remains in each file into
    the single canonical ``SINGLE_USER_LABEL`` so the whole batch is analysed
    as one aggregated participant rather than compared candidate vs. candidate.

    Each returned turn keeps ``source`` (the originating filename) and
    ``participant`` (that file's real speaker name, before relabeling) so a
    per-transcript breakdown stays possible even though ``speaker`` is unified.
    """
    exts = {".txt", ".vtt", ".md", ".text", ".docx"}
    if not directory.is_dir():
        raise SystemExit(f"Batch input directory not found: {directory}")

    # --- Directory iteration: discover every transcript file in the folder --- #
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"No transcript files ({', '.join(sorted(exts))}) found in {directory}")

    combined_turns: List[dict] = []
    for f in files:
        text = read_docx(f) if f.suffix.lower() == ".docx" else f.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"^WEBVTT.*?\n", "", text, flags=re.IGNORECASE)
        raw_turns = parse_transcript_text(text, source=f.name)

        # --- Speaker filtering: drop every line spoken by the excluded speaker(s) --- #
        kept = [t for t in raw_turns if _normalize_speaker(t["speaker"]) not in excluded_speakers]

        remaining_names = sorted({t["speaker"] for t in kept})
        if len(remaining_names) != 1:
            found = "no speakers" if not remaining_names else f"{len(remaining_names)} speakers ({', '.join(remaining_names)})"
            print(f"  ! {f.name}: expected exactly one remaining participant after exclusion, found {found}")

        # --- Single-user relabel: fold whichever speaker(s) remain into the
        # canonical SINGLE_USER_LABEL, so downstream analysis treats this file's
        # (and every other file's) participant as one aggregated user. --- #
        for t in kept:
            t["participant"] = t["speaker"]
            t["speaker"] = SINGLE_USER_LABEL
        combined_turns.extend(kept)

    if not combined_turns:
        raise SystemExit(
            f"No speaker turns remained in {directory} after excluding "
            f"{', '.join(sorted(excluded_speakers)) or '(none)'}."
        )
    return combined_turns


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z][a-z'\-]+")


def tokenize(text: str, min_len: int, stopwords: set) -> List[str]:
    """Lowercase, split into word tokens, drop stopwords and short tokens."""
    out = []
    for tok in _WORD.findall(text.lower()):
        tok = tok.strip("'-")
        if len(tok) < min_len:
            continue
        if tok in stopwords:
            continue
        out.append(tok)
    return out


def build_plural_map(vocab: Sequence[str]) -> Dict[str, str]:
    """Map a plural form to its singular when both share a stem in the vocab.

    Conservative: only collapses ``word`` + ``words`` and ``word`` + ``wordes``
    when the singular also appears in the vocabulary. Avoids aggressive stemming.
    """
    vocab_set = set(vocab)
    mapping: Dict[str, str] = {}
    for w in vocab:
        if w.endswith("s") and not w.endswith("ss"):
            singular = w[:-1]
            if singular in vocab_set and len(singular) >= 3:
                mapping[w] = singular
            elif w.endswith("es") and w[:-2] in vocab_set and len(w) - 2 >= 3:
                mapping[w] = w[:-2]
    return mapping


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
class Analysis:
    """Container for all computed statistics."""

    def __init__(self) -> None:
        self.turns: List[dict] = []
        self.speakers: List[str] = []
        self.tokens_by_speaker: Dict[str, List[str]] = {}
        self.tokens_by_turn: List[List[str]] = []
        self.turn_speaker: List[str] = []
        self.overall_counts: Counter = Counter()
        self.counts_by_speaker: Dict[str, Counter] = {}
        self.word_counts_by_speaker: Dict[str, int] = {}
        self.turn_counts_by_speaker: Dict[str, int] = {}
        self.tfidf_by_speaker: Dict[str, List[Tuple[str, float]]] = {}
        self.bigrams: Counter = Counter()
        self.trigrams: Counter = Counter()
        self.theme_totals: Dict[str, int] = {}
        self.theme_by_speaker: Dict[str, Dict[str, int]] = {}
        self.theme_timeline: Dict[str, List[float]] = {}
        self.timeline_bins: int = 0
        self.themes: Dict[str, List[str]] = {}


def ngrams(tokens: Sequence[str], n: int) -> List[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_tfidf(tokens_by_speaker: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, float]]]:
    """Smoothed TF-IDF treating each speaker as one document.

    Surfaces words that are characteristic of a speaker relative to the others.
    """
    docs = list(tokens_by_speaker.keys())
    n_docs = len(docs)
    df: Counter = Counter()
    tf: Dict[str, Counter] = {}
    for sp, toks in tokens_by_speaker.items():
        c = Counter(toks)
        tf[sp] = c
        for w in c:
            df[w] += 1

    result: Dict[str, List[Tuple[str, float]]] = {}
    for sp in docs:
        total = sum(tf[sp].values()) or 1
        scored = []
        for w, freq in tf[sp].items():
            tf_val = freq / total
            idf = math.log((1 + n_docs) / (1 + df[w])) + 1.0
            scored.append((w, tf_val * idf))
        scored.sort(key=lambda x: x[1], reverse=True)
        result[sp] = scored
    return result


def match_themes(tokens: Sequence[str], text_lower: str, themes: Dict[str, List[str]]) -> Dict[str, int]:
    """Count theme hits for one turn.

    Single-word theme terms match against the token set (whole words);
    multi-word phrases match as substrings against the lowercased text.
    """
    token_set = Counter(tokens)
    hits: Dict[str, int] = {}
    for theme, terms in themes.items():
        count = 0
        for term in terms:
            term = term.lower()
            if " " in term:
                count += text_lower.count(term)
            else:
                count += token_set.get(term, 0)
        hits[theme] = count
    return hits


def analyze(
    turns: List[dict],
    themes: Dict[str, List[str]],
    stopwords: set,
    min_word_len: int,
    merge_plurals: bool,
    timeline_bins: int,
) -> Analysis:
    a = Analysis()
    a.turns = turns
    a.themes = themes

    # Stable speaker order = order of first appearance.
    seen: List[str] = []
    for t in turns:
        if t["speaker"] not in seen:
            seen.append(t["speaker"])
    a.speakers = seen

    # Speaker names should not be counted as conversation keywords.
    stopwords = set(stopwords)
    for sp in a.speakers:
        for tok in re.findall(r"[a-z]+", sp.lower()):
            if len(tok) >= 2:
                stopwords.add(tok)

    a.tokens_by_speaker = {sp: [] for sp in a.speakers}
    a.word_counts_by_speaker = {sp: 0 for sp in a.speakers}
    a.turn_counts_by_speaker = {sp: 0 for sp in a.speakers}

    for t in turns:
        toks = tokenize(t["text"], min_word_len, stopwords)
        a.tokens_by_turn.append(toks)
        a.turn_speaker.append(t["speaker"])
        a.tokens_by_speaker[t["speaker"]].extend(toks)
        a.word_counts_by_speaker[t["speaker"]] += len(t["text"].split())
        a.turn_counts_by_speaker[t["speaker"]] += 1

    # Optional plural merge across the full vocabulary.
    if merge_plurals:
        full_vocab = [w for toks in a.tokens_by_turn for w in toks]
        pmap = build_plural_map(full_vocab)
        if pmap:
            a.tokens_by_turn = [[pmap.get(w, w) for w in toks] for toks in a.tokens_by_turn]
            for sp in a.speakers:
                a.tokens_by_speaker[sp] = [pmap.get(w, w) for w in a.tokens_by_speaker[sp]]

    # Counts
    for sp in a.speakers:
        c = Counter(a.tokens_by_speaker[sp])
        a.counts_by_speaker[sp] = c
        a.overall_counts.update(c)

    # TF-IDF (only meaningful with 2+ speakers, but computed regardless)
    a.tfidf_by_speaker = compute_tfidf(a.tokens_by_speaker)

    # Phrases
    for toks in a.tokens_by_turn:
        a.bigrams.update(ngrams(toks, 2))
        a.trigrams.update(ngrams(toks, 3))

    # Themes: totals and per-speaker
    a.theme_totals = {th: 0 for th in themes}
    a.theme_by_speaker = {sp: {th: 0 for th in themes} for sp in a.speakers}
    per_turn_theme_hits: List[Dict[str, int]] = []
    for toks, t in zip(a.tokens_by_turn, turns):
        hits = match_themes(toks, t["text"].lower(), themes)
        per_turn_theme_hits.append(hits)
        for th, n in hits.items():
            a.theme_totals[th] += n
            a.theme_by_speaker[t["speaker"]][th] += n

    # Theme timeline across conversation progress.
    n_turns = len(turns)
    bins = max(1, min(timeline_bins, n_turns))
    a.timeline_bins = bins
    a.theme_timeline = {th: [0.0] * bins for th in themes}
    for i, hits in enumerate(per_turn_theme_hits):
        b = min(bins - 1, int(i * bins / n_turns))
        for th, n in hits.items():
            a.theme_timeline[th][b] += n

    return a


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _speaker_colors(speakers: Sequence[str]) -> Dict[str, str]:
    """Assign a fixed categorical color per speaker (stable by order)."""
    return {sp: CATEGORICAL[i % len(CATEGORICAL)] for i, sp in enumerate(speakers)}


def chart_top_keywords(a: Analysis, top: int, out: Path) -> Optional[Path]:
    items = a.overall_counts.most_common(top)
    if not items:
        return None
    words = [w for w, _ in items][::-1]
    vals = [c for _, c in items][::-1]
    colors = sequential_colors(len(words))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.38 * len(words) + 1)))
    bars = ax.barh(words, vals, color=colors, edgecolor=INK["surface"], linewidth=1.2)
    ax.set_title(f"Top {len(words)} keywords (whole conversation)")
    ax.set_xlabel("Mentions")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.12)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", ha="left", color=INK["secondary"], fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_speaker_comparison(a: Analysis, top: int, out: Path) -> Optional[Path]:
    if len(a.speakers) < 2:
        return None
    # Top keywords across the whole conversation, compared per speaker.
    top_words = [w for w, _ in a.overall_counts.most_common(top)]
    if not top_words:
        return None
    top_words = top_words[::-1]
    colors = _speaker_colors(a.speakers)
    n_sp = len(a.speakers)
    y = np.arange(len(top_words))
    height = 0.8 / n_sp
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.5 * len(top_words) + 1)))
    for i, sp in enumerate(a.speakers):
        vals = [a.counts_by_speaker[sp].get(w, 0) for w in top_words]
        offset = (i - (n_sp - 1) / 2) * height
        ax.barh(y + offset, vals, height=height, label=sp,
                color=colors[sp], edgecolor=INK["surface"], linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top_words)
    ax.set_title(f"Who says what: top {len(top_words)} keywords by speaker")
    ax.set_xlabel("Mentions")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_distinctive_words(a: Analysis, top: int, out: Path) -> Optional[Path]:
    if len(a.speakers) < 2:
        return None
    colors = _speaker_colors(a.speakers)
    n_sp = len(a.speakers)
    fig, axes = plt.subplots(1, n_sp, figsize=(5.2 * n_sp, max(4, 0.34 * top + 1.2)), squeeze=False)
    per = max(5, top // 1)
    for ax, sp in zip(axes[0], a.speakers):
        scored = [(w, s) for w, s in a.tfidf_by_speaker.get(sp, []) if a.counts_by_speaker[sp].get(w, 0) >= 2]
        scored = scored[:per][::-1]
        if not scored:
            ax.set_visible(False)
            continue
        words = [w for w, _ in scored]
        vals = [s for _, s in scored]
        ax.barh(words, vals, color=colors[sp], edgecolor=INK["surface"], linewidth=1.0)
        ax.set_title(f"{sp}: distinctive words", fontsize=12)
        ax.set_xlabel("TF-IDF weight")
        ax.grid(axis="y", visible=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Words that set each speaker apart", fontsize=14, fontweight="bold")
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_theme_distribution(
    a: Analysis, out: Path, theme_colors: Optional[Dict[str, str]] = None
) -> Optional[Path]:
    items = [(th, n) for th, n in a.theme_totals.items() if n > 0]
    if not items:
        return None
    items.sort(key=lambda x: x[1])
    names = [t for t, _ in items]
    vals = [n for _, n in items]
    # Reuse the fixed per-theme palette when given, so this chart's colors match
    # the qualitative-coding figures for the same theme names.
    colors = [theme_colors[n] for n in names] if theme_colors else sequential_colors(len(names))
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.5 * len(names) + 1)))
    bars = ax.barh(names, vals, color=colors, edgecolor=INK["surface"], linewidth=1.2)
    ax.set_title("Theme distribution (keyword hits across the conversation)")
    ax.set_xlabel("Keyword hits")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.12)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", ha="left", color=INK["secondary"], fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_theme_heatmap(a: Analysis, out: Path) -> Optional[Path]:
    active_themes = [th for th in a.themes if a.theme_totals.get(th, 0) > 0]
    if len(a.speakers) < 2 or not active_themes:
        return None
    mat = np.array([[a.theme_by_speaker[sp].get(th, 0) for th in active_themes] for sp in a.speakers],
                   dtype=float)
    # Normalise each speaker's row to a share so emphasis is comparable.
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    share = mat / row_sums

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seqblue", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(active_themes) + 2), 1.1 * len(a.speakers) + 2))
    im = ax.imshow(share, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(active_themes)))
    ax.set_xticklabels(active_themes, rotation=35, ha="right")
    ax.set_yticks(range(len(a.speakers)))
    ax.set_yticklabels(a.speakers)
    ax.set_title("Theme emphasis by speaker (row-normalised share)")
    ax.grid(False)
    thresh = share.max() * 0.55 if share.max() else 0.5
    for i in range(len(a.speakers)):
        for j in range(len(active_themes)):
            ax.text(j, i, f"{share[i, j]*100:.0f}%", ha="center", va="center",
                    fontsize=9, color="white" if share[i, j] > thresh else INK["primary"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Share of speaker's theme hits", color=INK["secondary"])
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_theme_timeline(a: Analysis, out: Path, max_lines: int = 6) -> Optional[Path]:
    # Show the most prominent themes only, to keep the chart legible.
    ranked = sorted(a.theme_totals.items(), key=lambda x: x[1], reverse=True)
    ranked = [(t, n) for t, n in ranked if n > 0][:max_lines]
    if not ranked or a.timeline_bins < 2:
        return None
    x = np.arange(a.timeline_bins)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (th, _) in enumerate(ranked):
        ax.plot(x, a.theme_timeline[th], marker="o", markersize=4, linewidth=2,
                color=CATEGORICAL[i % len(CATEGORICAL)], label=th)
    ax.set_title("How themes rise and fall across the conversation")
    ax.set_xlabel("Conversation progress  (start → end)")
    ax.set_ylabel("Keyword hits")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(100*(i)/a.timeline_bins)}–{int(100*(i+1)/a.timeline_bins)}%"
                        for i in x], rotation=0, fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=min(3, len(ranked)), loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_talk_share(a: Analysis, out: Path) -> Optional[Path]:
    if len(a.speakers) < 2:
        return None
    colors = _speaker_colors(a.speakers)
    words = [a.word_counts_by_speaker[sp] for sp in a.speakers]
    turns = [a.turn_counts_by_speaker[sp] for sp in a.speakers]
    total_w = sum(words) or 1
    total_t = sum(turns) or 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, max(3.0, 0.7 * len(a.speakers) + 2.0)))
    for ax, data, total, title in (
        (ax1, words, total_w, "Share of words spoken"),
        (ax2, turns, total_t, "Share of turns taken"),
    ):
        left = 0.0
        for sp in a.speakers:
            val = (words if data is words else turns)[a.speakers.index(sp)]
            frac = 100 * val / total
            ax.barh(0, frac, left=left, color=colors[sp], edgecolor=INK["surface"],
                    linewidth=1.5, label=sp)
            if frac > 6:
                ax.text(left + frac / 2, 0, f"{sp}\n{frac:.0f}%", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
            left += frac
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("Percent")
        ax.grid(False)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Talk share per speaker", fontsize=14, fontweight="bold", y=1.0)
    fig.subplots_adjust(top=0.74, wspace=0.15)
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_wordclouds(a: Analysis, out_dir: Path) -> List[Path]:
    if not HAVE_WORDCLOUD:
        return []
    made: List[Path] = []

    def render(counts: Counter, title: str, path: Path) -> None:
        if not counts:
            return
        wc = WordCloud(width=1000, height=560, background_color=INK["surface"],
                       colormap="viridis", prefer_horizontal=0.95, max_words=120)
        wc.generate_from_frequencies(dict(counts))
        fig, ax = plt.subplots(figsize=(10, 5.6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title)
        fig.savefig(path)
        plt.close(fig)

    render(a.overall_counts, "Word cloud — whole conversation", out_dir / "wordcloud_overall.png")
    made.append(out_dir / "wordcloud_overall.png")
    # A single aggregated speaker's word cloud is identical to the overall one
    # above, so only render per-speaker clouds when there is a real comparison.
    if len(a.speakers) > 1:
        for sp in a.speakers:
            p = out_dir / f"wordcloud_{_safe(sp)}.png"
            render(a.counts_by_speaker[sp], f"Word cloud — {sp}", p)
            if p.exists():
                made.append(p)
    return made


def chart_cooccurrence(a: Analysis, out: Path, top_nodes: int = 25, min_edge: int = 2) -> Optional[Path]:
    if not HAVE_NETWORKX:
        return None
    top_words = [w for w, _ in a.overall_counts.most_common(top_nodes)]
    if len(top_words) < 3:
        return None
    word_set = set(top_words)
    edges: Counter = Counter()
    for toks in a.tokens_by_turn:
        present = sorted(word_set.intersection(toks))
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                edges[(present[i], present[j])] += 1

    G = nx.Graph()
    for w in top_words:
        G.add_node(w, freq=a.overall_counts[w])
    for (u, v), wt in edges.items():
        if wt >= min_edge:
            G.add_edge(u, v, weight=wt)
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    if G.number_of_edges() == 0:
        return None

    pos = nx.spring_layout(G, k=0.7, seed=42, weight="weight")
    freqs = np.array([G.nodes[n]["freq"] for n in G.nodes], dtype=float)
    sizes = 250 + 1400 * (freqs - freqs.min()) / (np.ptp(freqs) or 1)
    weights = np.array([G[u][v]["weight"] for u, v in G.edges], dtype=float)
    widths = 0.6 + 3.5 * (weights - weights.min()) / (np.ptp(weights) or 1)

    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=INK["axis"], alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=CATEGORICAL[0],
                           edgecolors=INK["surface"], linewidths=1.5, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color=INK["primary"])
    ax.set_title("Keyword co-occurrence network (words used together in the same turn)")
    ax.axis("off")
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "speaker"


def summarize_by_source(turns: List[dict]) -> List[dict]:
    """Per-transcript stats keyed by ``source``, independent of the single-user
    relabeling done by ``load_transcript_batch``, so the aggregate report can
    still show what each original transcript contributed.
    """
    by_source: Dict[str, dict] = {}
    for t in turns:
        src = t.get("source", "")
        rec = by_source.setdefault(
            src, {"source": src, "participant": t.get("participant", t["speaker"]), "turns": 0, "words": 0}
        )
        rec["turns"] += 1
        rec["words"] += len(t["text"].split())
    return list(by_source.values())


def write_csv_transcripts(summaries: List[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "participant", "turns", "words"])
        for s in summaries:
            w.writerow([s["source"], s["participant"], s["turns"], s["words"]])


def write_csv_keywords(a: Analysis, top: int, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["keyword", "total"] + [f"count_{sp}" for sp in a.speakers])
        for word, total in a.overall_counts.most_common(top):
            w.writerow([word, total] + [a.counts_by_speaker[sp].get(word, 0) for sp in a.speakers])


def write_csv_themes(a: Analysis, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["theme", "total"] + [f"hits_{sp}" for sp in a.speakers])
        for th in sorted(a.theme_totals, key=lambda t: a.theme_totals[t], reverse=True):
            w.writerow([th, a.theme_totals[th]] + [a.theme_by_speaker[sp].get(th, 0) for sp in a.speakers])


def write_report(
    a: Analysis,
    top: int,
    charts: List[Path],
    out_dir: Path,
    transcript_summaries: Optional[List[dict]] = None,
    excluded_speakers: Optional[Sequence[str]] = None,
) -> Path:
    lines: List[str] = []
    lines.append("# Thematic analysis report\n")
    total_words = sum(a.word_counts_by_speaker.values())
    lines.append(f"- Speakers: {', '.join(a.speakers)}")
    lines.append(f"- Turns: {len(a.turns)}")
    lines.append(f"- Words spoken (approx.): {total_words}")
    lines.append(f"- Unique keywords (after stopword removal): {len(a.overall_counts)}")
    if excluded_speakers:
        lines.append(f"- Excluded speaker(s): {', '.join(excluded_speakers)}")
    lines.append("")

    if transcript_summaries:
        lines.append(f"## Transcripts included ({len(transcript_summaries)} aggregated)\n")
        for s in transcript_summaries:
            lines.append(f"- **{s['source']}** — {s['participant']}: {s['words']} words, {s['turns']} turns")
        lines.append("")

    lines.append("## Talk share\n")
    tw = total_words or 1
    for sp in a.speakers:
        lines.append(f"- **{sp}**: {a.word_counts_by_speaker[sp]} words "
                     f"({100*a.word_counts_by_speaker[sp]/tw:.0f}%), "
                     f"{a.turn_counts_by_speaker[sp]} turns")
    lines.append("")

    lines.append(f"## Top {top} keywords\n")
    for word, c in a.overall_counts.most_common(top):
        per = ", ".join(f"{sp} {a.counts_by_speaker[sp].get(word, 0)}" for sp in a.speakers)
        lines.append(f"- **{word}** — {c}  ({per})")
    lines.append("")

    if len(a.speakers) >= 2:
        lines.append("## Distinctive words per speaker (TF-IDF)\n")
        for sp in a.speakers:
            distinctive = [w for w, s in a.tfidf_by_speaker.get(sp, [])
                           if a.counts_by_speaker[sp].get(w, 0) >= 2][:12]
            lines.append(f"- **{sp}**: {', '.join(distinctive) if distinctive else '(none)'}")
        lines.append("")

    lines.append("## Top phrases\n")
    lines.append("**Bigrams:** " + ", ".join(f"{p} ({c})" for p, c in a.bigrams.most_common(12)))
    lines.append("")
    lines.append("**Trigrams:** " + ", ".join(f"{p} ({c})" for p, c in a.trigrams.most_common(8)))
    lines.append("")

    lines.append("## Themes\n")
    for th in sorted(a.theme_totals, key=lambda t: a.theme_totals[t], reverse=True):
        if a.theme_totals[th] == 0:
            continue
        per = ", ".join(f"{sp} {a.theme_by_speaker[sp].get(th, 0)}" for sp in a.speakers)
        lines.append(f"- **{th}** — {a.theme_totals[th]} hits  ({per})")
    empty = [th for th in a.theme_totals if a.theme_totals[th] == 0]
    if empty:
        lines.append(f"\n_Themes with no hits: {', '.join(empty)}_")
    lines.append("")

    if charts:
        lines.append("## Charts\n")
        for c in charts:
            rel = c.name
            lines.append(f"### {rel}\n\n![{rel}]({rel})\n")

    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Sample data (for --demo)
# --------------------------------------------------------------------------- #
SAMPLE_TRANSCRIPT = """[00:00:02] Alex: Honestly, work has been crushing me lately. The deadlines keep piling up and my manager just adds more projects.
[00:00:15] Jordan: That sounds really stressful. Are you getting any sleep? You mentioned last week you were exhausted.
[00:00:24] Alex: Barely. I'm so tired most mornings. The stress from the office follows me home and I can't switch off.
[00:00:37] Jordan: Your health matters more than any deadline. Have you thought about talking to your manager about the workload?
[00:00:49] Alex: I keep meaning to, but I'm worried it looks bad before the promotion review. The money would really help our family budget.
[00:01:02] Jordan: I understand the financial pressure, but burnout will cost you more in the long run. How are the kids handling your long hours?
[00:01:16] Alex: They miss me. My partner has been amazing, picking up everything at home while I'm stuck at the office late.
[00:01:29] Jordan: Family is everything. Maybe plan a proper break? Even a weekend away could reset your energy.
[00:01:40] Alex: We've talked about a trip. The savings are tight though with the mortgage and rising rent on the old place.
[00:01:53] Jordan: Money worries feed the anxiety, I get it. Have you considered whether this job is the right fit for your future goals?
[00:02:05] Alex: That's the big question. Part of me dreams of leaving to start my own small business, something in technology.
[00:02:18] Jordan: A startup! That's exciting. You've always been good with software and building apps. What's holding you back?
[00:02:31] Alex: Fear, mostly. The steady salary, the health insurance, the fear of failing my family if the business flops.
[00:02:44] Jordan: Those fears are valid, but staying somewhere that wrecks your health and sleep has a cost too. What would make you feel secure?
[00:02:58] Alex: Honestly, a financial cushion. If we saved enough to cover a year of bills, I'd feel brave enough to try.
[00:03:10] Jordan: So a concrete savings goal could turn the dream into a plan. That changes the whole conversation.
[00:03:22] Alex: You're right. Framing it as a plan instead of a leap makes the anxiety smaller. I feel a little lighter already.
[00:03:35] Jordan: Start small. Track the budget, set the savings target, and talk to your partner about the timeline. Protect your sleep along the way.
[00:03:48] Alex: Thank you. I needed to hear that. I'll bring it up with my family this weekend and map out the goals.
[00:04:00] Jordan: Anytime. Your health, your family, and your future are worth planning for. The job is just one piece.
[00:04:12] Alex: Agreed. Less fear, more planning. Maybe this stress was pushing me toward a change I actually want.
[00:04:25] Jordan: Exactly. Turn the worry into a roadmap. And please, get some rest tonight before you think about any of it.
"""

SAMPLE_TRANSCRIPT_2 = """Alex: The new software rollout at work went surprisingly well this week. The team really came together.
Jordan: That's great news. Did the long hours ease up at all, or is the office still draining your energy?
Alex: A bit better. I actually slept eight hours two nights in a row. My partner noticed I was less stressed.
Jordan: Sleep makes everything easier. How's the savings goal for the business idea coming along?
Alex: We opened a separate account and started putting money aside every month. The budget feels tighter but hopeful.
Jordan: That's real progress toward the future you described. Did you talk to your manager about the workload?
Alex: I did. We agreed to drop two projects. It's a small win but it protects my health and my time with the kids.
Jordan: Huge win, honestly. Protecting family time and cutting the stress is exactly what you needed.
Alex: The dream of the startup feels less like a fantasy now and more like a plan with real numbers and goals.
Jordan: You sound calmer. Less fear, more clarity. Keep tracking the money and guarding your sleep.
Alex: I will. Thanks for pushing me to think about my wellbeing and not just the salary and the deadlines.
Jordan: That's what friends are for. Your health and your family come first, the career will follow.
"""


def write_sample_data(transcripts_dir: Path) -> None:
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    (transcripts_dir / "sample_conversation_1.txt").write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    (transcripts_dir / "sample_conversation_2.txt").write_text(SAMPLE_TRANSCRIPT_2, encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_themes(path: Optional[Path]) -> Dict[str, List[str]]:
    if path is None:
        return DEFAULT_THEMES
    if not path.exists():
        raise SystemExit(f"Themes file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Themes JSON must be an object of {theme: [keywords]}.")
    return {str(k): [str(x) for x in v] for k, v in data.items()}


def load_extra_stopwords(value: Optional[str]) -> set:
    if not value:
        return set()
    p = Path(value)
    if p.exists():
        return {w.strip().lower() for w in p.read_text(encoding="utf-8").split() if w.strip()}
    return {w.strip().lower() for w in value.split(",") if w.strip()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch thematic analysis of interview transcripts, aggregated to a single participant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", default=DEFAULT_BATCH_DIR,
                   help="Directory of transcripts to batch-load (.txt/.vtt/.md/.docx).")
    p.add_argument("-o", "--output", default="output", help="Directory for charts and reports.")
    p.add_argument("-t", "--themes", default=str(Path(__file__).parent / "themes.qualitative.json"),
                   help="Path to a themes JSON file (defaults to the bundled 6 qualitative-coding themes).")
    p.add_argument("--top", type=int, default=20, help="How many top keywords to chart/report.")
    p.add_argument("--min-word-len", type=int, default=3, help="Ignore tokens shorter than this.")
    p.add_argument("--extra-stopwords", default=None,
                   help="Comma-separated words or a path to a file of extra stopwords.")
    p.add_argument("--timeline-bins", type=int, default=12,
                   help="Number of segments for the theme-over-time chart.")
    p.add_argument("--no-merge-plurals", action="store_true",
                   help="Do not merge simple plural/singular forms.")
    p.add_argument("--exclude-speaker", action="append", default=None,
                   help="Speaker to drop from every transcript before analysis (repeatable). "
                        f"Default: {DEFAULT_EXCLUDED_SPEAKERS[0]!r}. Pass an empty string "
                        "to disable exclusion/aggregation and compare all speakers directly.")
    p.add_argument("--demo", action="store_true",
                   help="Write sample transcripts into the input directory, then analyze them.")
    return p


def run_keyword_pipeline(
    input_path: Path,
    out_dir: Path,
    themes: Dict[str, List[str]],
    stopwords: set,
    excluded_speakers: set,
    excluded_display: List[str],
    top: int = 20,
    min_word_len: int = 3,
    merge_plurals: bool = True,
    timeline_bins: int = 12,
) -> dict:
    """Run the keyword/theme-lexicon analysis end to end and write its charts,
    CSVs, and report into ``out_dir``. Shared by ``main()`` (CLI) and
    ``run_pipeline.py`` (the combined driver), so both stay in sync.

    Returns a dict with the computed ``Analysis`` and everything written, so a
    caller like run_pipeline.py can reuse this data instead of re-parsing.
    """
    apply_chart_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading transcripts from: {input_path}")
    if excluded_speakers:
        print(f"Excluding speaker(s): {', '.join(excluded_display)}")
        turns = load_transcript_batch(input_path, excluded_speakers)
    else:
        turns = load_transcripts(input_path)
    transcript_summaries = summarize_by_source(turns)
    a = analyze(
        turns, themes=themes, stopwords=stopwords, min_word_len=min_word_len,
        merge_plurals=merge_plurals, timeline_bins=timeline_bins,
    )
    print(f"Parsed {len(turns)} turns from {len(a.speakers)} speaker(s): {', '.join(a.speakers)}")

    # Use the fixed qualitative-theme palette whenever this run's themes are
    # (a subset of) the 6 qualitative-coding themes, so the keyword-hit chart
    # colors match the excerpt-coding figures for the same theme names.
    theme_colors = qualitative_theme_colors() if set(themes) <= set(QUALITATIVE_THEMES) else None

    charts: List[Path] = []
    builders = [
        ("keywords_top.png", lambda p: chart_top_keywords(a, top, p)),
        ("keywords_by_speaker.png", lambda p: chart_speaker_comparison(a, top, p)),
        ("distinctive_words.png", lambda p: chart_distinctive_words(a, min(15, top), p)),
        ("themes_distribution.png", lambda p: chart_theme_distribution(a, p, theme_colors=theme_colors)),
        ("themes_by_speaker_heatmap.png", lambda p: chart_theme_heatmap(a, p)),
        ("themes_timeline.png", lambda p: chart_theme_timeline(a, p)),
        ("talk_share.png", lambda p: chart_talk_share(a, p)),
        ("cooccurrence_network.png", lambda p: chart_cooccurrence(a, p)),
    ]
    for name, fn in builders:
        try:
            result = fn(out_dir / name)
        except Exception as e:  # keep going if one chart fails
            print(f"  ! skipped {name}: {e}")
            result = None
        if result:
            charts.append(result)
            print(f"  wrote {result}")

    charts.extend(chart_wordclouds(a, out_dir))
    for p in charts:
        if p.name.startswith("wordcloud"):
            print(f"  wrote {p}")

    if not HAVE_WORDCLOUD:
        print("  (word clouds skipped: `pip install wordcloud` to enable)")
    if not HAVE_NETWORKX:
        print("  (co-occurrence network skipped: `pip install networkx` to enable)")

    write_csv_keywords(a, top, out_dir / "keywords.csv")
    write_csv_themes(a, out_dir / "themes.csv")
    write_csv_transcripts(transcript_summaries, out_dir / "transcripts.csv")
    report = write_report(
        a, top, charts, out_dir,
        transcript_summaries=transcript_summaries,
        excluded_speakers=excluded_display,
    )
    for name in ("keywords.csv", "themes.csv", "transcripts.csv"):
        print(f"  wrote {out_dir / name}")
    print(f"  wrote {report}")
    print(f"\nDone. {len(charts)} charts + report in {out_dir}/")

    return {
        "analysis": a, "charts": charts, "transcript_summaries": transcript_summaries,
        "report": report, "themes": themes, "excluded_display": excluded_display,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    input_path = Path(args.input)
    out_dir = Path(args.output)

    # Resolve which speaker(s) to exclude. Default excludes the recurring
    # interviewer; passing --exclude-speaker "" disables exclusion entirely.
    excluded_input = DEFAULT_EXCLUDED_SPEAKERS if args.exclude_speaker is None else args.exclude_speaker
    excluded_display = [s.strip() for s in excluded_input if s.strip()]
    excluded_speakers = {_normalize_speaker(s) for s in excluded_display}

    if args.demo:
        demo_dir = input_path if input_path.suffix == "" else input_path.parent
        write_sample_data(demo_dir)
        input_path = demo_dir
        # The bundled sample data is a plain two-speaker chat with no
        # interviewer to exclude, so keep the classic comparative demo.
        excluded_display = []
        excluded_speakers = set()
        print(f"[demo] wrote sample transcripts to {demo_dir}/")

    themes = load_themes(Path(args.themes) if args.themes else None)
    stopwords = set(STOPWORDS) | load_extra_stopwords(args.extra_stopwords)

    run_keyword_pipeline(
        input_path, out_dir, themes, stopwords, excluded_speakers, excluded_display,
        top=args.top, min_word_len=args.min_word_len,
        merge_plurals=not args.no_merge_plurals, timeline_bins=args.timeline_bins,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
