# interdisciplinary_app.py
"""
Fieldtrip Index

Measures the cross-domain impact of academic research using 4 complementary metrics:
- External Diversity, Internal Diversity, Reference Diversity, and Bridge Score.

Analyses an author's 25 most-cited papers.
"""

import gradio as gr
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
import logging
import re
import math
import base64
import pathlib
import unicodedata
import urllib.parse
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from collections import Counter, defaultdict
from scipy.stats import gaussian_kde, entropy
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile
import uuid
import shutil
import httpx
from reproducible_cache import (
    is_cached, load_author_cache, build_author_cache,
    get_cache_timestamp, clear_author_cache, fetch_with_retry,
    reconstruct_abstract, fetch_author_signals, CACHE_SCHEMA,
    verify_api_key, log_api_key_status,
    OPENALEX_BASE_URL, RateLimitExceeded, NotEnoughData, Throttled, InvalidAPIKey
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MODEL_NAME = 'minishlab/potion-base-32M'
    DEVICE = "cpu"
    # 10 was too few: the bridge score fell 28-38% between 10 and 40 papers,
    # because a longer reference list leaves less of the audience unexplained.
    # External, internal and reference diversity were stable across that range.
    TOP_N_PAPERS = 25
    # 20 -> 100 citing works moves external diversity by 0.6 points, so 50 is
    # comfortably past the point where more would buy anything.
    CITATIONS_PER_PAPER = 50
    REFERENCES_PER_PAPER = 50
    KDE_BANDWIDTH = 0.2
    REQUEST_TIMEOUT = 30

    SEED = 42
    EXCLUDE_SELF_CITATIONS = True

    # Level of OpenAlex's topic hierarchy used as "field".
    # subfield ~252 buckets, field ~26, domain 4. Falls back up the chain.
    FIELD_LEVELS = ("subfield", "field", "domain")
    FIELD_LEVEL = "field"

    # Bridge score: a field is named as bridged once the citing work it sends
    # exceeds the author's own citing of it by this share of the total.
    BRIDGE_MIN_SHARE = 0.02

# ============================================================================
# LOGGING & MODEL SETUP
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Loading model: {Config.MODEL_NAME}")
sentence_model = SentenceTransformer(Config.MODEL_NAME, device=Config.DEVICE)
kw_model = KeyBERT(sentence_model)
logger.info("Models loaded successfully")

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

SESSION_DIR_PREFIX = "interdisciplinary_"


def create_session_cache_dir() -> str:
    session_id = str(uuid.uuid4())[:8]
    cache_dir = os.path.join(tempfile.gettempdir(), f"{SESSION_DIR_PREFIX}{session_id}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def cleanup_session_cache(cache_dir: str):
    if cache_dir and os.path.exists(cache_dir) and cache_dir.startswith(tempfile.gettempdir()):
        try:
            shutil.rmtree(cache_dir)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


def sweep_stale_session_caches(max_age_hours: float = 24.0) -> int:
    """Remove session cache directories left behind by earlier runs.

    Gradio's unload hook takes no arguments, so it cannot be told which session
    directory to remove. Sweeping by age at startup is the reliable alternative:
    only directories this app created, and only ones untouched for a day.
    """
    root = tempfile.gettempdir()
    cutoff = datetime.now().timestamp() - max_age_hours * 3600
    removed = 0
    try:
        entries = os.listdir(root)
    except OSError:
        return 0
    for name in entries:
        if not name.startswith(SESSION_DIR_PREFIX):
            continue
        path = os.path.join(root, name)
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path)
                removed += 1
        except OSError as e:
            logger.warning(f"Could not remove stale cache {name}: {e}")
    if removed:
        logger.info(f"Removed {removed} stale session cache director{'y' if removed == 1 else 'ies'}")
    return removed

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def progress_block(fraction: float, message: str) -> str:
    """A progress bar rendered in the app's own styling.

    Gradio draws its own indicator on every output component of an event, which
    is why a search showed two of them, and its bar reports elapsed seconds
    rather than the step. Yielding this from the handler puts both the
    appearance and the wording under the app's control.
    """
    pct = max(0.0, min(1.0, fraction)) * 100
    return (
        '<div class="progress-box">'
        '<div class="progress-spinner"></div>'
        '<div class="progress-body">'
        f'<div class="progress-text">{message}</div>'
        f'<div class="progress-rail"><div class="progress-fill" style="width:{pct:.0f}%"></div></div>'
        '</div></div>'
    )


def notice(title: str, body: str, kind: str = "info") -> str:
    """A single inline message block — used for empty states and failures alike."""
    return f'<div class="notice notice-{kind}"><b>{title}.</b> {body}</div>'



def _normalise_author_query(q: str) -> tuple[str, str]:
    """Accept a bare name, an ORCID, or an OpenAlex author ID.

    Returns (mode, value) where mode is 'orcid', 'id', or 'search'.
    """
    q = q.strip()
    bare = q.rsplit("/", 1)[-1].upper()
    if bare.startswith("A") and bare[1:].isdigit():
        return "id", bare
    digits = q.replace("-", "").replace(" ", "")
    tail = digits.rsplit("/", 1)[-1]
    if len(tail) == 16 and tail[:15].isdigit() and (tail[15].isdigit() or tail[15] == "X"):
        canonical = "-".join(tail[i:i + 4] for i in range(0, 16, 4))
        return "orcid", canonical
    return "search", q


def _candidate_from_result(result: dict) -> dict:
    institutions = result.get("last_known_institutions") or []
    institution = institutions[0].get("display_name", "") if institutions else ""

    years = [c.get("year") for c in (result.get("counts_by_year") or []) if c.get("year")]
    span = f"{min(years)}–{max(years)}" if years else ""

    topics = ", ".join(
        t.get("display_name", "") for t in (result.get("topics") or [])[:2] if t.get("display_name")
    )

    orcid = result.get("orcid") or ""
    if orcid:
        orcid = orcid.rsplit("/", 1)[-1]

    return {
        "id": result["id"].split("/")[-1],
        "name": result.get("display_name") or "Unknown",
        "works_count": result.get("works_count", 0),
        "cited_by_count": result.get("cited_by_count", 0),
        "institution": institution,
        "orcid": orcid,
        "span": span,
        "topics": topics,
    }


async def search_authors(author_query: str, limit: int = 6) -> list[dict]:
    """Find candidate authors. Accepts a name, an ORCID, or an OpenAlex ID."""
    mode, value = _normalise_author_query(author_query)
    select = ("id,display_name,orcid,works_count,cited_by_count,"
              "last_known_institutions,counts_by_year,topics")

    async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
        if mode in ("orcid", "id"):
            path = f"orcid:{value}" if mode == "orcid" else value
            data = await fetch_with_retry(client, f"{OPENALEX_BASE_URL}/authors/{path}",
                                          params={"select": select})
            return [_candidate_from_result(data)] if data and data.get("id") else []

        data = await fetch_with_retry(
            client, f"{OPENALEX_BASE_URL}/authors",
            params={"search": value, "per_page": limit, "select": select},
        )
        return [_candidate_from_result(r) for r in (data or {}).get("results", []) if r.get("id")]


# ============================================================================
# PROFILE COHERENCE
# ============================================================================
# A merged author record raises all four diversity measures at once, so the
# composite peaks exactly when the data are least trustworthy. These checks are
# the only thing standing between that and a confident-looking number.
#
# Thresholds are tuned for recall, not precision: this warns a real person, so
# a false accusation costs more than a missed one. A single signal is never
# enough — only conflicting ORCIDs, which are close to proof, warn on their own.

COHERENCE_LIMITS = {
    "repeat_rate": 0.20,      # measured: ~32% median for a clean random draw
    "institutions": 35,       # measured: 11-29 for clean records, 61+ when merged
    "career_span": 60,        # years; longer than a plausible single career
    "community_overlap": 0.05,  # Jaccard between two field communities
    "community_share": 0.20,  # each community must hold this share of the work
}


NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

# Letters that carry no combining mark, so NFKD leaves them alone, but which
# are the same letter for name-matching purposes. Without this, "Abhışhek" and
# "Abhishek" read as two people — and getting that wrong would flag
# non-Anglophone researchers far more often than anyone else.
_LETTER_FOLD = str.maketrans({
    "ı": "i", "ł": "l", "ø": "o", "đ": "d", "ð": "d",
    "þ": "th", "ß": "ss", "æ": "ae", "œ": "oe", "ħ": "h", "ŧ": "t",
})


def _fold_name(text: str) -> str:
    """Lowercase, strip diacritics, and fold look-alike letters to ASCII."""
    lowered = (text or "").lower().translate(_LETTER_FOLD)
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _name_identities(raw_names: list[str]) -> list[str]:
    """How many distinct people the bylines on a record appear to name.

    Byline order is not stable — the same person appears as "Yoshua Bengio" and
    as "Bengio, Yoshua" — so this compares unordered token sets rather than
    assuming a position. Initials are absorbed into full names, including the
    run-together form ("Harrell FE" alongside "Frank E. Harrell"). Multi-letter
    given names are never treated as abbreviations of one another: "Yan" is not
    read as short for "Yang", because collapsing those would erase exactly the
    collision this is looking for.
    """
    entries = []
    for name in raw_names:
        tokens = frozenset(
            t for t in re.sub(r"[^a-z\s]", " ", _fold_name(name)).split()
            if t not in NAME_SUFFIXES
        )
        if tokens:
            entries.append((tokens, (name or "").strip()))
    if not entries:
        return []

    def compatible(a: frozenset, b: frozenset) -> bool:
        small, large = (a, b) if len(a) <= len(b) else (b, a)
        for tok in small:
            if tok in large:
                continue
            if len(tok) == 1 and any(t.startswith(tok) for t in large):
                continue
            if any(len(t) == 1 and tok.startswith(t) for t in large):
                continue
            # run-together initials: "fe" standing for {frank, e}
            if len(tok) <= 3 and all(any(t.startswith(ch) for t in large) for ch in tok):
                continue
            return False
        return True

    clusters: list[list[tuple]] = []
    for sig, original in entries:
        for cluster in clusters:
            if any(compatible(sig, member_sig) for member_sig, _ in cluster):
                cluster.append((sig, original))
                break
        else:
            clusters.append([(sig, original)])

    # label each cluster with its most complete spelling
    return [max((o for _, o in c), key=len) for c in clusters]


def _split_communities(signals: dict) -> tuple[float | None, tuple[str, str] | None]:
    """Co-author overlap between the author's two largest field communities.

    This is what separates genuine breadth from a name collision. A real
    polymath carries a recurring core of collaborators across fields; two
    different people who happen to share a name do not. Without it, wide-ranging
    work would look suspicious purely for being wide-ranging.
    """
    field_works = signals.get("field_works") or {}
    field_coauthors = signals.get("field_coauthors") or {}
    total = signals.get("works_sampled") or 0
    if total < 10 or len(field_works) < 2:
        return None, None

    ranked = sorted(field_works.items(), key=lambda x: x[1], reverse=True)[:2]
    (f_a, n_a), (f_b, n_b) = ranked
    if min(n_a, n_b) / total < COHERENCE_LIMITS["community_share"]:
        return None, None

    a = set(field_coauthors.get(f_a) or [])
    b = set(field_coauthors.get(f_b) or [])
    if not a or not b:
        return None, None
    return len(a & b) / len(a | b), (f_a, f_b)


def assess_coherence(signals: dict) -> dict:
    """Judge whether an OpenAlex author record plausibly describes one person.

    Signals are split by how much weight they can carry alone. Sparse
    co-authorship and institutional sprawl are *primary*: they are hard to
    produce accidentally. Name variation, disjoint field communities, and an
    implausible career span are *corroborating* — each has honest explanations
    (romanisation, a methodologist serving several applied fields, a long
    career), so none of them raises a warning without a primary signal beside
    it. Conflicting ORCIDs are close to proof and stand alone.
    """
    if not signals or not signals.get("works_sampled"):
        return {"verdict": "unknown", "flags": [], "stats": {}}

    lim = COHERENCE_LIMITS
    primary, corroborating = [], []

    n_co = signals.get("n_coauthors") or 0
    repeat_rate = (signals.get("n_repeat_coauthors") or 0) / n_co if n_co else None
    if repeat_rate is not None and n_co >= 20 and repeat_rate < lim["repeat_rate"]:
        primary.append(f"only {repeat_rate:.0%} of {n_co} co-authors appear on more than one paper")

    if (signals.get("n_institutions") or 0) > lim["institutions"]:
        primary.append(f"{signals['n_institutions']} different institutions across the work")

    names = _name_identities(signals.get("raw_names") or [])
    if len(names) > 1:
        # Two clusters can share their longest spelling; listing it twice reads
        # as a bug, so only distinct spellings are shown.
        distinct = list(dict.fromkeys(names))
        detail = f" ({'; '.join(distinct[:3])})" if len(distinct) > 1 else ""
        corroborating.append(
            f"bylines naming {len(names)} different people{detail}")

    y0, y1 = signals.get("year_min"), signals.get("year_max")
    if y0 and y1 and (y1 - y0) > lim["career_span"]:
        corroborating.append(f"publications spanning {y1 - y0} years ({y0}\u2013{y1})")

    overlap, pair = _split_communities(signals)
    if overlap is not None and overlap < lim["community_overlap"]:
        corroborating.append(
            f"its {pair[0]} and {pair[1]} work share almost no co-authors "
            f"({overlap:.0%} overlap)"
        )

    stats = {"repeat_rate": repeat_rate, "names": names, "community_overlap": overlap}
    orcids = signals.get("orcids") or []
    if len(orcids) > 1:
        return {"verdict": "conflated", "flags": primary + corroborating,
                "orcid_conflict": orcids, "stats": stats}

    flagged = len(primary) >= 2 or (len(primary) >= 1 and len(corroborating) >= 1)
    return {"verdict": "check" if flagged else "ok",
            "flags": (primary + corroborating) if flagged else [],
            "stats": stats}


def coherence_notice(assessment: dict) -> str:
    """Render a coherence warning, or nothing when the record looks sound."""
    verdict = assessment.get("verdict")
    if verdict in ("ok", "unknown"):
        return ""

    items = "".join(f"<li>{f}</li>" for f in assessment.get("flags", []))

    if verdict == "conflated":
        orcids = ", ".join(assessment.get("orcid_conflict", []))
        body = (
            "Two different ORCIDs appear on work filed under this single OpenAlex "
            f"record ({orcids}), which means it holds more than one person's papers. "
            "Every measure below is inflated by that: unrelated papers look like "
            "range. Treat these numbers as unusable and pick a narrower profile."
        )
    else:
        body = (
            "This OpenAlex record may hold more than one person's work. Merged "
            "records raise all four measures at once, because papers by different "
            "people read as unusual range. Worth checking the affiliation and dates "
            "before relying on the numbers."
        )

    return (f'<div class="notice notice-warn"><b>Check this profile</b>. {body}'
            + (f"<ul>{items}</ul>" if items else "") + "</div>")

# ============================================================================
# METRIC CALCULATIONS
# ============================================================================

def domain_of(topic: dict | None) -> str | None:
    """The top-level domain a topic sits in; there are four."""
    return topic.get("domain") if isinstance(topic, dict) else None


def field_of(topic: dict | None) -> str | None:
    """The field label at the configured level of OpenAlex's topic hierarchy.

    Returns None — never a placeholder — when a work carries no topic, so that
    unclassified works can be counted rather than silently dropped.
    """
    if not isinstance(topic, dict):
        return None
    for level in Config.FIELD_LEVELS[Config.FIELD_LEVELS.index(Config.FIELD_LEVEL):]:
        name = topic.get(level)
        if name:
            return name
    return None


# Disparity comes from OpenAlex's own hierarchy rather than from embedding the
# field names. Embedding the labels was tried first and barely discriminated:
# d(Medicine, Nursing) came out 0.30 against d(Medicine, Astronomy) 0.33, so the
# term was close to a constant and Rao-Stirling collapsed onto its balance
# component. The taxonomy separates them properly.
DISPARITY_SAME_DOMAIN = 0.5      # e.g. Medicine and Nursing, both Health Sciences
DISPARITY_CROSS_DOMAIN = 1.0     # e.g. Medicine and Physics and Astronomy
DISPARITY_UNKNOWN = 0.5          # a field whose domain never appeared in the data


def field_disparity(fields: list[str], domains: dict) -> np.ndarray:
    """How far apart each pair of fields is, on OpenAlex's four-domain hierarchy.

    Same field 0; different field inside one domain 0.5; different domain 1.
    """
    n = len(fields)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            di, dj = domains.get(fields[i]), domains.get(fields[j])
            if not di or not dj:
                value = DISPARITY_UNKNOWN
            elif di == dj:
                value = DISPARITY_SAME_DOMAIN
            else:
                value = DISPARITY_CROSS_DOMAIN
            d[i, j] = d[j, i] = value
    return d


def diversity_of(field_counts: dict, field_domains: dict | None = None) -> dict:
    """Rao–Stirling diversity, plus the effective number of fields.

    Rao–Stirling combines variety, balance, and disparity:
        RS = sum over i != j of  d_ij * p_i * p_j
    It sits well below 100 in practice, which is why the interface presents
    these numbers as comparative rather than as a grade.
    """
    if not field_counts:
        return {"field_counts": {}, "diversity_index": 0.0, "entropy": 0.0,
                "effective_fields": 0.0, "unique_fields": 0, "total": 0}

    fields = list(field_counts)
    counts = np.array([field_counts[f] for f in fields], dtype=float)
    p = counts / counts.sum()

    shannon = float(entropy(p, base=np.e))
    effective = float(np.exp(shannon))

    if len(fields) == 1:
        rs = 0.0
    else:
        d = field_disparity(fields, field_domains or {})
        rs = float(p @ d @ p)

    return {
        "field_counts": dict(field_counts),
        "diversity_index": min(100.0, rs * 100.0),
        "entropy": shannon,
        "effective_fields": effective,
        "unique_fields": len(fields),
        "total": int(counts.sum()),
    }


def calculate_embedding_dispersion(abstracts: list[str]) -> dict:
    """Mean pairwise cosine distance between the author's own papers."""
    if len(abstracts) < 2:
        return {"dispersion_score": 0, "embeddings": None}

    with torch.no_grad():
        embeddings = sentence_model.encode(abstracts, convert_to_tensor=True, device=Config.DEVICE)
        embeddings_np = embeddings.cpu().numpy()

    avg_distance = float(np.mean(pdist(embeddings_np, metric="cosine")))
    return {"dispersion_score": min(100.0, avg_distance * 100),
            "embeddings": embeddings_np}


def calculate_reference_diversity(papers: list[dict], reference_topics: dict) -> dict:
    """How widely the author's reference lists spread across fields.

    Counts every reference occurrence, so a work cited by two of the author's
    papers counts twice, while the field lookup itself was de-duplicated.
    """
    field_counts = defaultdict(int)
    field_domains: dict[str, str] = {}
    resolved = unresolved = 0

    for paper in papers:
        for ref_id in paper.get("referenced_works", []):
            topic = reference_topics.get(ref_id)
            field = field_of(topic)
            if field:
                field_counts[field] += 1
                domain = domain_of(topic)
                if domain:
                    field_domains.setdefault(field, domain)
                resolved += 1
            else:
                unresolved += 1

    result = diversity_of(dict(field_counts), field_domains)
    result["field_domains"] = field_domains
    result["classified"] = resolved
    result["unclassified"] = unresolved
    result["coverage"] = resolved / (resolved + unresolved) if (resolved + unresolved) else 0.0
    return result


def audience_field_counts(papers: list[dict]) -> tuple[dict, int, int]:
    """Field distribution of the works citing this author, plus coverage."""
    counts = defaultdict(int)
    resolved = unresolved = 0
    for paper in papers:
        for citing in paper.get("citing", []):
            field = field_of(citing.get("topic"))
            if field:
                counts[field] += 1
                resolved += 1
            else:
                unresolved += 1
    return dict(counts), resolved, unresolved


def calculate_bridge_score(source_counts: dict, audience_counts: dict) -> dict:
    """How much of the citing work exceeds what the author's own citing explains.

    For each field, the amount by which its share of the citing work exceeds its
    share of the reference list; summed over fields where that is positive. This
    is the total variation distance between the two distributions, taken in one
    direction only, so it is bounded 0-1 and reads as "the share of citing work
    that your own reading does not account for".

    Weighted by volume, so one citing paper from an unrelated field moves it by a
    fraction of a percent. And continuous, so a field you cite a little but that
    cites you a lot still counts in proportion — no threshold to fall the wrong
    side of.
    """
    if not source_counts or not audience_counts:
        return {"source_fields": dict(source_counts), "audience_fields": dict(audience_counts),
                "bridge_score": 0.0, "bridged_fields": [], "common_fields": []}

    src_total = sum(source_counts.values()) or 1
    aud_total = sum(audience_counts.values()) or 1
    src_share = {f: c / src_total for f, c in source_counts.items()}
    aud_share = {f: c / aud_total for f, c in audience_counts.items()}

    imbalance = {f: share - src_share.get(f, 0.0) for f, share in aud_share.items()}
    bridged_mass = sum(v for v in imbalance.values() if v > 0)

    bridged = sorted(
        (f for f, v in imbalance.items() if v >= Config.BRIDGE_MIN_SHARE),
        key=lambda f: imbalance[f], reverse=True,
    )
    common = sorted(
        (f for f in aud_share if imbalance[f] < Config.BRIDGE_MIN_SHARE),
        key=lambda f: aud_share[f], reverse=True,
    )

    return {
        "source_fields": dict(source_counts),
        "audience_fields": dict(audience_counts),
        "bridge_score": min(100.0, bridged_mass * 100.0),
        "bridged_fields": bridged,
        "common_fields": common,
        "imbalance": imbalance,
    }


def calculate_similarity_and_index(original_abstract: str, citing_abstracts: list[str]) -> tuple:
    """External diversity for one paper: 1 - mean similarity to its citing work."""
    with torch.no_grad():
        e_orig = sentence_model.encode(original_abstract, convert_to_tensor=True, device=Config.DEVICE)
        e_cite = sentence_model.encode(citing_abstracts, convert_to_tensor=True, device=Config.DEVICE)
        sims = util.cos_sim(e_orig, e_cite)[0].cpu().tolist()
        sims = [max(0.0, s) for s in sims]
        avg_sim = sum(sims) / len(sims)
        return avg_sim, 1.0 - avg_sim, len(sims), sims

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
# Palette: the interface is ink-on-paper; colour is spent only where it does
# semantic work. Two hues total — a cool/warm diverging pair for "you cite" vs
# "cites you", and a single-hue sequential ramp for the similarity matrix.
# Both come from the validated categorical slots 1 and 2 (all-pairs safe).

INK, INK_2, INK_3 = "#14161A", "#4A5058", "#868D97"
RULE, RULE_SOFT, SURFACE = "#E4E7EB", "#F1F3F5", "#FFFFFF"
COOL, WARM = "#2A78D6", "#EB6834"
SANS = "IBM Plex Sans, -apple-system, sans-serif"
MONO = "IBM Plex Mono, ui-monospace, monospace"
SEQ = [[0.0, "#F7F9FC"], [0.35, "#CBDFF6"], [0.7, "#7FB0E8"], [1.0, COOL]]
# Categorical slots 1-3 of the validated palette: the only run that clears every
# colour-vision gate with all pairs on screen at once.
COMPARE_SERIES = (COOL, WARM, "#1BAF7A")
MAX_COMPARE = 3


def get_chart_layout(title: str, height: int = 340, width: int = None) -> dict:
    """Shared chart chrome: left-aligned title, recessive axes, mono numerals."""
    layout = {
        "title": dict(
            text=title,
            font=dict(size=13, color=INK, family=SANS),
            x=0, xanchor="left", y=0.97, yanchor="top",
        ),
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": dict(family=SANS, color=INK_2, size=12),
        "margin": dict(t=46, b=42, l=60, r=26),
        "height": height,
        "autosize": True,
        "modebar": dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
        "hoverlabel": dict(
            bgcolor=SURFACE, bordercolor=RULE, font_size=12,
            font_family=MONO, font_color=INK,
        ),
    }
    if width:
        layout["width"] = width
    return layout


def _mono_ticks(size: int = 11, color: str = None) -> dict:
    return dict(family=MONO, size=size, color=color or INK_3)


def _short(label: str, limit: int = 30) -> str:
    """Field names like 'Biochemistry, Genetics and Molecular Biology' overflow
    the axis gutter; the full name stays in the hover text."""
    return label if len(label) <= limit else label[:limit - 1] + "\u2026"


def create_dispersion_chart(dispersion_data: dict, paper_titles: list[str]) -> go.Figure:
    """Pairwise cosine similarity between the analysed papers (sequential, one hue)."""
    embeddings = dispersion_data.get("embeddings")
    if embeddings is None or len(embeddings) < 2:
        return None

    sim_matrix = cosine_similarity(embeddings)
    n_papers = len(embeddings)

    hover_text = []
    for i in range(n_papers):
        row = []
        for j in range(n_papers):
            ti = paper_titles[i][:44] + "..." if len(paper_titles[i]) > 44 else paper_titles[i]
            tj = paper_titles[j][:44] + "..." if len(paper_titles[j]) > 44 else paper_titles[j]
            if i == j:
                row.append(f"P{i+1}  {ti}")
            else:
                row.append(f"P{i+1} vs P{j+1}<br>{ti}<br>{tj}<br>similarity {sim_matrix[i, j]:.2f}")
        hover_text.append(row)

    avg_sim = float(np.mean(sim_matrix[np.triu_indices(n_papers, k=1)]))
    labels = [f"P{i+1}" for i in range(n_papers)]
    # Square cells mean the plot is only as wide as it is tall, so the box has
    # to grow with the paper count or 25 papers collapse into a corner.
    side = int(min(760, max(380, 90 + 22 * n_papers)))
    # And every label cannot fit once there are more than a dozen rows.
    every = max(1, -(-n_papers // 12))
    shown = [lbl if i % every == 0 else "" for i, lbl in enumerate(labels)]

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix.tolist(), x=labels, y=labels,
        hovertext=hover_text, hovertemplate="%{hovertext}<extra></extra>",
        colorscale=SEQ, zmin=0, zmax=1, xgap=2, ygap=2,
        colorbar=dict(
            thickness=8, len=0.62, outlinewidth=0, x=1.02, xpad=0,
            tickvals=[0, 0.5, 1], ticktext=["0", ".5", "1"],
            tickfont=_mono_ticks(10),
        ),
    ))

    fig.update_layout(**get_chart_layout("Paper similarity", side))
    fig.update_layout(
        margin=dict(t=46, b=54, l=52, r=26),
        xaxis=dict(type="category", categoryorder="array", categoryarray=labels,
                   tickmode="array", tickvals=labels, ticktext=shown,
                   tickfont=_mono_ticks(10), title=""),
        yaxis=dict(type="category", categoryorder="array", categoryarray=labels,
                   autorange="reversed", scaleanchor="x", constrain="domain",
                   tickmode="array", tickvals=labels, ticktext=shown,
                   tickfont=_mono_ticks(10), title=""),
        annotations=[dict(
            x=0, y=-0.14, xref="paper", yref="paper", xanchor="left",
            text=f"Mean pairwise similarity {avg_sim:.2f} — lower means the papers sit further apart",
            showarrow=False, font=dict(size=11.5, color=INK_3, family=SANS),
        )],
    )
    return fig


def create_reference_diversity_chart(ref_diversity: dict) -> go.Figure:
    """Fields this author draws on. Cool — same entity as 'you cite' in the flow chart."""
    field_counts = ref_diversity.get("field_counts", {})
    if not field_counts:
        return None

    ordered = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:10][::-1]
    labels, values = [f[0] for f in ordered], [f[1] for f in ordered]

    fig = go.Figure(go.Bar(
        y=[_short(l) for l in labels], x=values, orientation="h",
        marker=dict(color=COOL, line=dict(width=0)),
        text=values, textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=INK, family=MONO),
        customdata=labels,
        hovertemplate="%{customdata}<br>%{x} references<extra></extra>",
    ))
    fig.update_layout(**get_chart_layout("Fields you draw on"))
    fig.update_layout(margin=dict(t=46, b=42, l=180, r=90), bargap=0.42)
    fig.update_xaxes(title="", showgrid=True, gridcolor=RULE_SOFT, zeroline=False,
                     tickfont=_mono_ticks())
    fig.update_yaxes(tickfont=dict(size=12, color=INK_2, family=SANS), automargin=True)
    return fig


def create_bridge_chart(bridge_data: dict) -> go.Figure:
    """Diverging: fields you cite (cool, left) against fields citing you (warm, right)."""
    source_fields = bridge_data.get("source_fields", {})
    audience_fields = bridge_data.get("audience_fields", {})
    if not source_fields or not audience_fields:
        return None

    top = [f for f, _ in sorted(
        {k: source_fields.get(k, 0) + audience_fields.get(k, 0)
         for k in set(source_fields) | set(audience_fields)}.items(),
        key=lambda x: x[1], reverse=True)[:8]][::-1]

    cited = [source_fields.get(f, 0) for f in top]
    citing = [audience_fields.get(f, 0) for f in top]

    fig = go.Figure()
    short = [_short(f) for f in top]
    fig.add_trace(go.Bar(
        name="You cite", y=short, x=[-v for v in cited], orientation="h",
        marker_color=COOL, customdata=list(zip(top, cited)),
        hovertemplate="%{customdata[0]}<br>you cite %{customdata[1]}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Cites you", y=short, x=citing, orientation="h",
        marker_color=WARM, customdata=top,
        hovertemplate="%{customdata}<br>cites you %{x}<extra></extra>",
    ))

    span = max(max(cited or [0]), max(citing or [0])) or 1
    step = max(1, int(round(span / 2)))

    fig.update_layout(**get_chart_layout("Knowledge flow"))
    fig.update_layout(
        barmode="relative", bargap=0.42,
        margin=dict(t=52, b=42, l=180, r=40),
        legend=dict(orientation="h", y=1.16, x=1, xanchor="right",
                    font=dict(size=11.5, color=INK_2, family=SANS)),
    )
    fig.update_xaxes(
        title="", gridcolor=RULE_SOFT, zeroline=True, zerolinecolor=INK_3, zerolinewidth=1,
        tickvals=[-2 * step, -step, 0, step, 2 * step],
        ticktext=[str(2 * step), str(step), "0", str(step), str(2 * step)],
        tickfont=_mono_ticks(),
    )
    fig.update_yaxes(tickfont=dict(size=12, color=INK_2, family=SANS), automargin=True)
    return fig


def create_citation_fields_chart(audience_fields: dict) -> go.Figure:
    """Fields citing this author. Warm — same entity as 'cites you' in the flow chart."""
    if not audience_fields:
        return None

    ordered = sorted(audience_fields.items(), key=lambda x: x[1], reverse=True)[:10][::-1]
    labels, values = [f[0] for f in ordered], [f[1] for f in ordered]
    total = sum(values) or 1

    fig = go.Figure(go.Bar(
        y=[_short(l) for l in labels], x=values, orientation="h",
        marker=dict(color=WARM, line=dict(width=0)),
        text=[f"{v}  {v/total*100:.0f}%" for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=INK, family=MONO),
        customdata=labels,
        hovertemplate="%{customdata}<br>%{x} citing works<extra></extra>",
    ))
    fig.update_layout(**get_chart_layout("Who cites your work"))
    fig.update_layout(margin=dict(t=46, b=42, l=180, r=110), bargap=0.42)
    fig.update_xaxes(title="", showgrid=True, gridcolor=RULE_SOFT, zeroline=False,
                     tickfont=_mono_ticks())
    fig.update_yaxes(tickfont=dict(size=12, color=INK_2, family=SANS), automargin=True)
    return fig


def create_scatter_chart(df: pd.DataFrame) -> go.Figure:
    """External diversity of each paper against its publication year. Single series."""
    fig = go.Figure(go.Scatter(
        x=df["year"].astype(int).tolist(),
        y=df["paper_index"].tolist(),
        mode="markers",
        marker=dict(size=9, color=COOL, line=dict(width=1.5, color=SURFACE)),
        text=df["title"].tolist(),
        hovertemplate="%{text}<br>%{x} · external diversity %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(**get_chart_layout("External diversity by year"))
    # A long career spans ~30 years; forcing dtick=1 crams every label.
    fig.update_xaxes(title="", showgrid=True, gridcolor=RULE_SOFT, zeroline=False,
                     nticks=8, tickformat="d", tickfont=_mono_ticks())
    fig.update_yaxes(title="", showgrid=True, gridcolor=RULE_SOFT, zeroline=False,
                     range=[0, 100], tickfont=_mono_ticks())
    return fig


def create_kde_chart(similarities: list) -> go.Figure:
    """Distribution of paper-to-citer similarity across every citing work."""
    if not similarities or len(similarities) < 2:
        return None
    try:
        sims = [float(s) for s in similarities if s is not None]
        if len(sims) < 2:
            return None
        kde = gaussian_kde(sims, bw_method=Config.KDE_BANDWIDTH)
        x_vals = np.linspace(min(sims), max(sims), 200)

        fig = go.Figure(go.Scatter(
            x=x_vals.tolist(), y=kde(x_vals).tolist(),
            fill="tozeroy", mode="lines",
            line=dict(color=COOL, width=2), fillcolor="rgba(42,120,214,0.10)",
            hovertemplate="similarity %{x:.2f}<extra></extra>",
        ))
        fig.update_layout(**get_chart_layout("Similarity to citing work"))
        fig.update_xaxes(title="", showgrid=True, gridcolor=RULE_SOFT, zeroline=False,
                         tickfont=_mono_ticks())
        fig.update_yaxes(title="", showgrid=False, zeroline=False, showticklabels=False)
        return fig
    except Exception as e:
        logger.warning(f"Could not create KDE chart: {e}")
        return None


def create_keywords_chart(keyword_counts: list) -> go.Figure:
    """Most frequent terms in the citing literature."""
    if not keyword_counts:
        return None

    top_keywords = keyword_counts[:10][::-1]
    labels, values = [kw[0] for kw in top_keywords], [kw[1] for kw in top_keywords]

    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker=dict(color=COOL, line=dict(width=0)),
        text=values, textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=INK, family=MONO),
        hovertemplate="%{y}<br>%{x} mentions<extra></extra>",
    ))
    fig.update_layout(**get_chart_layout("Terms in the citing literature"))
    fig.update_layout(margin=dict(t=46, b=42, l=150, r=90), bargap=0.42)
    fig.update_xaxes(title="", showgrid=True, gridcolor=RULE_SOFT, zeroline=False,
                     tickfont=_mono_ticks())
    fig.update_yaxes(tickfont=dict(size=12, color=INK_2, family=SANS), automargin=True)
    return fig
def create_comparison_chart(entries: list[dict]) -> go.Figure:
    """Four measures side by side for several researchers.

    Capped at three series on purpose. The validated palette clears every
    colour-vision gate for its first three slots when all pairs appear together,
    as they do inside each metric group here; a fourth would put yellow beside
    orange, which fails the normal-vision floor outright and cannot be rescued
    by labelling.
    """
    if not entries:
        return None

    metrics = [("External diversity", "citation_index"),
               ("Internal diversity", "dispersion_score"),
               ("Reference diversity", "reference_diversity"),
               ("Bridge", "bridge_score")]
    labels = [m[0] for m in metrics][::-1]

    fig = go.Figure()
    for entry, colour in zip(entries, COMPARE_SERIES):
        values = [entry["metrics"].get(key, 0) for _, key in metrics][::-1]
        fig.add_trace(go.Bar(
            name=entry["name"], y=labels, x=values, orientation="h",
            marker_color=colour, cliponaxis=False,
            text=[f"{v:.0f}" for v in values], textposition="outside",
            textfont=dict(size=11, color=INK, family=MONO),
            hovertemplate=f"<b>{entry['name']}</b><br>%{{y}} %{{x:.1f}}<extra></extra>",
        ))

    fig.update_layout(**get_chart_layout("Four measures compared", 420))
    fig.update_layout(
        barmode="group", bargap=0.34, bargroupgap=0.08,
        margin=dict(t=58, b=44, l=176, r=64),
        legend=dict(orientation="h", y=1.13, x=0, xanchor="left",
                    font=dict(size=11.5, color=INK_2, family=SANS)),
    )
    fig.update_xaxes(title="", range=[0, 108], showgrid=True, gridcolor=RULE_SOFT,
                     zeroline=False, tickvals=[0, 25, 50, 75, 100], tickfont=_mono_ticks())
    fig.update_yaxes(tickfont=dict(size=12, color=INK_2, family=SANS), automargin=True)
    return fig


# ============================================================================
# HTML EXPORT
# ============================================================================

def render_track(label: str, value: float, index: bool = False) -> str:
    """One row of the profile: label, measured rail, value. The Fieldtrip Index row
    is set apart with a heavier rail."""
    v = max(0.0, min(100.0, float(value)))
    ticks = "".join(f'<div class="track-tick" style="left:{t}%"></div>' for t in (0, 25, 50, 75, 100))
    return (
        f'<div class="track{" track-index" if index else ""}">'
        f'<div class="track-label">{label}</div>'
        '<div class="track-rail">'
        f'{ticks}'
        f'<div class="track-fill" style="width:{v}%"></div>'
        f'<div class="track-mark" style="left:{v}%"></div>'
        '</div>'
        f'<div class="track-val">{v:.0f}</div>'
        '</div>'
    )


REPORT_CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
:root {
  --paper:#FAFAFB; --surface:#FFFFFF; --ink:#14161A; --ink-2:#4A5058; --ink-3:#868D97;
  --rule:#E4E7EB; --rule-soft:#F1F3F5; --cool:#2A78D6;
  --sans:'IBM Plex Sans',-apple-system,BlinkMacSystemFont,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,monospace;
}
body { background:var(--paper); color:var(--ink); font-family:var(--sans);
       font-size:15px; line-height:1.5; -webkit-font-smoothing:antialiased; }
.container { max-width:1080px; margin:0 auto; padding:0 32px 96px; }
.masthead { display:flex; align-items:center; gap:13px; padding:24px 0 16px;
            border-bottom:1px solid var(--ink); margin-bottom:28px; }
.masthead-logo { display:flex; flex:none; }
.masthead-logo img { height:30px; width:auto; display:block; }
.masthead-mark { font-family:var(--mono); font-size:13px; font-weight:500;
                 letter-spacing:.14em; text-transform:uppercase; }
.masthead-note { margin-left:auto; font-family:var(--mono); font-size:12px; color:var(--ink-3); }
.masthead-note a { color:var(--ink-2); text-decoration:underline;
                   text-decoration-color:var(--rule); text-underline-offset:2px; }
.card { background:var(--surface); border:1px solid var(--rule); border-radius:10px;
        padding:26px 30px; margin-bottom:14px; }
.profile-head { display:flex; align-items:baseline; justify-content:space-between; gap:16px;
                padding-bottom:20px; border-bottom:1px solid var(--rule-soft); flex-wrap:wrap; }
.profile-who { font-size:21px; font-weight:600; letter-spacing:-.01em; }
.profile-id { font-family:var(--mono); font-size:12px; color:var(--ink-3); }
.track { display:grid; grid-template-columns:186px 1fr 60px; align-items:center; gap:20px;
         padding:14px 0; border-bottom:1px solid var(--rule-soft); }
.track:last-of-type { border-bottom:none; }
.track-label { font-size:12.5px; font-weight:500; letter-spacing:.07em;
               text-transform:uppercase; color:var(--ink-2); }
.track-rail { position:relative; height:22px; }
.track-rail::before { content:""; position:absolute; left:0; right:0; top:10px; height:2px;
                      background:var(--rule-soft); border-radius:1px; }
.track-fill { position:absolute; left:0; top:10px; height:2px; background:var(--cool); border-radius:1px; }
.track-mark { position:absolute; top:2px; width:2px; height:18px; background:var(--cool);
              border-radius:1px; transform:translateX(-1px); }
.track-tick { position:absolute; top:14px; width:1px; height:4px; background:var(--rule); }
.track-index { border-top:1px solid var(--rule); margin-top:4px; padding-top:18px; }
.track-index .track-label { color:var(--cool); }
.track-index .track-fill { height:4px; top:9px; border-radius:2px; }
.track-index .track-mark { width:3px; }
.track-index .track-val { color:var(--cool); }
.track-val { font-family:var(--mono); font-size:22px; font-weight:500; text-align:right;
             font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
.profile-foot { display:flex; gap:24px; flex-wrap:wrap; padding-top:17px; margin-top:8px;
                border-top:1px solid var(--rule-soft); font-size:12.5px; color:var(--ink-3); }
.profile-foot b { font-family:var(--mono); font-weight:500; color:var(--ink-2); }
.section-label { font-size:12px; font-weight:500; letter-spacing:.1em; text-transform:uppercase;
                 color:var(--ink-3); margin:34px 0 10px; }
.chart-container { background:var(--surface); border:1px solid var(--rule); border-radius:10px;
                   padding:18px 20px; margin-bottom:14px; }
.chart-container .plotly-graph-div { width:100% !important; }
table { width:100%; border-collapse:collapse; font-size:14px; }
th { text-align:left; font-size:11.5px; font-weight:500; letter-spacing:.09em; text-transform:uppercase;
     color:var(--ink-3); padding:0 12px 10px; border-bottom:1px solid var(--rule); }
td { padding:11px 12px; border-bottom:1px solid var(--rule-soft); color:var(--ink-2); }
tr:last-child td { border-bottom:none; }
td.n { font-family:var(--mono); font-variant-numeric:tabular-nums; text-align:right; color:var(--ink); }
.rank { font-family:var(--mono); color:var(--ink-3); font-size:12px; }
.note { font-size:13px; color:var(--ink-3); margin-top:10px; max-width:70ch; }
.footer { color:var(--ink-3); font-size:12.5px; margin-top:40px; padding-top:20px;
          border-top:1px solid var(--rule); font-family:var(--mono); }
@media (max-width:860px) { .track { grid-template-columns:1fr 56px; gap:6px 14px; }
                           .track-rail { grid-column:1/3; order:3; } }
"""


def generate_html_report(author_name: str, df: pd.DataFrame, metrics: dict, composite: float,
                         scatter_fig, kde_fig, dispersion_fig, ref_div_fig, bridge_fig,
                         field_breakdown_fig, keyword_fig) -> str:
    tempdir = tempfile.gettempdir()
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in author_name) or "report"
    html_path = os.path.join(tempdir, f"interdisciplinary_{safe_name}.html")

    charts_html = []
    chart_configs = [
        ("Paper similarity", dispersion_fig),
        ("Knowledge flow", bridge_fig),
        ("Fields you draw on", ref_div_fig),
        ("Who cites your work", field_breakdown_fig),
        ("External diversity by year", scatter_fig),
        ("Similarity to citing work", kde_fig),
        ("Terms in the citing literature", keyword_fig),
    ]

    for title, fig in chart_configs:
        if fig is not None:
            try:
                chart_html = fig.to_html(full_html=False, include_plotlyjs=False,
                                         config={'displayModeBar': False, 'responsive': True})
                charts_html.append(f'<div class="chart-container">{chart_html}</div>')
            except Exception as e:
                logger.warning(f"Could not convert {title} chart: {e}")

    papers_table = ""
    if not df.empty:
        rows = []
        for idx, row in df.iterrows():
            title_text = str(row["Title"])
            if len(title_text) > 130:
                title_text = title_text[:130] + "…"
            rows.append(
                f'<tr><td class="rank">{idx+1:02d}</td><td>{title_text}</td>'
                f'<td class="n">{row["Year"]}</td>'
                f'<td class="n">{row["Index (%)"]:.0f}</td>'
                f'<td class="n">{row.get("citation_count", "—"):,}</td></tr>'
            )
        papers_table = (
            '<table><thead><tr><th style="width:44px"></th><th>Paper</th>'
            '<th style="width:70px">Year</th><th style="width:104px">External</th>'
            '<th style="width:96px">Cited by</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table>'
        )

    tracks = (
        render_track("External diversity", metrics.get('citation_index', 0))
        + render_track("Internal diversity", metrics.get('dispersion_score', 0))
        + render_track("Reference diversity", metrics.get('reference_diversity', 0))
        + render_track("Bridge", metrics.get('bridge_score', 0))
        + render_track("Fieldtrip Index", composite, index=True)
    )

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fieldtrip Index — {author_name}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="container">
  <div class="masthead">
    <span class="masthead-logo">{MARK_IMG}</span>
    <span class="masthead-mark">Fieldtrip Index</span>
    <span class="masthead-note">via <a href="https://openalex.org">openalex</a> &middot; {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
  </div>

  <div class="card">
    <div class="profile-head">
      <div class="profile-who">{author_name}</div>
      <div class="profile-id">Fieldtrip Index {composite:.0f}</div>
    </div>
    {tracks}
    <p class="note">Each measure runs 0&ndash;100. Internal and reference diversity
    average into <em>range</em>, external diversity and bridge into <em>reach</em>, and
    the <em>Fieldtrip Index</em> is the geometric mean of the two.
    These are relative measures: because scientific abstracts share a great deal of
    language, the semantic scores rarely approach zero even for tightly focused work,
    so they are most useful compared against another researcher or against the same
    researcher over time.</p>
  </div>

  <div class="section-label">Papers analysed</div>
  <div class="card">{papers_table}</div>

  <div class="section-label">Charts</div>
  {''.join(charts_html)}

  <div class="footer">Generated by the Fieldtrip Index</div>
</div>
<script>
  window.addEventListener('resize', function () {{
    document.querySelectorAll('.js-plotly-plot').forEach(function (p) {{ Plotly.Plots.resize(p); }});
  }});
</script>
</body>
</html>'''

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return html_path

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

async def analyze_author(author_id: str, author_name: str, cache_dir: str = None,
                         progress_cb=None, exclude_ids: set | None = None):
    logger.info(f"={'='*50}")
    logger.info(f"Starting analysis for: {author_name} ({author_id})")
    start_time = datetime.now()

    def step(frac: float, message: str):
        """Report a step to the console and, when the UI supplies one, to the progress bar."""
        logger.info(f"      {message}")
        if progress_cb is not None:
            try:
                progress_cb(frac, desc=message)
            except Exception:
                pass  # a progress sink must never break the analysis

    requested = Config.TOP_N_PAPERS
    fetch_params = dict(
        top_n=requested,
        citations_per_paper=Config.CITATIONS_PER_PAPER,
        references_per_paper=Config.REFERENCES_PER_PAPER,
        seed=Config.SEED,
        exclude_self_citations=Config.EXCLUDE_SELF_CITATIONS,
    )
    cache_key = {"schema": CACHE_SCHEMA, **fetch_params}

    # ---- Everything the analysis needs, in ~21 requests, fetched once ----
    if is_cached(author_id, cache_dir, params=cache_key):
        step(0.05, "Reading cached data")
    else:
        step(0.05, "Fetching papers, citations and references")
        await build_author_cache(author_id, cache_dir=cache_dir, **fetch_params)

    data = load_author_cache(author_id, cache_dir)
    top_papers = data["top_papers"]
    reference_topics = data.get("reference_topics", {})

    # Papers the user has marked as not theirs are removed before anything is
    # measured: a misattributed paper contaminates all four measures at once,
    # contributing its own text, its references and its audience.
    def paper_option(p):
        return (f"{p['title'][:110]}{'…' if len(p['title']) > 110 else ''}  ·  "
                f"{p.get('year') or '—'}  ·  {p.get('citation_count', 0):,} cited", p["id"])

    # Built before the filter runs: an excluded paper has to stay on the list,
    # ticked, or there is no way to see what was removed or to put one back.
    all_paper_choices = [paper_option(p) for p in top_papers]

    excluded_count = 0
    if exclude_ids:
        kept = [p for p in top_papers if p["id"] not in exclude_ids]
        excluded_count = len(top_papers) - len(kept)
        top_papers = kept
        logger.info(f"      excluding {excluded_count} paper(s) marked as not this author's")

    logger.info(f"      {len(top_papers)} papers in cache")

    if not top_papers:
        raise NotEnoughData(
            f"OpenAlex lists no works for {author_name} that carry both an abstract and a DOI, "
            "so there is nothing to compare. This is common for very new profiles and for fields "
            "where abstracts are not indexed."
        )

    # ---- External diversity, over the cached citing sample ----
    results, uncited, all_keywords, similarities_all, all_citing_texts = [], [], [], [], []
    total = len(top_papers)
    for idx, paper in enumerate(top_papers):
        step(0.15 + 0.45 * (idx / max(total, 1)),
             f"Comparing citing work — paper {idx + 1} of {total}")
        abs_orig = reconstruct_abstract(paper.get("abstract_inverted_index"))
        if not abs_orig:
            continue
        citing_texts = [c["abstract"] for c in paper.get("citing", []) if c.get("abstract")]
        if not citing_texts:
            # Nothing usable cites it, so it has no external diversity — but
            # its own text still says something about the spread of the work.
            uncited.append({"title": paper["title"], "year": paper["year"],
                            "citation_count": paper.get("citation_count", 0),
                            "abstract": abs_orig})
            continue
        # Embedding is CPU-bound and was running straight on the event loop,
        # which froze the whole server — and every progress update with it —
        # for the length of the analysis. torch releases the GIL during compute,
        # so a worker thread genuinely gives the loop back.
        avg_sim, idx_val, count, sims = await asyncio.to_thread(
            calculate_similarity_and_index, abs_orig, citing_texts)
        results.append({
            "title": paper["title"],
            "doi": paper.get("doi"),
            "year": paper["year"],
            "paper_index": idx_val * 100,
            "citation_count": paper.get("citation_count", 0),
            "abstract": abs_orig,
        })
        similarities_all.extend(sims)
        all_citing_texts.extend(citing_texts)

    if not results:
        raise NotEnoughData(
            f"None of the {len(top_papers)} indexed papers for {author_name} has citing work with "
            "an abstract, so external diversity cannot be computed. Well-cited work published some "
            "years ago gives the most reliable reading."
        )

    # ---- Reference and bridge diversity, from the same cached records ----
    step(0.62, "Measuring the spread of the reference lists")
    ref_diversity = calculate_reference_diversity(top_papers, reference_topics)

    step(0.70, "Working out which fields cite back")
    audience_counts, aud_ok, aud_missing = audience_field_counts(top_papers)
    bridge_data = calculate_bridge_score(ref_diversity["field_counts"], audience_counts)
    audience_coverage = aud_ok / (aud_ok + aud_missing) if (aud_ok + aud_missing) else 0.0

    step(0.80, "Pulling out common terms")

    def extract_keywords(texts):
        found = []
        for text in texts:
            try:
                found.extend(kw[0] for kw in
                             kw_model.extract_keywords(text, top_n=3, stop_words='english'))
            except Exception:
                pass
        return found

    all_keywords = await asyncio.to_thread(extract_keywords, all_citing_texts[:50])

    step(0.90, "Drawing the charts")
    df = pd.DataFrame(results).dropna().sort_values(by="paper_index", ascending=False).reset_index(drop=True)
    citation_index = df['paper_index'].mean() if not df.empty else 0

    # Internal diversity asks how far the author's own papers sit from each
    # other, which needs no citing work, so papers whose citers carry no
    # abstracts still belong here.
    uncited_sorted = sorted(uncited, key=lambda p: p["citation_count"], reverse=True)
    spread_abstracts = (df['abstract'].tolist() if 'abstract' in df.columns else []) \
        + [p["abstract"] for p in uncited_sorted]
    sorted_titles = (df['title'].tolist() if 'title' in df.columns else []) \
        + [p["title"] for p in uncited_sorted]

    dispersion_data = await asyncio.to_thread(calculate_embedding_dispersion, spread_abstracts)

    all_metrics = {
        'citation_index': citation_index,
        'dispersion_score': dispersion_data['dispersion_score'],
        'reference_diversity': ref_diversity['diversity_index'],
        'bridge_score': bridge_data['bridge_score'],
    }
    axes = composite_score(all_metrics)
    composite = axes["composite"]

    scatter = create_scatter_chart(df) if not df.empty else None
    kde_fig = create_kde_chart(similarities_all)
    dispersion_chart = create_dispersion_chart(dispersion_data, sorted_titles)
    ref_diversity_chart = create_reference_diversity_chart(ref_diversity)
    bridge_chart = create_bridge_chart(bridge_data)
    field_breakdown_chart = create_citation_fields_chart(bridge_data.get('audience_fields', {}))
    keyword_chart = create_keywords_chart(Counter(all_keywords).most_common(10))

    df_top_papers = papers_table_html(
        df[["title", "doi", "year", "paper_index", "citation_count"]].to_dict("records"))

    df_report = df.rename(columns={"title": "Title", "year": "Year", "paper_index": "Index (%)"})
    html_path = generate_html_report(author_name, df_report, all_metrics, composite, scatter, kde_fig, dispersion_chart, ref_diversity_chart, bridge_chart, field_breakdown_chart, keyword_chart)

    # ---- provenance / coverage, shown under the profile ----
    n_refs = ref_diversity["classified"] + ref_diversity["unclassified"]
    n_citing = aud_ok + aud_missing
    coverage = min(ref_diversity["coverage"], audience_coverage)
    retrieved = get_cache_timestamp(author_id, cache_dir)
    retrieved_str = retrieved.strftime("%Y-%m-%d %H:%M") if retrieved else "just now"

    # The effective paper count varies by author — OpenAlex may hold fewer works
    # with an abstract and a DOI than were asked for — and the bridge score
    # depends on it, so the shortfall is stated rather than left to be inferred.
    missing = requested - len(results)
    shortfall = ""
    if missing > 0:
        reasons = []
        if excluded_count:
            reasons.append(f"{excluded_count} excluded by you")
        if uncited:
            reasons.append(f"{len(uncited)} not cited yet")
        reason = ", ".join(reasons) or "OpenAlex holds no more with an abstract and a DOI"
        shortfall = (f" <span class=\"foot-note\">of {requested}"
                     f" &mdash; {reason}</span>")

    results_html = f"""
    <div class="profile">
      <div class="profile-head">
        <div class="profile-who">{author_name}</div>
        <div class="profile-id">{author_id} &middot; retrieved {retrieved_str}</div>
      </div>
      {render_track("External diversity", citation_index)}
      {render_track("Internal diversity", dispersion_data['dispersion_score'])}
      {render_track("Reference diversity", ref_diversity['diversity_index'])}
      {render_track("Bridge", bridge_data['bridge_score'])}
      {render_track("Fieldtrip Index", composite, index=True)}
      <div class="profile-foot">
        <span>Range <b>{axes['range']:.0f}</b></span>
        <span>Reach <b>{axes['reach']:.0f}</b></span>
        <span>Papers analysed <b>{len(results)}</b>{shortfall}</span>
        <span>Topic spread over <b>{len(spread_abstracts)}</b></span>
        <span>Citing works <b>{n_citing}</b></span>
        <span>References <b>{n_refs}</b></span>
        <span>Effectively <b>{ref_diversity['effective_fields']:.1f}</b> fields</span>
        <span>Bridged <b>{', '.join(bridge_data['bridged_fields'][:3]) or 'none'}</b></span>
        <span>Classified <b>{coverage:.0%}</b></span>
      </div>
    </div>
    """

    explanation = """### Reading these four numbers

Each measures something different, on a 0&ndash;100 scale. They pair into two
axes: **range**, what the researcher themselves does (internal + reference), and
**reach**, how far the work travels (external + bridge). The **Fieldtrip Index** is
the geometric mean of the two, so it cannot be earned by one side alone.

| | What it measures |
|---|---|
| **External diversity** | How far the work citing you sits from your own, semantically |
| **Internal diversity** | How far your own papers sit from each other |
| **Reference diversity** | How widely your references spread across fields, weighting how far apart those fields are |
| **Bridge** | Share of the work citing you that your own reading does not account for |

These are **relative** measures. Because any two scientific abstracts share a
good deal of language, the semantic scores rarely fall near zero even for tightly
focused work &mdash; so a number here is most useful compared against another
researcher, or against the same researcher over time, rather than read as an
absolute grade. The four are also not independent: external and internal
diversity share an embedding space, and reference diversity and bridge share a
field taxonomy. Each axis mixes one of each, which is why they are paired that way.
Prefer the four-part profile above to the single number.
"""

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"={'='*50}")
    logger.info(f"Analysis completed for {author_name}")
    logger.info(f"      Time: {elapsed:.1f}s | Fieldtrip Index: {composite:.1f}")
    logger.info(f"      Ext: {citation_index:.1f} | Int: {dispersion_data['dispersion_score']:.1f} | Ref: {ref_diversity['diversity_index']:.1f} | Bridge: {bridge_data['bridge_score']:.1f}")
    logger.info(f"={'='*50}")

    detail = {
        "paper_choices": all_paper_choices,
        "reference_fields": ref_diversity["field_counts"],
        "field_domains": ref_diversity.get("field_domains", {}),
        "audience_fields": bridge_data["audience_fields"],
        "imbalance": bridge_data.get("imbalance", {}),
        "bridged_fields": bridge_data.get("bridged_fields", []),
        "effective_fields": ref_diversity["effective_fields"],
        "papers_analysed": len(results),
        "papers_excluded": excluded_count,
        "coverage": min(ref_diversity["coverage"], audience_coverage),
    }

    return (df_report, results_html, df_top_papers, scatter, kde_fig, dispersion_chart, ref_diversity_chart, bridge_chart, field_breakdown_chart, keyword_chart, explanation, html_path, all_metrics, detail)

def candidate_choices(candidates: list[dict]) -> list[tuple[str, str]]:
    """Radio options, one per candidate, as (label, author_id).

    A table was tried first and truncated every column in a narrow pane, which
    destroys the only information that tells two same-named researchers apart.
    Two stacked lines survive any width.
    """
    options = []
    for c in candidates:
        # The name alone occupies the first line, which the card styling sets in
        # bold; anything else there would be emphasised with it.
        head = c["name"]
        facts = [f for f in (c.get("institution"), c.get("span")) if f]
        facts.append(f"{c['works_count']:,} works")
        facts.append(f"{c['cited_by_count']:,} cited")
        if c.get("orcid"):
            facts.append(f"ORCID {c['orcid']}")
        detail = " · ".join(facts)
        if c.get("topics"):
            detail += f"\n{c['topics']}"
        options.append((f"{head}\n{detail}", c["id"]))
    return options


async def analyse_for_comparison(picked: list[dict], cache_dir: str = None,
                                 on_step=None) -> tuple[list[dict], list[str]]:
    """Analyse researchers the user has already chosen by profile.

    Resolution happens in the picker, so this never guesses which "J. Smith" was
    meant — the earlier version took the top search hit, which could silently
    compare the wrong person, or a merged record.

    Returns (entries, problems); one researcher failing does not sink the rest.
    """
    entries: list[dict] = []
    problems: list[str] = []
    total = len(picked) or 1

    for i, who in enumerate(picked):
        if any(e["id"] == who["id"] for e in entries):
            continue

        # Map each researcher's own 0-1 progress onto their slice of the bar,
        # so the inner steps of a 15-second analysis are visible rather than
        # the bar sitting still between researchers.
        def slice_of(frac, desc, _i=i, _who=who):
            if on_step:
                on_step((_i + max(0.0, min(1.0, frac))) / total,
                        f"{_who['name']} — {desc}")

        if on_step:
            on_step(i / total, f"Analysing {who['name']}")
        try:
            out = await analyze_author(who["id"], who["name"], cache_dir,
                                       progress_cb=lambda f, desc="": slice_of(f, desc))
        except (RateLimitExceeded, Throttled, InvalidAPIKey):
            raise
        except NotEnoughData as e:
            problems.append(f"{who['name']}: {e}")
            continue
        except Exception as e:
            logger.exception(f"Comparison failed for {who['id']}")
            problems.append(f"{who['name']}: analysis failed ({type(e).__name__})")
            continue
        entries.append({"name": who["name"], "id": who["id"],
                        "metrics": out[12], "detail": out[13]})

    return entries, problems


def papers_table_html(rows: list[dict]) -> str:
    """The analysed papers as an HTML table.

    Gradio's dataframe truncates its headers rather than wrapping them, which
    turned "External diversity" into "E…" in a narrow pane. Rendering the table
    directly keeps the headers legible and lets the titles be real links.
    """
    if not rows:
        return ""
    body = []
    for i, r in enumerate(rows, 1):
        title = r["title"]
        cell = f'<a href="{r["doi"]}" target="_blank" rel="noopener">{title}</a>' if r.get("doi") else title
        body.append(
            f'<tr><td class="rank">{i:02d}</td><td class="title">{cell}</td>'
            f'<td class="num" data-label="Year">{r["year"]}</td>'
            f'<td class="num" data-label="External">{r["paper_index"]:.0f}</td>'
            f'<td class="num" data-label="Cited by">{r["citation_count"]:,}</td></tr>'
        )
    return (
        '<div class="section-label">Papers analysed, ranked by external diversity</div>'
        '<div class="table-card"><table class="data-table"><thead><tr>'
        '<th class="c-rank"></th><th>Paper</th>'
        '<th class="c-num">Year</th><th class="c-num">External<br>diversity</th>'
        '<th class="c-num">Cited by</th>'
        '</tr></thead><tbody>' + "".join(body) + '</tbody></table></div>'
    )


MEASURE_LABELS = [
    ("citation_index", "external diversity", "how far the work citing them sits from their own"),
    ("dispersion_score", "internal diversity", "how far their own papers sit from each other"),
    ("reference_diversity", "reference diversity", "how widely they read"),
    ("bridge_score", "bridge", "how much of their audience their own reading does not explain"),
]

def composite_score(metrics: dict) -> dict:
    """Range, reach, and their geometric mean.

    Range is what the researcher themselves does (internal + reference diversity);
    reach is how far the work travels (external diversity + bridge). Each axis pairs
    one text measure with one field measure, so the two inputs inside an axis come
    from different sources and the shared signal is not counted twice. The axes are
    averaged arithmetically because bridge is legitimately zero for a focused
    researcher and must not zero out the whole axis; across axes the geometric mean
    is deliberate — a lopsided profile is pulled toward its weaker side, so the
    composite cannot be earned by one measure alone.
    """
    range_ = (metrics["dispersion_score"] + metrics["reference_diversity"]) / 2
    reach = (metrics["citation_index"] + metrics["bridge_score"]) / 2
    return {"range": range_, "reach": reach, "composite": math.sqrt(range_ * reach)}


# Two 25-paper samples will differ by a few points for no reason at all. This is
# a rule of thumb, not an interval — the app reports point estimates, so a gap
# below it is simply not narrated rather than being called equal.
MEANINGFUL_GAP = 8.0


def _shares(counts: dict) -> dict:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _overlap(a: dict, b: dict) -> float:
    """How much two field distributions coincide, 0 to 1.

    The summed minimum of the two shares: 1 when identical, 0 when they have no
    field in common. Easier to read than a cosine and it has a plain meaning —
    the fraction of one distribution you could lay directly on top of the other.
    """
    if not a or not b:
        return 0.0
    sa, sb = _shares(a), _shares(b)
    return sum(min(sa.get(f, 0.0), sb.get(f, 0.0)) for f in set(sa) | set(sb))


def _distinctive(a: dict, b: dict, limit: int = 3) -> list[tuple[str, float, float]]:
    """Fields where the first draws much more heavily than the second."""
    sa, sb = _shares(a), _shares(b)
    gaps = [(f, sa[f], sb.get(f, 0.0)) for f in sa]
    gaps.sort(key=lambda x: x[1] - x[2], reverse=True)
    return [g for g in gaps[:limit] if g[1] - g[2] > 0.05]


def comparison_insights(entries: list[dict]) -> str:
    """What the two profiles actually say about each other.

    Deliberately descriptive. These measures describe citation patterns, not
    quality, so nothing here ranks anyone or recommends anyone.
    """
    if len(entries) < 2:
        return ""

    pairs = [(entries[i], entries[j])
             for i in range(len(entries)) for j in range(i + 1, len(entries))]
    blocks = []

    for a, b in pairs:
        an, bn = a["name"], b["name"]
        da, db = a.get("detail") or {}, b.get("detail") or {}
        lines = []

        # --- where they differ ---
        gaps = []
        for key, label, gloss in MEASURE_LABELS:
            va, vb = a["metrics"].get(key, 0), b["metrics"].get(key, 0)
            if abs(va - vb) >= MEANINGFUL_GAP:
                higher, lower = (an, bn) if va > vb else (bn, an)
                gaps.append(f"<li><b>{label.capitalize()}</b> — {higher} is "
                            f"{abs(va - vb):.0f} points higher ({max(va, vb):.0f} against "
                            f"{min(va, vb):.0f}): {gloss}.</li>")
        if gaps:
            lines.append("<p>Where they differ</p><ul>" + "".join(gaps) + "</ul>")
        else:
            lines.append("<p>No measure separates them by more than "
                         f"{MEANINGFUL_GAP:.0f} points, which is inside what two "
                         "25-paper samples can differ by for no reason.</p>")

        # --- what they read, and who reads them ---
        read = _overlap(da.get("reference_fields", {}), db.get("reference_fields", {}))
        heard = _overlap(da.get("audience_fields", {}), db.get("audience_fields", {}))

        def describe(x):
            return ("almost entirely the same" if x > 0.75 else
                    "largely the same" if x > 0.5 else
                    "partly shared" if x > 0.25 else "barely overlapping")

        lines.append(
            f"<p>Overlap</p><ul>"
            f"<li>What they <b>read</b> is {describe(read)} "
            f"(<span class='ins-num'>{read:.0%}</span> of their reference fields coincide).</li>"
            f"<li>Who <b>reads them</b> is {describe(heard)} "
            f"(<span class='ins-num'>{heard:.0%}</span>).</li></ul>")

        # --- what each brings that the other does not ---
        only_a = _distinctive(da.get("reference_fields", {}), db.get("reference_fields", {}))
        only_b = _distinctive(db.get("reference_fields", {}), da.get("reference_fields", {}))
        if only_a or only_b:
            def render(name, items):
                if not items:
                    return f"<li>{name} draws on nothing the other does not.</li>"
                bits = ", ".join(f"{f} ({sa:.0%} against {sb:.0%})" for f, sa, sb in items)
                return f"<li><b>{name}</b> draws on {bits}.</li>"
            lines.append("<p>Literature each brings</p><ul>"
                         + render(an, only_a) + render(bn, only_b) + "</ul>")

        # --- complementarity: do they reach the same places? ---
        ba, bb = set(da.get("bridged_fields", [])), set(db.get("bridged_fields", []))
        shared, a_only, b_only = ba & bb, ba - bb, bb - ba
        if ba or bb:
            parts = []
            if shared:
                parts.append(f"both reach {', '.join(sorted(shared))}")
            if a_only:
                parts.append(f"only {an} reaches {', '.join(sorted(a_only))}")
            if b_only:
                parts.append(f"only {bn} reaches {', '.join(sorted(b_only))}")
            verdict = ("They bridge into the same places, so on this evidence they "
                       "extend into similar territory."
                       if shared and not (a_only or b_only) else
                       "They bridge into different places, so on this evidence they "
                       "extend into different territory."
                       if (a_only and b_only) and not shared else
                       "Their bridging partly coincides.")
            lines.append(f"<p>Where they reach beyond their own reading</p>"
                         f"<ul><li>{'; '.join(parts)}.</li><li>{verdict}</li></ul>")

        blocks.append(f'<div class="insight"><h4>{an} and {bn}</h4>{"".join(lines)}</div>')

    caveat = ('<p class="ins-caveat">These describe citation patterns, not quality or '
              'fit. Nothing here ranks anyone: a higher number means work travelling '
              'further from its own field, which is not the same as work being better, '
              'and the comparison is only valid because both were analysed the same way.</p>')
    return f'<div class="insights"><div class="section-label">What the comparison shows</div>{"".join(blocks)}{caveat}</div>'


def comparison_table_html(entries: list[dict]) -> str:
    """Comparison figures as an HTML table, for the same reason."""
    if not entries:
        return ""
    body = []
    for e in entries:
        m = e["metrics"]
        body.append(
            f'<tr><td class="title">{e["name"]}</td>'
            f'<td class="num" data-label="External">{m["citation_index"]:.1f}</td>'
            f'<td class="num" data-label="Internal">{m["dispersion_score"]:.1f}</td>'
            f'<td class="num" data-label="Reference">{m["reference_diversity"]:.1f}</td>'
            f'<td class="num" data-label="Bridge">{m["bridge_score"]:.1f}</td>'
            f'<td class="num strong" data-label="Fieldtrip Index">{composite_score(m)["composite"]:.1f}</td></tr>'
        )
    return (
        '<div class="table-card"><table class="data-table"><thead><tr><th>Researcher</th>'
        '<th class="c-num">External<br>diversity</th><th class="c-num">Internal<br>diversity</th>'
        '<th class="c-num">Reference<br>diversity</th><th class="c-num">Bridge</th>'
        '<th class="c-num">Fieldtrip<br>Index</th></tr></thead><tbody>' + "".join(body) + '</tbody></table></div>'
    )


# ============================================================================
# GRADIO UI
# ============================================================================

# The mark is a painted hex (brand/fieldtrip-hex.png, generated by the p5.brush
# sketch in brand/fieldtrip/). It is read once at import and inlined as data URIs,
# so it needs no static route and cannot 404. The masthead uses the 256px cut,
# the favicon the 64px one.
_BRAND = pathlib.Path(__file__).resolve().parent / "brand"

def _data_uri(name: str) -> str:
    try:
        return "data:image/png;base64," + base64.b64encode((_BRAND / name).read_bytes()).decode()
    except OSError:
        logger.warning(f"Brand asset {name} not found; the mark will not render")
        return ""

MARK_IMG = f'<img src="{_data_uri("fieldtrip-hex@256.png")}" alt="Fieldtrip Index" width="26" height="30">'

# Gradio 6's launch(favicon_path=...) emits no icon link at all, so the tab icon
# is set here instead as a data URI.
_FAVICON = _data_uri("fieldtrip-hex@64.png")

PAGE_HEAD = f"""
<link rel="icon" type="image/png" href="{_FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
"""

custom_css = """
:root {
    --paper: #FAFAFB;
    --surface: #FFFFFF;
    --ink: #14161A;
    --ink-2: #4A5058;
    --ink-3: #868D97;
    --rule: #E4E7EB;
    --rule-soft: #F1F3F5;
    --cool: #2A78D6;
    --warm: #EB6834;
    --sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace;
}

/* ---------- shell ---------- */
/* Gradio 6 makes this a flex child, so it sizes to content unless told to
   stretch; max-width alone leaves the page collapsed to about half width. */
.gradio-container {
    width: 100% !important;
    max-width: 1120px !important;
    margin: 0 auto !important;
    background: var(--paper) !important;
    font-family: var(--sans) !important;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased;
}
.gradio-container > .main,
.gradio-container .wrap > main.contain { width: 100% !important; }
.gradio-container .prose, .gradio-container p, .gradio-container li { color: var(--ink-2); }
.gradio-container h1, .gradio-container h2, .gradio-container h3 { color: var(--ink); font-weight: 600; letter-spacing: -0.01em; }

/* ---------- masthead ---------- */
.masthead {
    display: flex; align-items: center; gap: 13px;
    padding: 22px 2px 16px;
    border-bottom: 1px solid var(--ink);
    margin-bottom: 28px;
}
.masthead-logo { display: flex; flex: none; }
.masthead-logo img { height: 30px; width: auto; display: block; }
.masthead-mark {
    font-family: var(--mono); font-size: 13px; font-weight: 500;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink);
}
.masthead-note {
    margin-left: auto; font-family: var(--mono); font-size: 12px; color: var(--ink-3);
}
.masthead-note a {
    color: var(--ink-2); text-decoration: underline;
    text-decoration-color: var(--rule); text-underline-offset: 2px;
}
.masthead-note a:hover { color: var(--ink); text-decoration-color: var(--ink); }
.masthead-lede {
    font-size: 14.5px; color: var(--ink-2); margin: -14px 2px 26px; max-width: 62ch;
}

/* ---------- form controls ---------- */
.gradio-container label, .gradio-container .label-wrap span {
    font-family: var(--sans) !important; font-weight: 500 !important;
    font-size: 13px !important; color: var(--ink-2) !important;
}
.gradio-container input[type="text"], .gradio-container textarea, .gradio-container select {
    font-family: var(--sans) !important; font-size: 15px !important;
    border: 1px solid var(--rule) !important; border-radius: 6px !important;
    background: var(--surface) !important; color: var(--ink) !important;
    padding: 10px 13px !important; box-shadow: none !important;
}
.gradio-container input[type="text"]:focus, .gradio-container textarea:focus {
    border-color: var(--ink) !important;
    box-shadow: 0 0 0 3px rgba(20,22,26,0.07) !important;
    outline: none !important;
}

.gradio-container button {
    font-family: var(--sans) !important; font-weight: 500 !important;
    font-size: 14px !important; border-radius: 6px !important;
}
.primary-btn, .primary-btn * {
    background: var(--ink) !important; color: #FFF !important; fill: #FFF !important;
    border: 1px solid var(--ink) !important; padding: 10px 22px !important;
}
.primary-btn:hover { background: #000 !important; }
.ghost-btn, .ghost-btn * {
    background: transparent !important; color: var(--ink) !important;
    border: 1px solid var(--rule) !important; padding: 10px 18px !important;
}
.ghost-btn:hover { background: var(--rule-soft) !important; border-color: var(--ink-3) !important; }
.gradio-container button:focus-visible { outline: 2px solid var(--ink) !important; outline-offset: 2px !important; }

/* ---------- the profile: hero ---------- */
.profile {
    background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
    padding: 28px 30px 22px; margin: 4px 0 6px;
}
.profile-head {
    display: flex; align-items: baseline; justify-content: space-between; gap: 16px;
    padding-bottom: 20px; border-bottom: 1px solid var(--rule-soft); flex-wrap: wrap;
}
.profile-who { font-size: 21px; font-weight: 600; letter-spacing: -0.01em; color: var(--ink); }
.profile-id { font-family: var(--mono); font-size: 12px; color: var(--ink-3); }

.track {
    display: grid; grid-template-columns: 186px 1fr 60px; align-items: center;
    gap: 20px; padding: 14px 0; border-bottom: 1px solid var(--rule-soft);
}
.track:last-of-type { border-bottom: none; }
.track-label {
    font-size: 12.5px; font-weight: 500; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--ink-2);
}
.track-rail { position: relative; height: 22px; }
.track-rail::before {
    content: ""; position: absolute; left: 0; right: 0; top: 10px;
    height: 2px; background: var(--rule-soft); border-radius: 1px;
}
.track-fill { position: absolute; left: 0; top: 10px; height: 2px; background: var(--cool); border-radius: 1px; }
.track-mark { position: absolute; top: 2px; width: 2px; height: 18px; background: var(--cool); border-radius: 1px; transform: translateX(-1px); }
.track-tick { position: absolute; top: 14px; width: 1px; height: 4px; background: var(--rule); }
.track-index { border-top: 1px solid var(--rule); margin-top: 4px; padding-top: 18px; }
.track-index .track-label { color: var(--cool); }
.track-index .track-fill { height: 4px; top: 9px; border-radius: 2px; }
.track-index .track-mark { width: 3px; }
.track-index .track-val { color: var(--cool); }
.track-val {
    font-family: var(--mono); font-size: 22px; font-weight: 500; text-align: right;
    font-variant-numeric: tabular-nums; letter-spacing: -0.02em; color: var(--ink);
}
.profile-foot {
    display: flex; gap: 24px; flex-wrap: wrap; padding-top: 17px; margin-top: 8px;
    border-top: 1px solid var(--rule-soft); font-size: 12.5px; color: var(--ink-3);
}
.profile-foot b { font-family: var(--mono); font-weight: 500; color: var(--ink-2); }
.profile-foot .foot-note { color: var(--ink-3); }

/* ---------- notices ---------- */
.notice { border-radius: 0 6px 6px 0; padding: 13px 16px; font-size: 13.5px;
    color: var(--ink-2); margin: 6px 0; }
.notice b { color: var(--ink); font-weight: 600; }
.notice-warn { border-left: 2px solid var(--warm); background: #FFF7F3; }
.notice-info { border-left: 2px solid var(--cool); background: #F4F8FD; }
.notice ul { margin: 8px 0 0 18px; }
.notice li { margin: 3px 0; color: var(--ink-2); }

/* ---------- progress ---------- */
.progress-box {
    display: flex; align-items: center; gap: 14px; padding: 20px 22px;
    background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; margin: 6px 0;
}
.progress-spinner {
    width: 18px; height: 18px; flex: none;
    border: 2px solid var(--rule); border-top-color: var(--ink);
    border-radius: 50%; animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .progress-spinner { animation-duration: 3s; } }
.progress-body { flex: 1; min-width: 0; }
.progress-text { font-size: 14px; font-weight: 500; color: var(--ink); }
.progress-rail {
    height: 3px; border-radius: 2px; background: var(--rule-soft); margin-top: 9px; overflow: hidden;
}
.progress-fill {
    height: 100%; background: var(--ink); border-radius: 2px;
    transition: width .25s ease;
}
.progress-sub { font-family: var(--mono); font-size: 12px; color: var(--ink-3); margin-top: 2px; }

/* ---------- section labels ---------- */
.section-label {
    font-size: 12px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--ink-3); margin: 28px 0 2px;
}

/* ---------- tables ---------- */
.gradio-container [data-testid="dataframe"] {
    border: 1px solid var(--rule) !important; border-radius: 10px !important;
    background: var(--surface) !important; overflow: hidden !important;
}
.gradio-container .dataframe, .gradio-container .dataframe *,
.gradio-container table:not(.data-table), .gradio-container table:not(.data-table) * {
    font-family: var(--sans) !important;
}
.gradio-container table:not(.data-table) thead th {
    background: var(--surface) !important; color: var(--ink-3) !important;
    font-size: 11.5px !important; font-weight: 500 !important;
    letter-spacing: 0.09em !important; text-transform: uppercase !important;
    border-bottom: 1px solid var(--rule) !important; padding: 12px 14px !important;
    position: sticky !important; top: 0 !important; z-index: 5 !important;
}
.gradio-container table:not(.data-table) tbody td {
    color: var(--ink-2) !important; font-size: 14px !important;
    border-bottom: 1px solid var(--rule-soft) !important; padding: 11px 14px !important;
}
.gradio-container table:not(.data-table) tbody tr:hover td { background: var(--rule-soft) !important; }
.gradio-container table:not(.data-table) tbody td:nth-child(1),
.gradio-container table:not(.data-table) tbody td:nth-child(3),
.gradio-container table:not(.data-table) tbody td:nth-child(4),
.gradio-container table:not(.data-table) tbody td:nth-child(5) {
    font-family: var(--mono) !important; font-variant-numeric: tabular-nums !important;
    color: var(--ink) !important;
}
.gradio-container table:not(.data-table) tbody td a { color: var(--ink) !important; text-decoration: none !important;
    border-bottom: 1px solid var(--rule) !important; }
.gradio-container table:not(.data-table) tbody td a:hover { border-bottom-color: var(--ink) !important; }

/* ---------- plots & panels ---------- */
.gradio-container .block, .gradio-container .form {
    background: transparent !important; border: none !important; box-shadow: none !important;
}
.gradio-container .block:has(.js-plotly-plot) > label.float { display: none !important; }
.gradio-container .plot-container, .gradio-container [data-testid="plot"] {
    background: var(--surface) !important; border: 1px solid var(--rule) !important;
    border-radius: 10px !important; padding: 8px !important;
}
.gradio-container .tab-nav button {
    color: var(--ink-3) !important; border: none !important; background: transparent !important;
    font-size: 13.5px !important; padding: 10px 2px !important; margin-right: 22px !important;
}
.gradio-container .tab-nav button.selected {
    color: var(--ink) !important; border-bottom: 2px solid var(--ink) !important;
}
.gradio-container .tab-nav { border-bottom: 1px solid var(--rule) !important; }

/* ---------- markdown tables in About ---------- */
.gradio-container .prose table:not(.data-table) { border-collapse: collapse; width: 100%; font-size: 14px; }
.gradio-container .prose th:not(.c-num):not(.c-rank) {
    background: var(--surface) !important; color: var(--ink-3) !important;
    text-align: left; font-size: 11.5px; letter-spacing: 0.09em; text-transform: uppercase;
    font-weight: 500; padding: 10px 14px; border-bottom: 1px solid var(--rule) !important;
}
.gradio-container .prose table:not(.data-table) td { padding: 10px 14px; border-bottom: 1px solid var(--rule-soft) !important; color: var(--ink-2); }

/* The status slots stay mounted so the progress bar has an anchor; collapse
   them to nothing while they hold no message. */
.gradio-container .status-slot:not(:has(.notice)) { min-height: 0 !important; }
.gradio-container .status-slot:not(:has(.notice)) .html-container { padding: 0 !important; }

/* ---------- candidate picker ---------- */
/* Each option is a two-line card: name and ORCID above, affiliation, dates and
   counts below. Radio rather than a table so nothing truncates in a narrow pane. */
.gradio-container .picker { border: none !important; background: transparent !important; }
.gradio-container .picker > label,
.gradio-container .picker .wrap { display: block !important; gap: 0 !important; }
.gradio-container .picker fieldset,
.gradio-container .picker .wrap {
    display: flex !important; flex-direction: column !important; gap: 6px !important;
}
.gradio-container .picker label {
    display: flex !important; align-items: flex-start !important; gap: 10px !important;
    padding: 12px 14px !important; margin: 0 !important;
    background: var(--surface) !important;
    border: 1px solid var(--rule) !important; border-radius: 8px !important;
    cursor: pointer; transition: border-color .12s ease, background .12s ease;
    white-space: pre-line !important;
    font-size: 13.5px !important; line-height: 1.45 !important;
    color: var(--ink-2) !important; font-weight: 400 !important;
}
.gradio-container .picker label:hover { background: var(--rule-soft) !important; }
.gradio-container .picker label:has(input:checked) {
    border-color: var(--ink) !important; box-shadow: inset 0 0 0 1px var(--ink) !important;
    background: var(--surface) !important;
}
.gradio-container .picker label input[type="radio"] { margin-top: 3px !important; accent-color: var(--ink); }
.gradio-container .picker label span { white-space: pre-line !important; }
/* first line of the label reads as the name */
.gradio-container .picker label span::first-line {
    font-weight: 600; color: var(--ink); font-size: 14.5px;
}

/* ---------- comparison list ---------- */
.chips { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 12px 0 2px; }
.chip {
    display: inline-flex; align-items: baseline; gap: 8px;
    padding: 7px 14px; border: 1px solid var(--ink); border-radius: 999px;
    background: var(--surface); font-size: 13.5px;
}
.chip b { color: var(--ink); font-weight: 600; }
.chip-meta { font-size: 12px; color: var(--ink-3); }
.chip-note { font-family: var(--mono); font-size: 11.5px; color: var(--ink-3); margin-left: 4px; }

.setting-note { font-size: 12.5px; color: var(--ink-3); margin: 2px 2px 12px; max-width: 76ch; line-height: 1.5; }

/* ---------- exclude-a-paper list ---------- */
.gradio-container .exclude-list { border: none !important; background: transparent !important; }
.gradio-container .exclude-list label {
    display: flex !important; align-items: flex-start !important; gap: 10px !important;
    padding: 9px 12px !important; margin: 0 0 4px !important;
    background: var(--surface) !important;
    border: 1px solid var(--rule) !important; border-radius: 6px !important;
    font-size: 13px !important; line-height: 1.45 !important;
    color: var(--ink-2) !important; font-weight: 400 !important; cursor: pointer;
}
.gradio-container .exclude-list label:hover { background: var(--rule-soft) !important; }
.gradio-container .exclude-list label:has(input:checked) {
    border-color: var(--warm) !important; background: #FFF7F3 !important;
    text-decoration: line-through; color: var(--ink-3) !important;
}
.gradio-container .exclude-list input[type="checkbox"] { margin-top: 2px !important; accent-color: var(--warm); }

/* ---------- comparison insights ---------- */
.insights { margin-top: 20px; }
.insight {
    background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
    padding: 18px 22px; margin-bottom: 12px;
}
.insight h4 {
    font-size: 14.5px; font-weight: 600; color: var(--ink); margin: 0 0 12px;
}
.insight > p {
    font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--ink-3); margin: 14px 0 4px;
}
.insight > p:first-of-type { margin-top: 0; }
.insight ul { margin: 0; padding-left: 18px; }
.insight li { font-size: 13.5px; color: var(--ink-2); margin: 4px 0; line-height: 1.5; }
.insight li b { color: var(--ink); font-weight: 600; }
.ins-num { font-family: var(--mono); color: var(--ink); }
.ins-caveat {
    font-size: 12.5px; color: var(--ink-3); margin: 4px 2px 0; max-width: 78ch; line-height: 1.5;
}

/* ---------- data tables ---------- */
.table-card {
    background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
    padding: 4px 6px 2px; overflow-x: auto; margin-bottom: 6px;
}
table.data-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
table.data-table th {
    text-align: left; vertical-align: bottom;
    font-size: 11px; font-weight: 500; letter-spacing: 0.07em; text-transform: uppercase;
    color: var(--ink-3); padding: 12px 12px 9px; border-bottom: 1px solid var(--rule);
    white-space: normal; line-height: 1.3;
}
table.data-table th.c-num { text-align: right; }
table.data-table th.c-rank { width: 38px; }
table.data-table td {
    padding: 10px 12px; border-bottom: 1px solid var(--rule-soft);
    color: var(--ink-2); vertical-align: top;
}
table.data-table tr:last-child td { border-bottom: none; }
table.data-table tr:hover td { background: var(--rule-soft); }
table.data-table td.num {
    font-family: var(--mono); font-variant-numeric: tabular-nums;
    text-align: right; color: var(--ink); white-space: nowrap;
}
table.data-table td.num.strong { font-weight: 500; }
table.data-table td.rank { font-family: var(--mono); font-size: 11.5px; color: var(--ink-3); }
table.data-table td.title { color: var(--ink); }
table.data-table td.title a {
    color: var(--ink); text-decoration: underline;
    text-decoration-color: var(--rule); text-underline-offset: 2px;
}
table.data-table td.title a:hover { text-decoration-color: var(--ink); }

/* Below this width five columns cannot coexist: the title wraps to seven lines
   and the last column falls off the edge. Stack each row instead, with every
   figure announcing which measure it is. */
@media (max-width: 720px) {
    table.data-table thead { display: none; }
    table.data-table, table.data-table tbody, table.data-table tr { display: block; width: 100%; }
    table.data-table td { display: inline-block; border: none; padding: 2px 0; }
    table.data-table tr {
        padding: 12px 10px; border-bottom: 1px solid var(--rule-soft);
    }
    table.data-table tr:hover td { background: transparent; }
    table.data-table td.rank { display: inline-block; margin-right: 8px; }
    table.data-table td.title { display: inline; font-size: 14px; }
    table.data-table td.num {
        text-align: left; margin: 6px 18px 0 0; white-space: nowrap;
    }
    table.data-table td.num::before {
        content: attr(data-label);
        font-family: var(--sans); font-size: 10.5px; letter-spacing: 0.07em;
        text-transform: uppercase; color: var(--ink-3); margin-right: 6px;
    }
}

/* ---------- responsive ---------- */
@media (max-width: 860px) {
    .track { grid-template-columns: 1fr 56px; gap: 6px 14px; }
    .track-rail { grid-column: 1 / 3; order: 3; }
    .profile { padding: 22px 18px 18px; }
}

footer, footer * { color: var(--ink-3) !important; font-family: var(--sans) !important; }
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.gray,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
)

def create_interface():
    with gr.Blocks(title="Fieldtrip Index") as demo:
        cache_dir = gr.State(value=create_session_cache_dir)
        selected_author_id = gr.State(value="")

        gr.HTML(f"""
        <div class="masthead">
            <span class="masthead-logo">{MARK_IMG}</span>
            <span class="masthead-mark">Fieldtrip Index</span>
            <span class="masthead-note">via <a href="https://openalex.org" target="_blank" rel="noopener">openalex</a></span>
        </div>
        <p class="masthead-lede">Measures how far a researcher's work travels across
        disciplinary boundaries &mdash; who cites it, what it draws on, and which fields
        it reaches that it never cites back.</p>
        """)

        with gr.Tabs():
            with gr.Tab("Analyse"):
                with gr.Row():
                    with gr.Column(scale=5):
                        author_input = gr.Textbox(
                            label="Researcher",
                            placeholder="Name, ORCID, or OpenAlex ID — e.g. Yoshua Bengio",
                        )
                    with gr.Column(scale=1, min_width=130):
                        search_btn = gr.Button("Search", elem_classes=["primary-btn"])

                author_radio = gr.Radio(
                    choices=[], label="Which profile?", visible=False,
                    elem_classes=["picker"], container=False,
                )
                with gr.Row():
                    analyze_btn = gr.Button("Run analysis", elem_classes=["primary-btn"], visible=False)
                    refresh_btn = gr.Button("Refetch data", elem_classes=["ghost-btn"], visible=False)

                # Always mounted, empty when idle: Gradio needs a visible output
                # component to draw the progress bar on, and on a first run every
                # other result slot is still hidden.
                progress_html = gr.HTML(visible=True, elem_classes=["status-slot"])
                results_html = gr.HTML(visible=False)

                with gr.Column(visible=False) as charts_group:
                    gr.HTML('<div class="section-label">Where the work travels</div>')
                    with gr.Row():
                        ref_diversity_plot = gr.Plot(label=None)
                        field_breakdown_plot = gr.Plot(label=None)
                    # Both of these need the full width: the flow chart carries
                    # long field names, and the matrix grows with paper count.
                    with gr.Row():
                        bridge_plot = gr.Plot(label=None)
                    with gr.Row():
                        dispersion_plot = gr.Plot(label=None)
                    gr.HTML('<div class="section-label">Distribution and drift</div>')
                    with gr.Row():
                        scatter_plot = gr.Plot(label=None)
                        kde_plot = gr.Plot(label=None)
                    with gr.Row():
                        keywords_plot = gr.Plot(label=None)

                top_papers_table = gr.HTML(visible=False)

                with gr.Accordion("Not their paper? Exclude it",
                                  open=False, visible=False) as exclude_accordion:
                    gr.HTML(
                        '<p class="setting-note">OpenAlex sometimes files another '
                        'person\'s work under this record. A misattributed paper affects '
                        'every measure at once — its text, its references and its '
                        'audience all count. Tick anything that is not theirs and '
                        'recalculate; this reuses the data already fetched, so it is '
                        'immediate.</p>'
                    )
                    exclude_group = gr.CheckboxGroup(
                        choices=[], value=[], label=None, show_label=False,
                        elem_classes=["exclude-list"], container=False,
                    )
                    with gr.Row():
                        recalc_btn = gr.Button("Recalculate without these",
                                               elem_classes=["primary-btn"])
                        restore_btn = gr.Button("Put them all back",
                                                elem_classes=["ghost-btn"])
                    # Mirrors the main status slot. Recalculating is triggered from
                    # the bottom of a very long page, and the other slot is far out
                    # of view up there.
                    exclude_status = gr.HTML(visible=True, elem_classes=["status-slot"])

                with gr.Accordion("Full paper detail", open=False, visible=False) as papers_accordion:
                    papers_table = gr.Dataframe(max_height=420)

                html_download = gr.File(label="Download report (HTML)", visible=False)
                methodology_md = gr.Markdown(visible=False)

            with gr.Tab("Compare"):
                gr.HTML(
                    '<p class="masthead-lede">These measures carry no absolute scale, so they '
                    'mean most set against each other. Search for each researcher, pick the '
                    'right profile, and add up to three.</p>'
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        compare_search_box = gr.Textbox(
                            label="Add a researcher",
                            placeholder="Name, ORCID, or OpenAlex ID",
                        )
                    with gr.Column(scale=1, min_width=130):
                        compare_search_btn = gr.Button("Search", elem_classes=["primary-btn"])

                compare_candidates_state = gr.State([])
                compare_list_state = gr.State([])

                compare_radio = gr.Radio(
                    choices=[], label="Which profile?", visible=False,
                    elem_classes=["picker"], container=False,
                )
                with gr.Row():
                    compare_add_btn = gr.Button("Add to comparison",
                                                elem_classes=["primary-btn"], visible=False)
                    compare_clear_btn = gr.Button("Clear list",
                                                  elem_classes=["ghost-btn"], visible=False)

                compare_chips = gr.HTML(visible=False)
                compare_btn = gr.Button("Compare", elem_classes=["primary-btn"], visible=False)
                compare_status = gr.HTML(visible=True, elem_classes=["status-slot"])
                compare_plot = gr.Plot(label=None, visible=False)
                compare_table = gr.HTML(visible=False)

            with gr.Tab("About"):
                gr.Markdown(
                    latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                      {"left": "$", "right": "$", "display": False}],
                    value=r"""
## What this measures

Some research stays inside one field. Some of it draws on several, or gets picked
up by people the author never reads. This tool puts four numbers on that, from a
researcher's **25 most-cited papers** with an abstract and a DOI.

Two of the measures compare *text*, using an embedding model that turns each
abstract into a vector. Two compare *fields*, using OpenAlex's subject
classification. None of them is a grade — see **Reading the numbers** at the end.

---

## How one analysis works

| Step | Requests |
|---|---|
| Fetch the 25 most-cited papers, with abstracts and reference lists | 1 |
| For each paper, sample up to 50 works citing it, with their abstracts and fields | 25 |
| Look up the field of every referenced work, de-duplicated, 50 ids per request | ~10 |
| Check the author record describes one person | 1 |

Citing works are drawn with a **seeded random sample**, not "most recent" — the
recent slice of a well-cited paper reflects whatever is currently fashionable.
The same seed returns the same sample, so a rerun reproduces the result exactly.
**Self-citations are excluded.**

---

## The measure of similarity

Everything text-based rests on one operation. Each abstract becomes a vector
$\mathbf{e}$, and the closeness of two abstracts is the cosine of the angle
between their vectors:

$$\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}$$

It runs from 1 (same direction) through 0 (unrelated). **Distance** is
$1-\cos$. Two papers on the same topic might sit at 0.7 similarity, two on
unrelated topics at 0.2.

---

## 1 · External diversity — how far your work travels

**The question:** are the people citing you working on the same thing you are?

For one paper $p$ with citing works $C_p$, take the average similarity between
your abstract and theirs, then flip it so that *far* scores *high*:

$$E_p = 100 \times \left(1 - \frac{1}{|C_p|}\sum_{c \in C_p} \cos(\mathbf{e}_p, \mathbf{e}_c)\right)$$

The score for the researcher is the mean of $E_p$ over their papers.

**Worked example.** A paper's 50 citing works have similarities averaging 0.52.
Then $E_p = 100 \times (1 - 0.52) = 48$. Had the citing work been closer in
subject, averaging 0.70, the paper would score 30.

**Reading it:** high means your readers are working on something other than what
you wrote. Low means the conversation around your paper stays close to it.

---

## 2 · Internal diversity — how far your own work ranges

**The question:** do your own papers resemble each other?

Every pair of your papers, averaged:

$$I = 100 \times \frac{2}{n(n-1)}\sum_{i \lt j}\bigl(1 - \cos(\mathbf{e}_i,\mathbf{e}_j)\bigr)$$

With 25 papers that is 300 pairs. The **Paper similarity** heatmap shows the
whole matrix; two dark blocks with a pale gap between them mean two separate
strands of work.

**This measure needs no citations at all**, so papers nothing has cited yet still
count — restricting it to cited work would narrow it for no reason.

**Reading it:** high means the papers cover different ground. But note that
embedding distance partly reflects *writing*, not only subject: a methods paper
and a theory paper in one field can sit far apart.

---

## 3 · Reference diversity — how widely you read

**The question:** do the works you cite come from many fields, evenly, and are
those fields far apart?

Counting fields is not enough. Citing Medicine and Nursing is not the same as
citing Medicine and Astronomy, yet a plain count — or Shannon entropy — scores
them identically. So this uses **Rao–Stirling diversity**, which folds in how far
apart the fields are:

$$RS = \sum_{i \neq j} d_{ij}\, p_i\, p_j$$

- $p_i$ — the share of your references in field $i$
- $d_{ij}$ — the distance between fields $i$ and $j$

Distance comes from OpenAlex's hierarchy, which nests 26 fields inside 4 domains
(Health Sciences, Life Sciences, Physical Sciences, Social Sciences):

| Relationship | $d_{ij}$ | Example |
|---|---|---|
| Same field | 0 | Medicine – Medicine |
| Different field, same domain | 0.5 | Medicine – Nursing |
| Different domain | 1.0 | Medicine – Physics and Astronomy |

**Worked example.** Two reference lists, both split exactly in half, so both have
identical entropy and both span "two fields":

- 50% Medicine, 50% Nursing → both in Health Sciences → $RS = 2(0.5)(0.5)(0.5) = 0.25$ → **25**
- 50% Medicine, 50% Astronomy → different domains → $RS = 2(0.5)(0.5)(1.0) = 0.50$ → **50**

The second reads twice as diverse, which is the point. A count or an entropy
would have called them the same.

**Real profiles.** A deep-learning researcher whose references are 88% Computer
Science, everything else scattered, scores about **18**. A biostatistician
splitting 44% Medicine and 32% Mathematics — two different domains — scores
about **65**.

**Effective fields.** Alongside the score the profile reports
$\exp(H)$, where $H = -\sum_i p_i \ln p_i$ — the number of *equally-used* fields
that would produce the same spread. Four fields used evenly gives 4.0; 88% in one
field with a long tail gives about 1.9. It is easier to read than the raw score.

---

## 4 · Bridge — who reads you that you don't read

**The question:** how much of the attention your work gets comes from fields your
own reading does not explain?

For each field, compare its share of the work **citing** you against its share of
the works **you cite**, and keep only the excess:

$$B = 100 \times \sum_{f} \max\bigl(0,\; a_f - s_f\bigr)$$

- $a_f$ — field $f$'s share of the citing works
- $s_f$ — field $f$'s share of your references

This is the total variation distance between the two distributions, taken in one
direction. It is bounded 0–100 and reads as *the share of your audience your own
reading does not account for*.

**Worked example.** A computer scientist:

| Field | Share of references $s_f$ | Share of citers $a_f$ | $\max(0, a_f - s_f)$ |
|---|---|---|---|
| Computer Science | 88.3% | 66.2% | 0 — you cite it more than it cites you |
| Engineering | 1.4% | 14.6% | **0.132** |
| Everything else | 10.3% | 19.2% | **0.089** |

$B = 100 \times (0 + 0.132 + 0.089) = 22$.

Engineering carries most of it: they cite this author ten times more than the
author cites them.

**Why not simply count fields that cite you but that you never cite?** Because
one stray citation from an unrelated field would then count as much as four
hundred. Weighting by volume, a single paper out of 500 moves the score by 0.2
rather than by several points. And because it is continuous, a field you cite a
*little* but that cites you a *lot* still counts, in proportion — there is no
threshold to fall the wrong side of.

---

## The Fieldtrip Index

The single number is the **Fieldtrip Index**. The four measures pair into two axes. **Range** is what the researcher themselves does;
**reach** is how far the work travels.

$$\text{Range} = \frac{I + RS}{2} \qquad \text{Reach} = \frac{E + B}{2}$$

$$\text{Fieldtrip Index} = \sqrt{\text{Range} \times \text{Reach}}$$

Each axis pairs one text measure with one field measure, so the two inputs come
from different sources and the shared signal is not counted twice. Inside an axis
the mean is arithmetic, because bridge is legitimately zero for a focused
researcher and should not zero out the axis. Across the axes it is geometric on
purpose: a researcher with range 50 and reach 10 scores 22, not 30, because
interdisciplinarity needs both. Prefer the four-part profile; the pair of axes says
more than the product.

---

## Is the profile one person?

OpenAlex assigns author identifiers algorithmically, so a record can hold several
people's work, or one person's work can be split across records. A merged record
is the dangerous case: unrelated papers read as range, so **all four measures rise
at once** and the score peaks exactly when the data are worst.

Before analysing, one request samples the record and checks:

| Signal | Weight |
|---|---|
| Share of co-authors appearing on more than one paper | primary |
| Number of distinct institutions across the work | primary |
| Two or more conflicting ORCIDs on one record | conclusive |
| Bylines naming more than one person | corroborating |
| Implausible career span | corroborating |
| Two field communities sharing almost no co-authors | corroborating |

Co-author cohesion separates real breadth from a name collision: a genuine
polymath keeps a recurring core of collaborators across fields; two people who
share a name do not. Corroborating signals never warn alone — each has an honest
explanation — so a warning needs a primary signal beside it.

Measured on real records: clean profiles run 26–43% co-author repeat across 11–29
institutions; a known merged record showed 8% across 65.

---

## Reading the numbers

**There is no absolute scale here, and the interface deliberately shows no
grades.** Any two scientific abstracts share a great deal of ordinary English, so
the cosine-based measures rarely approach zero even for tightly focused work.
A 50 does not mean "half as interdisciplinary as possible" — on its own it means
very little.

These numbers earn their keep by **comparison**: one researcher against another
analysed the same way, or the same researcher at two points in a career. That is
what the Compare tab is for.

Other limits worth holding in mind:

- Only the 25 most-cited papers are analysed, which favours older, established
  work. Internal and reference diversity therefore describe the spread of a
  researcher's most-cited output, not necessarily of everything they have written.
- External diversity and bridge need papers that have been cited. Where a paper's
  citing works carry no abstracts it is set aside, and the profile says how many.
- Field distance uses OpenAlex's four-domain hierarchy, which is coarse: it cannot
  tell that two Physical Sciences fields are further apart than two others.
- Embedding distance partly reflects writing style and venue conventions, not only
  subject matter.
- Coverage is reported, not hidden: works OpenAlex has not classified are counted
  and shown as a percentage beneath the profile.

**Data source:** [OpenAlex](https://openalex.org/) · **Embeddings:**
`minishlab/potion-base-32M` via Sentence Transformers
""")


        async def resolve_candidates(query: str) -> tuple[list[dict], str]:
            """Look up a query; return (candidates, error_html)."""
            if not (query or "").strip():
                return [], ""
            try:
                found = await search_authors(query, limit=6)
            except InvalidAPIKey as e:
                return [], notice("OpenAlex rejected the API key", str(e), kind="warn")
            except Throttled as e:
                return [], notice("OpenAlex is throttling requests", str(e), kind="warn")
            except Exception as e:
                logger.error(f"Author search failed: {e}")
                return [], notice("Search failed",
                                  "Could not reach OpenAlex. Check your connection and try again.",
                                  kind="warn")
            if not found:
                return [], notice("No matches",
                                  f"Nothing in OpenAlex matches &ldquo;{query}&rdquo;. "
                                  "Try a fuller name, or paste an ORCID.")
            return found, ""

        def same_name_banner(candidates: list[dict]) -> str:
            twins = [c for c in candidates[1:]
                     if c["name"].lower() == candidates[0]["name"].lower()]
            if not twins:
                return ""
            stranded = sum(c["works_count"] for c in twins)
            return notice(
                "More than one profile carries this name",
                f"{len(twins)} other {'profile holds' if len(twins) == 1 else 'profiles hold'} "
                f"about {stranded:,} further works. OpenAlex may have split one person across "
                "several records, or merged several people into one. Check the affiliation and "
                "dates before choosing.", kind="warn")

        async def search_and_show_authors(query):
            candidates, problem = await resolve_candidates(query)
            if not candidates:
                return (gr.update(choices=[], visible=False), gr.update(visible=False),
                        gr.update(visible=False), "",
                        gr.update(value=problem, visible=bool(problem)))
            options = candidate_choices(candidates)
            return (gr.update(choices=options, value=options[0][1], visible=True),
                    gr.update(visible=True), gr.update(visible=True),
                    candidates[0]["id"],
                    gr.update(value=same_name_banner(candidates),
                              visible=bool(same_name_banner(candidates))))

        BLANK_RESULTS = 18



        def fail(title: str, body: str):
            out = [gr.update() for _ in range(BLANK_RESULTS)]
            block = notice(title, body, kind="warn")
            out[0] = gr.update(value=block, visible=True)
            out[-1] = gr.update(value=block, visible=True)
            return out

        def stage(fraction, message):
            """One frame of the run: only the status slots change.

            Both are updated because the run can be started from either end of
            the page and the feedback has to appear where the click happened.
            """
            out = [gr.update() for _ in range(BLANK_RESULTS)]
            block = progress_block(fraction, message)
            out[0] = gr.update(value=block, visible=True)
            out[-1] = gr.update(value=block, visible=True)
            return out

        async def _analyse(author_id, cache_dir_val, force_refresh, progress, exclude_ids=None):
            if not author_id:
                yield fail("Pick a researcher first",
                           "Search above, then choose the profile you want to analyse.")
                return

            yield stage(0.02, "Looking up the author")
            coherence = {"verdict": "unknown"}
            try:
                async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
                    data = await fetch_with_retry(client, f"{OPENALEX_BASE_URL}/authors/{author_id}")
                    author_name = (data or {}).get("display_name") or "Unknown"
                    # One extra call, before anything expensive: does this record
                    # plausibly describe a single person? A merged record raises
                    # every measure at once, so this has to be settled first.
                    yield stage(0.05, "Checking the profile holds one person")
                    coherence = assess_coherence(await fetch_author_signals(author_id, client))
                    if coherence["verdict"] != "ok":
                        logger.warning(f"Coherence {coherence['verdict']} for {author_id}: "
                                       f"{'; '.join(coherence.get('flags', [])) or 'no detail'}")
            except InvalidAPIKey as e:
                yield fail("OpenAlex rejected the API key", str(e)); return
            except Throttled as e:
                yield fail("OpenAlex is throttling requests", str(e)); return
            except RateLimitExceeded as e:
                yield fail("OpenAlex daily limit reached", str(e)); return
            except Exception as e:
                logger.error(f"Author lookup failed: {e}")
                yield fail("Could not reach OpenAlex",
                           "The lookup failed before the analysis started. Try again in a moment.")
                return

            if force_refresh:
                clear_author_cache(author_id, cache_dir_val)

            # The analysis runs as a task while this loop reports what it is
            # doing; a plain await would leave the interface silent for its
            # whole duration.
            live = {"frac": 0.08, "msg": f"Reading {author_name}"}

            def note(frac, desc=""):
                live["frac"], live["msg"] = frac, desc or live["msg"]

            task = asyncio.create_task(
                analyze_author(author_id, author_name, cache_dir_val, progress_cb=note,
                               exclude_ids=set(exclude_ids or [])))
            while not task.done():
                yield stage(live["frac"], live["msg"])
                await asyncio.sleep(0.2)

            try:
                out = task.result()
            except InvalidAPIKey as e:
                yield fail("OpenAlex rejected the API key", str(e)); return
            except Throttled as e:
                yield fail("OpenAlex is throttling requests", str(e)); return
            except RateLimitExceeded as e:
                yield fail("OpenAlex daily budget used up", str(e)); return
            except NotEnoughData as e:
                yield fail("Not enough indexed work to analyse", str(e)); return
            except Exception as e:
                logger.exception("Analysis failed")
                yield fail("The analysis did not finish",
                           f"Something went wrong partway through: {type(e).__name__}. "
                           "The console log has the detail.")
                return

            (df_report, profile_html, df_top, scatter, kde_fig, dispersion_fig,
             ref_fig, bridge_fig, field_fig, keyword_fig, explanation, html_path,
             _metrics, detail) = out

            # The warning belongs above the numbers it qualifies, not beside them.
            profile_html = coherence_notice(coherence) + profile_html

            yield [
                gr.update(value="", visible=True),              # status slot: cleared
                gr.update(value=profile_html, visible=True),    # results_html
                gr.update(value=df_top, visible=True),          # top_papers_table
                bridge_fig, dispersion_fig, ref_fig, field_fig, # charts row 1-2
                scatter, kde_fig, keyword_fig,                  # charts row 3-4
                df_report,                                      # papers_table
                gr.update(value=explanation, visible=True),     # methodology
                gr.update(value=html_path, visible=True),       # download
                gr.update(visible=True),                        # charts_group
                gr.update(visible=True),                        # papers_accordion
                # the checkbox list is rebuilt from the papers that survived, and
                # keeps whatever the user had already ticked
                gr.update(choices=detail["paper_choices"],
                          value=list(exclude_ids or []), visible=True),
                gr.update(visible=True),                        # exclude_accordion
                gr.update(value="", visible=True),              # mirrored status: cleared
            ]

        async def run_analysis(author_id, cache_dir_val):
            async for frame in _analyse(author_id, cache_dir_val, False, None):
                yield frame

        async def rerun_analysis(author_id, cache_dir_val):
            async for frame in _analyse(author_id, cache_dir_val, True, None):
                yield frame

        async def recalculate(author_id, cache_dir_val, excluded):
            # No refetch: the cache already holds every paper, so dropping some
            # is only a recomputation.
            async for frame in _analyse(author_id, cache_dir_val, False, None, excluded):
                yield frame

        async def restore_all(author_id, cache_dir_val):
            async for frame in _analyse(author_id, cache_dir_val, False, None, None):
                yield frame

        async def compare_search(query):
            candidates, problem = await resolve_candidates(query)
            if not candidates:
                return (gr.update(choices=[], visible=False), [],
                        gr.update(value=problem or "", visible=True),
                        gr.update(visible=False))
            options = candidate_choices(candidates)
            return (gr.update(choices=options, value=options[0][1], visible=True),
                    candidates, gr.update(value="", visible=True), gr.update(visible=True))

        def render_chips(chosen: list[dict]) -> str:
            if not chosen:
                return ""
            chips = "".join(
                f'<span class="chip"><b>{c["name"]}</b>'
                f'<span class="chip-meta">{c.get("institution") or c["id"]}</span></span>'
                for c in chosen)
            room = MAX_COMPARE - len(chosen)
            tail = (f'<span class="chip-note">room for {room} more</span>' if room
                    else '<span class="chip-note">list full</span>')
            return f'<div class="chips">{chips}{tail}</div>'

        def compare_add(picked_id, candidates, chosen):
            chosen = list(chosen or [])
            picked = next((c for c in (candidates or []) if c["id"] == picked_id), None)
            if not picked:
                return chosen, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            if any(c["id"] == picked["id"] for c in chosen):
                return (chosen,
                        gr.update(value=render_chips(chosen) +
                                  notice("Already on the list",
                                         f"{picked['name']} is in the comparison."), visible=True),
                        gr.update(visible=True), gr.update(visible=len(chosen) >= 2))
            if len(chosen) >= MAX_COMPARE:
                return (chosen,
                        gr.update(value=render_chips(chosen) +
                                  notice("List is full",
                                         f"Comparisons hold {MAX_COMPARE} researchers. "
                                         "Clear the list to start again."), visible=True),
                        gr.update(visible=True), gr.update(visible=True))
            chosen.append({"id": picked["id"], "name": picked["name"],
                           "institution": picked.get("institution")})
            return (chosen, gr.update(value=render_chips(chosen), visible=True),
                    gr.update(visible=True), gr.update(visible=len(chosen) >= 2))

        def compare_clear():
            return ([], gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(value="", visible=True),
                    gr.update(visible=False), gr.update(visible=False))

        async def run_comparison(chosen, cache_dir_val):
            hide = (gr.update(visible=False), gr.update(visible=False))

            def problem(title, body):
                return (gr.update(value=notice(title, body, kind="warn"), visible=True), *hide)

            if not chosen or len(chosen) < 2:
                yield problem("Not enough to compare", "Add at least two researchers.")
                return

            # Same reporting loop as the single analysis: a comparison runs one
            # full analysis per researcher, so awaiting it silently would leave
            # the interface blank for the better part of a minute.
            live = {"frac": 0.0, "msg": "Starting"}

            def note(frac, msg):
                live["frac"], live["msg"] = frac, msg

            task = asyncio.create_task(
                analyse_for_comparison(chosen, cache_dir_val, on_step=note))
            while not task.done():
                yield (gr.update(value=progress_block(live["frac"], live["msg"]), visible=True), *hide)
                await asyncio.sleep(0.2)

            try:
                entries, problems = task.result()
            except (RateLimitExceeded, Throttled, InvalidAPIKey) as e:
                yield problem("OpenAlex is unavailable right now", str(e))
                return
            except Exception as e:
                logger.exception("Comparison failed")
                yield problem("The comparison did not finish",
                              f"Something went wrong: {type(e).__name__}.")
                return

            if len(entries) < 2:
                yield problem("Could not build a comparison",
                              "; ".join(problems) if problems else
                              "not enough resolved to compare.")
                return

            yield (gr.update(
                       value=notice("Some entries were skipped", "; ".join(problems))
                       if problems else "", visible=True),
                   gr.update(value=create_comparison_chart(entries), visible=True),
                   gr.update(value=comparison_table_html(entries)
                                   + comparison_insights(entries), visible=True))

        RESULT_SLOTS = [progress_html, results_html, top_papers_table,
                        bridge_plot, dispersion_plot, ref_diversity_plot, field_breakdown_plot,
                        scatter_plot, kde_plot, keywords_plot,
                        papers_table, methodology_md, html_download,
                        charts_group, papers_accordion,
                        exclude_group, exclude_accordion, exclude_status]

        SEARCH_SLOTS = [author_radio, analyze_btn, refresh_btn,
                        selected_author_id, progress_html]

        search_btn.click(search_and_show_authors, [author_input], SEARCH_SLOTS,
                         show_progress="minimal", show_progress_on=[search_btn])
        author_input.submit(search_and_show_authors, [author_input], SEARCH_SLOTS,
                            show_progress="minimal", show_progress_on=[search_btn])
        author_radio.change(lambda v: v or "", [author_radio], [selected_author_id])

        analyze_btn.click(run_analysis, [selected_author_id, cache_dir],
                          RESULT_SLOTS, show_progress="hidden")
        refresh_btn.click(rerun_analysis, [selected_author_id, cache_dir],
                          RESULT_SLOTS, show_progress="hidden")
        recalc_btn.click(recalculate, [selected_author_id, cache_dir, exclude_group],
                         RESULT_SLOTS, show_progress="hidden")
        restore_btn.click(restore_all, [selected_author_id, cache_dir],
                          RESULT_SLOTS, show_progress="hidden")

        COMPARE_SEARCH_SLOTS = [compare_radio, compare_candidates_state,
                                compare_status, compare_add_btn]
        compare_search_btn.click(compare_search, [compare_search_box], COMPARE_SEARCH_SLOTS,
                                 show_progress="minimal", show_progress_on=[compare_search_btn])
        compare_search_box.submit(compare_search, [compare_search_box], COMPARE_SEARCH_SLOTS,
                                  show_progress="minimal", show_progress_on=[compare_search_btn])
        compare_add_btn.click(compare_add,
                              [compare_radio, compare_candidates_state, compare_list_state],
                              [compare_list_state, compare_chips, compare_clear_btn, compare_btn])
        compare_clear_btn.click(compare_clear, None,
                                [compare_list_state, compare_chips, compare_clear_btn,
                                 compare_btn, compare_status, compare_plot, compare_table])
        # Every output here starts hidden, so without an explicit anchor there
        # was nothing for Gradio to draw progress on at all.
        compare_btn.click(run_comparison, [compare_list_state, cache_dir],
                          [compare_status, compare_plot, compare_table],
                          show_progress="hidden")

    return demo

if __name__ == "__main__":
    sweep_stale_session_caches()
    # Free single-entity fetch; the response reveals which budget applied, which
    # is the only reliable way to tell an accepted key from an ignored one.
    log_api_key_status(asyncio.run(verify_api_key()))
    create_interface().launch(css=custom_css, theme=THEME, head=PAGE_HEAD)
