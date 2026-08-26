# reproducible_cache.py
"""
OpenAlex data layer: fetching, retry logic, and on-disk caching.

One analysis costs roughly 21 API calls:
  1  top papers (with their reference lists, in the same response)
 10  citing works, one call per paper (abstracts and topics included)
~10  reference metadata, batched 50 ids at a time and de-duplicated

Citing works are drawn with a seeded random sample rather than "most recent",
so a rerun of the same author with the same seed returns the same sample.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Optional
import httpx

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENALEX_BASE_URL = "https://api.openalex.org"

# OpenAlex meters usage as a daily budget: $0.10/day without a key, $1/day with
# a free one. Single-entity fetches are free, list+filter costs $0.10 per 1,000
# calls, and a search costs $1 per 1,000 — ten times a data call. Neither tier
# needs a payment method, so exhausting the budget returns 429 rather than a bill.
#
# The key is read from the environment and sent as a bearer token. It never
# belongs in source: this file is published, and a query parameter would put the
# key into server logs, browser history and referrer headers.
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()

HEADERS = {'User-Agent': 'InterdisciplinaryIndexApp/1.0'}
if OPENALEX_API_KEY:
    HEADERS['Authorization'] = f'Bearer {OPENALEX_API_KEY}'

# A 429 whose retry-after exceeds this is the daily budget running out, not
# momentary throttling; retrying would be pointless.
DAILY_BUDGET_SECONDS = 900
DEFAULT_CACHE_DIR = "cache"
os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)

# Bump when the cached record shape changes; older caches are rebuilt.
CACHE_SCHEMA = 2

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0
REQUEST_TIMEOUT = 30
BATCH_SIZE = 50           # OpenAlex accepts 50 ids per pipe-separated filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("OpenAlex API key: %s",
            "found in OPENALEX_API_KEY" if OPENALEX_API_KEY else
            "not set — anonymous access is limited to $0.10 of usage per day")

# ============================================================================
# EXCEPTIONS
# ============================================================================

class RateLimitExceeded(Exception):
    """Raised when the OpenAlex daily usage budget is exhausted."""
    pass


class NotEnoughData(Exception):
    """Raised when there is too little usable data to analyse an author."""
    pass


class Throttled(Exception):
    """Raised when OpenAlex keeps returning 429 after the retries are spent."""
    pass


class InvalidAPIKey(Exception):
    """Raised when OpenAlex rejects the configured key."""
    pass

# ============================================================================
# ABSTRACTS
# ============================================================================

def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Rebuild plain text from OpenAlex's inverted-index abstract format."""
    if not inverted_index:
        return None
    try:
        indices = [idx for val in inverted_index.values() for idx in val]
        if not indices:
            return None
        max_idx = min(max(indices), 10000)
        abstract = [""] * (max_idx + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                if 0 <= pos < len(abstract):
                    abstract[pos] = word
        result = " ".join(filter(None, abstract))
        return result if result.strip() else None
    except Exception as e:
        logger.error(f"Abstract reconstruction error: {e}")
        return None

# ============================================================================
# CACHE MANAGEMENT (Session-aware)
# ============================================================================

def get_cache_path(author_id: str, cache_dir: Optional[str] = None) -> str:
    """Get the file path for an author's cached data"""
    directory = cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"author_{author_id}.json")


def is_cached(author_id: str, cache_dir: Optional[str] = None, params: Optional[dict] = None) -> bool:
    """True only for a cache written by this schema under the same parameters.

    A cache built with a different seed, paper count, or self-citation setting
    describes a different sample, so it is treated as absent rather than reused.
    """
    path = get_cache_path(author_id, cache_dir)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if data.get("schema") != CACHE_SCHEMA:
        return False
    if params is not None and data.get("params") != params:
        return False
    return True


def get_cache_timestamp(author_id: str, cache_dir: Optional[str] = None) -> Optional[datetime]:
    """Get the timestamp when the cache was created"""
    cache_path = get_cache_path(author_id, cache_dir)
    if os.path.exists(cache_path):
        return datetime.fromtimestamp(os.path.getmtime(cache_path))
    return None


def save_author_cache(author_id: str, data: dict, cache_dir: Optional[str] = None):
    """Save author data to cache with timestamp"""
    data["cached_at"] = datetime.now().isoformat()
    with open(get_cache_path(author_id, cache_dir), "w") as f:
        json.dump(data, f)
    logger.info(f"Cached data for author {author_id}")


def load_author_cache(author_id: str, cache_dir: Optional[str] = None) -> dict:
    """Load author data from cache"""
    with open(get_cache_path(author_id, cache_dir), "r") as f:
        return json.load(f)


def clear_author_cache(author_id: str, cache_dir: Optional[str] = None) -> bool:
    """Clear cached data for an author (for refresh functionality)"""
    cache_path = get_cache_path(author_id, cache_dir)
    if os.path.exists(cache_path):
        os.remove(cache_path)
        logger.info(f"Cleared cache for author {author_id}")
        return True
    return False


def list_cached_authors(cache_dir: Optional[str] = None) -> List[str]:
    """List all cached author IDs"""
    directory = cache_dir or DEFAULT_CACHE_DIR
    if not os.path.exists(directory):
        return []
    return [f[7:-5] for f in os.listdir(directory)
            if f.startswith("author_") and f.endswith(".json")]


# ============================================================================
# API FUNCTIONS WITH RETRY LOGIC
# ============================================================================

async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: Optional[dict] = None,
    max_retries: int = MAX_RETRIES
) -> Optional[dict]:
    """
    Fetch data from API with exponential backoff retry logic.

    Raises:
        RateLimitExceeded: When the daily usage budget is exhausted
        Throttled: When 429s persist after the retries are spent
    """
    throttled = False
    for attempt in range(max_retries):
        try:
            r = await client.get(url, params=params, headers=HEADERS)

            if r.status_code == 429:
                # Read the reset from the header rather than matching on the
                # error text: the old code looked for "100000 requests per day",
                # a limit OpenAlex no longer has.
                try:
                    reset_in = int(r.headers.get("retry-after", 0))
                except (TypeError, ValueError):
                    reset_in = 0

                if reset_in > DAILY_BUDGET_SECONDS:
                    hours = reset_in / 3600
                    have_key = "Authorization" in HEADERS
                    advice = ("Requests are anonymous, which allows $0.10 of usage a day; "
                              "a free API key in OPENALEX_API_KEY raises that tenfold."
                              if not have_key else
                              "This account's daily budget is spent.")
                    logger.error(f"Daily budget exhausted; resets in {reset_in}s")
                    raise RateLimitExceeded(
                        f"OpenAlex's daily budget for this client is used up and resets in "
                        f"about {hours:.0f} hour{'s' if hours >= 1.5 else ''}. {advice}")

                wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                await asyncio.sleep(wait_time)
                throttled = True
                continue

            if r.status_code in (401, 403):
                raise InvalidAPIKey(
                    "OpenAlex rejected the API key in OPENALEX_API_KEY "
                    f"(HTTP {r.status_code}). Every request will fail until it is "
                    "corrected or the variable is unset; without a key the app still "
                    "works on the smaller anonymous allowance.")

            if r.status_code >= 500:
                wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(f"Server error {r.status_code}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                await asyncio.sleep(wait_time)
                continue

            r.raise_for_status()
            return r.json()

        except (RateLimitExceeded, InvalidAPIKey):
            raise

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for {url}")
            if attempt < max_retries - 1:
                await asyncio.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
            else:
                return None

        except httpx.RequestError as e:
            logger.error(f"Network error for {url}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
            else:
                return None

        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None

    if throttled:
        raise Throttled(
            "OpenAlex is rate-limiting this client. Wait a minute and try again; "
            "if it persists, the daily usage budget may be close to spent.")
    return None


async def fetch_works_by_ids(work_ids: List[str], client: httpx.AsyncClient,
                             select: str = "id,primary_topic") -> List[dict]:
    """Fetch many works by id, 50 per request via the pipe-separated id filter."""
    out = []
    for i in range(0, len(work_ids), BATCH_SIZE):
        batch = work_ids[i:i + BATCH_SIZE]
        data = await fetch_with_retry(
            client, f"{OPENALEX_BASE_URL}/works",
            params={"filter": f"openalex_id:{'|'.join(batch)}",
                    "select": select, "per_page": BATCH_SIZE},
        )
        if data and data.get("results"):
            out.extend(data["results"])
    return out


# ============================================================================
# CACHE BUILDING
# ============================================================================

def _topic_of(work: dict) -> dict | None:
    """Keep the whole topic hierarchy so the field level stays configurable."""
    topic = work.get("primary_topic")
    if not isinstance(topic, dict):
        return None
    out = {}
    for level in ("subfield", "field", "domain"):
        node = topic.get(level)
        if isinstance(node, dict) and node.get("display_name"):
            out[level] = node["display_name"]
    return out or None


async def fetch_citing_works(paper_id: str, author_id: str, limit: int,
                             client: httpx.AsyncClient, seed: int,
                             exclude_self_citations: bool = True) -> list[dict]:
    """One call: a seeded random sample of the works citing this paper.

    Each record carries the reconstructed abstract and the topic hierarchy, so
    external diversity and the bridge score are computed over exactly the same
    sample. Note that OpenAlex reports meta.count as the sample size when
    `sample` is used, so the true citation total is taken from the paper's own
    cited_by_count instead.
    """
    filters = [f"cites:{paper_id}", "has_abstract:true", "has_doi:true"]
    if exclude_self_citations:
        filters.append(f"author.id:!{author_id}")

    data = await fetch_with_retry(
        client, f"{OPENALEX_BASE_URL}/works",
        params={
            "filter": ",".join(filters),
            "select": "id,abstract_inverted_index,primary_topic",
            "sample": limit,
            "seed": seed,
            "per_page": limit,
        },
    )
    if not data or not data.get("results"):
        return []

    return [{
        "id": w["id"].rsplit("/", 1)[-1],
        "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
        "topic": _topic_of(w),
    } for w in data["results"]]


async def build_author_cache(
    author_id: str,
    top_n: int = 10,
    citations_per_paper: int = 50,
    references_per_paper: int = 50,
    seed: int = 42,
    exclude_self_citations: bool = True,
    cache_dir: Optional[str] = None,
    progress=None,
) -> dict:
    """Fetch and cache everything one analysis needs."""
    if not author_id or not author_id.startswith("A"):
        raise ValueError(f"Invalid author ID format: {author_id}")

    top_n = max(1, min(100, top_n))
    citations_per_paper = max(1, min(200, citations_per_paper))
    references_per_paper = max(1, min(200, references_per_paper))

    params = {
        "schema": CACHE_SCHEMA,
        "top_n": top_n,
        "citations_per_paper": citations_per_paper,
        "references_per_paper": references_per_paper,
        "seed": seed,
        "exclude_self_citations": exclude_self_citations,
    }

    logger.info(f"Building cache for {author_id} "
                f"(top_n={top_n}, citations={citations_per_paper}, seed={seed})")

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # ---- 1 call: the papers themselves, with their reference lists ----
        select = ("id,title,publication_year,cited_by_count,doi,"
                  "abstract_inverted_index,primary_topic,referenced_works")
        query = {"filter": f"author.id:{author_id},has_abstract:true,has_doi:true",
                 "select": select, "per_page": top_n,
                 "sort": "cited_by_count:desc"}

        data = await fetch_with_retry(client, f"{OPENALEX_BASE_URL}/works", params=query)
        if data is None:
            # The request failed rather than came back empty. Writing that to
            # disk would persist a transient outage as a permanent "no data".
            raise NotEnoughData(
                "OpenAlex did not answer the request for this author's papers. "
                "Nothing has been cached; try again in a moment.")

        if not data.get("results"):
            logger.warning(f"No analysable works found for {author_id}")
            raise NotEnoughData(
                "OpenAlex lists no works for this author that carry both an abstract "
                "and a DOI, so there is nothing to compare.")

        papers = []
        for entry in data["results"]:
            refs = [u.rsplit("/", 1)[-1] for u in (entry.get("referenced_works") or [])]
            papers.append({
                "id": entry["id"].rsplit("/", 1)[-1],
                "title": entry.get("title") or "Untitled",
                "year": entry.get("publication_year"),
                "doi": entry.get("doi"),
                "abstract_inverted_index": entry.get("abstract_inverted_index"),
                "citation_count": entry.get("cited_by_count", 0),
                "topic": _topic_of(entry),
                "referenced_works": refs[:references_per_paper],
            })

        # ---- 1 call per paper: its citing works, sampled and self-citations dropped ----
        tasks = [
            fetch_citing_works(p["id"], author_id, citations_per_paper, client,
                               seed=seed, exclude_self_citations=exclude_self_citations)
            for p in papers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for paper, result in zip(papers, results):
            if isinstance(result, RateLimitExceeded):
                raise result
            if isinstance(result, Exception):
                logger.warning(f"Citing-works fetch failed for {paper['id']}: {result}")
                paper["citing"] = []
                continue
            paper["citing"] = result

        # ---- ~10 calls: field metadata for every referenced work, de-duplicated ----
        unique_refs = sorted({r for p in papers for r in p["referenced_works"]})
        logger.info(f"Resolving {len(unique_refs)} unique referenced works")
        ref_works = await fetch_works_by_ids(unique_refs, client, select="id,primary_topic")
        ref_topics = {w["id"].rsplit("/", 1)[-1]: _topic_of(w) for w in ref_works}
        if unique_refs and not ref_works:
            raise NotEnoughData(
                f"None of the {len(unique_refs)} referenced works could be retrieved, so "
                "reference diversity and the bridge score cannot be computed. This is "
                "usually a temporary OpenAlex problem rather than a property of the author.")

        cache_data = {
            "schema": CACHE_SCHEMA,
            "params": params,
            "author_id": author_id,
            "top_papers": papers,
            "reference_topics": ref_topics,
            "paper_count": len(papers),
            "total_citations": sum(p["citation_count"] for p in papers),
            "works_total": data.get("meta", {}).get("count", len(papers)),
        }

        save_author_cache(author_id, cache_data, cache_dir)
        logger.info(f"Cached {len(papers)} papers, "
                    f"{sum(len(p.get('citing') or []) for p in papers)} citing works, "
                    f"{len(ref_topics)} resolved references")
        return cache_data


# ============================================================================
# PROFILE COHERENCE
# ============================================================================

async def fetch_author_signals(author_id: str, client: httpx.AsyncClient,
                               limit: int = 100) -> dict:
    """One call's worth of evidence about whether an author record is one person.

    OpenAlex assigns author ids algorithmically, so a record can be one person
    split across several ids, or several people with the same name merged into
    one. A merged record inflates every diversity measure at once, which makes
    it the most dangerous failure mode for this tool: the score is highest
    exactly when the data are worst.

    Returns raw aggregates; the judgement lives in the app layer.
    """
    data = await fetch_with_retry(
        client, f"{OPENALEX_BASE_URL}/works",
        params={"filter": f"author.id:{author_id}",
                "select": "authorships,primary_topic,publication_year",
                # Deliberately OpenAlex's default ordering rather than a random
                # sample. The co-author repeat rate is confounded by sampling
                # density: drawing 100 works at random from a 1,200-work record
                # is an 8% sample, in which even a tight collaborator core
                # rarely appears twice, so prolific authors look incoherent
                # purely for being prolific. The default ordering concentrates
                # the core of a record, where a real collaborator group shows
                # up. The trade-off is that a split confined to a small corner
                # of a very large record can be missed.
                "per_page": min(limit, 200)},
    )
    works = (data or {}).get("results") or []
    if not works:
        return {"works_sampled": 0}

    raw_names, institutions, orcids, years = [], set(), set(), []
    coauthor_counts = {}
    by_field = {}          # field -> set of coauthor ids
    field_works = {}       # field -> work count

    for w in works:
        if w.get("publication_year"):
            years.append(w["publication_year"])

        topic = w.get("primary_topic") or {}
        field = (topic.get("field") or {}).get("display_name")

        others = set()
        for a in w.get("authorships") or []:
            author = a.get("author") or {}
            aid = (author.get("id") or "").rsplit("/", 1)[-1]
            if aid == author_id:
                if a.get("raw_author_name"):
                    raw_names.append(a["raw_author_name"])
                if author.get("orcid"):
                    orcids.add(author["orcid"].rsplit("/", 1)[-1])
                for inst in a.get("institutions") or []:
                    if inst.get("display_name"):
                        institutions.add(inst["display_name"])
            elif aid:
                others.add(aid)
                coauthor_counts[aid] = coauthor_counts.get(aid, 0) + 1

        if field:
            by_field.setdefault(field, set()).update(others)
            field_works[field] = field_works.get(field, 0) + 1

    return {
        "works_sampled": len(works),
        "raw_names": raw_names,
        "n_institutions": len(institutions),
        "institutions": sorted(institutions),
        "orcids": sorted(orcids),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "n_coauthors": len(coauthor_counts),
        "n_repeat_coauthors": sum(1 for c in coauthor_counts.values() if c > 1),
        "field_works": field_works,
        "field_coauthors": {f: sorted(c) for f, c in by_field.items()},
    }

async def verify_api_key() -> dict:
    """Confirm OpenAlex is honouring the key, rather than assuming it.

    A single-entity fetch is free, so this costs nothing. The response carries
    the budget that was applied — $0.10 anonymous, $1 with a key — which is the
    only way to tell an accepted key from a silently ignored one.
    """
    result = {"key_set": bool(OPENALEX_API_KEY), "budget_usd": None, "accepted": None}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{OPENALEX_BASE_URL}/works/W2741809807",
                                 params={"select": "id"}, headers=HEADERS)
        result["status"] = r.status_code
        if r.status_code in (401, 403):
            # A bad key is rejected outright rather than downgraded, so this is
            # definitive, not merely unconfirmed.
            result["accepted"] = False
            result["rejected"] = True
            return result
        budget = r.headers.get("x-ratelimit-limit-usd")
        if budget is not None:
            result["budget_usd"] = float(budget)
            # Anything above the anonymous $0.10 means the key was applied.
            result["accepted"] = result["budget_usd"] > 0.15
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def log_api_key_status(result: dict) -> None:
    if not result.get("key_set"):
        logger.warning("No OPENALEX_API_KEY set — anonymous access allows $0.10 of "
                       "usage a day, roughly 19 analyses.")
        return
    if result.get("rejected"):
        logger.error("OpenAlex REJECTED the API key (HTTP %s). Every request will fail "
                     "until it is corrected, or unset OPENALEX_API_KEY to fall back to "
                     "anonymous access.", result.get("status"))
    elif result.get("accepted") is True:
        logger.info("OpenAlex API key accepted — daily budget $%.2f", result["budget_usd"])
    elif result.get("accepted") is False:
        logger.error("OpenAlex API key was sent but the budget is still $%.2f, so it "
                     "was NOT applied. Check the key value.", result["budget_usd"])
    else:
        logger.warning("Could not confirm whether the API key was accepted (%s).",
                       result.get("error") or f"HTTP {result.get('status')}")
