# reproducible_cache.py
"""
Cache management for OpenAlex API data with proper error handling and retry logic.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
import httpx

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENALEX_BASE_URL = "https://api.openalex.org"
HEADERS = {'User-Agent': 'InterdisciplinaryIndexApp/1.0 (mailto:research@example.com)'}
DEFAULT_CACHE_DIR = "cache"
os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds
REQUEST_TIMEOUT = 30  # seconds

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTIONS
# ============================================================================

class RateLimitExceeded(Exception):
    """Raised when OpenAlex daily rate limit (100,000 requests) is hit."""
    pass

# ============================================================================
# CACHE MANAGEMENT (Session-aware)
# ============================================================================

def get_cache_path(author_id: str, cache_dir: Optional[str] = None) -> str:
    """Get the file path for an author's cached data"""
    directory = cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"author_{author_id}.json")


def is_cached(author_id: str, cache_dir: Optional[str] = None) -> bool:
    """Check if author data is already cached"""
    return os.path.exists(get_cache_path(author_id, cache_dir))


def get_cache_timestamp(author_id: str, cache_dir: Optional[str] = None) -> Optional[datetime]:
    """Get the timestamp when the cache was created"""
    cache_path = get_cache_path(author_id, cache_dir)
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        return datetime.fromtimestamp(mtime)
    return None


def save_author_cache(author_id: str, data: dict, cache_dir: Optional[str] = None):
    """Save author data to cache with timestamp"""
    data["cached_at"] = datetime.now().isoformat()
    cache_path = get_cache_path(author_id, cache_dir)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
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
    authors = []
    for filename in os.listdir(directory):
        if filename.startswith("author_") and filename.endswith(".json"):
            author_id = filename[7:-5]  # Remove "author_" prefix and ".json" suffix
            authors.append(author_id)
    return authors


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

    Handles:
    - Rate limiting (429) with daily limit detection
    - Server errors (5xx)
    - Network errors

    Raises:
        RateLimitExceeded: When the daily 100,000 request limit is hit
    """
    for attempt in range(max_retries):
        try:
            r = await client.get(url, params=params, headers=HEADERS)

            # Handle rate limiting
            if r.status_code == 429:
                # Check if this is the daily limit (not just per-second throttling)
                try:
                    error_data = r.json()
                    error_message = error_data.get("message", "")
                    if "100000 requests per day" in error_message:
                        retry_after = error_data.get("retryAfter", "unknown")
                        logger.error(f"Daily rate limit exceeded! Retry after: {retry_after}s")
                        raise RateLimitExceeded(
                            f"OpenAlex daily limit (100,000 requests) exceeded. "
                            f"Please try again tomorrow or wait {retry_after} seconds."
                        )
                except (json.JSONDecodeError, KeyError):
                    pass  # Not a JSON response, treat as regular rate limit

                wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                await asyncio.sleep(wait_time)
                continue

            # Handle server errors
            if r.status_code >= 500:
                wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(f"Server error {r.status_code}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                await asyncio.sleep(wait_time)
                continue

            r.raise_for_status()
            return r.json()

        except RateLimitExceeded:
            raise  # Don't catch this, let it bubble up

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

    return None


async def fetch_citing_ids(paper_id: str, limit: int, client: httpx.AsyncClient) -> List[str]:
    """Fetch IDs of papers that cite the given paper"""
    data = await fetch_with_retry(
        client,
        f"{OPENALEX_BASE_URL}/works",
        params={
            "filter": f"cites:{paper_id},has_abstract:true,has_doi:true",
            "per_page": limit,
            "sort": "publication_date:desc",
        }
    )

    if data and "results" in data:
        return [w["id"].split("/")[-1] for w in data["results"]]
    return []


async def fetch_paper_by_id(paper_id: str, client: httpx.AsyncClient) -> dict:
    """Fetch paper metadata by OpenAlex ID"""
    data = await fetch_with_retry(
        client,
        f"{OPENALEX_BASE_URL}/works/{paper_id}"
    )
    return data if data else {}


async def fetch_paper_with_topics(paper_id: str, client: httpx.AsyncClient) -> dict:
    """Fetch paper metadata including topics/concepts for field analysis"""
    data = await fetch_with_retry(
        client,
        f"{OPENALEX_BASE_URL}/works/{paper_id}",
        params={
            "select": "id,title,publication_year,abstract_inverted_index,cited_by_count,topics,concepts,primary_topic"
        }
    )
    return data if data else {}


# ============================================================================
# CACHE BUILDING
# ============================================================================

async def build_author_cache(
    author_id: str,
    top_n: int = 10,
    citations_per_paper: int = 10,
    cache_dir: Optional[str] = None
) -> dict:
    """
    Build and save cache for an author's top papers and their citations.

    Args:
        author_id: OpenAlex author ID
        top_n: Number of top papers to analyze (by citation count)
        citations_per_paper: Number of citing papers to fetch per paper
        cache_dir: Optional session-specific cache directory

    Returns:
        Dictionary with author's top papers and citing paper IDs
    """
    # Validate inputs
    if not author_id or not author_id.startswith("A"):
        raise ValueError(f"Invalid author ID format: {author_id}")

    if not 1 <= top_n <= 100:
        logger.warning(f"top_n={top_n} out of range, clamping to [1, 100]")
        top_n = max(1, min(100, top_n))

    if not 1 <= citations_per_paper <= 50:
        logger.warning(f"citations_per_paper={citations_per_paper} out of range, clamping to [1, 50]")
        citations_per_paper = max(1, min(50, citations_per_paper))

    logger.info(f"Building cache for author {author_id} (top {top_n} papers, {citations_per_paper} citations each)")

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        papers = []
        cursor = "*"
        max_papers = 2000  # Safety limit

        # Fetch all papers by author
        while True:
            data = await fetch_with_retry(
                client,
                f"{OPENALEX_BASE_URL}/works",
                params={
                    "filter": f"author.id:{author_id},has_abstract:true,has_doi:true",
                    "select": "id,title,publication_year,abstract_inverted_index,cited_by_count,topics,concepts,primary_topic",
                    "per_page": 200,
                    "cursor": cursor
                }
            )

            if not data:
                logger.error(f"Failed to fetch papers for author {author_id}")
                break

            for entry in data.get("results", []):
                abstract = entry.get("abstract_inverted_index")
                if abstract:
                    # Extract primary topic/field for field analysis
                    primary_topic = None
                    if entry.get("primary_topic"):
                        primary_topic = {
                            "name": entry["primary_topic"].get("display_name", ""),
                            "field": entry["primary_topic"].get("field", {}).get("display_name", ""),
                            "domain": entry["primary_topic"].get("domain", {}).get("display_name", "")
                        }

                    papers.append({
                        "id": entry["id"].split("/")[-1],
                        "title": entry["title"],
                        "year": entry["publication_year"],
                        "abstract_inverted_index": abstract,
                        "citation_count": entry.get("cited_by_count", 0),
                        "primary_topic": primary_topic,
                        "concepts": entry.get("concepts", [])  # Include concepts for co-occurrence analysis
                    })

            if len(papers) >= max_papers or not data.get("meta", {}).get("next_cursor"):
                break
            cursor = data["meta"]["next_cursor"]

        logger.info(f"Found {len(papers)} papers for author {author_id}")

        # Select top papers by citation count
        papers = sorted(papers, key=lambda p: p["citation_count"], reverse=True)[:top_n]

        # Fetch citing paper IDs for each top paper
        for paper in papers:
            citing_ids = await fetch_citing_ids(paper["id"], citations_per_paper, client)
            paper["citing_ids"] = citing_ids
            logger.debug(f"Paper '{paper['title'][:50]}...' has {len(citing_ids)} citing papers")

        cache_data = {
            "author_id": author_id,
            "top_papers": papers,
            "paper_count": len(papers),
            "total_citations": sum(p["citation_count"] for p in papers)
        }

        save_author_cache(author_id, cache_data, cache_dir)
        logger.info(f"Successfully cached {len(papers)} papers for author {author_id}")

        return cache_data


# ============================================================================
# FIELD/TOPIC ANALYSIS
# ============================================================================

async def fetch_citing_papers_with_topics(
    citing_ids: List[str],
    client: httpx.AsyncClient
) -> List[dict]:
    """
    Fetch citing papers with their topics for field breakdown analysis.

    Returns list of papers with topic/field information.
    """
    tasks = [fetch_paper_with_topics(cid, client) for cid in citing_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    papers = []
    for result in results:
        if isinstance(result, dict) and result:
            papers.append(result)

    return papers


def extract_fields_from_papers(papers: List[dict]) -> Dict[str, int]:
    """
    Extract and count fields/domains from a list of papers.

    Returns dictionary mapping field names to counts.
    """
    field_counts = {}

    for paper in papers:
        if paper.get("primary_topic"):
            field = paper["primary_topic"].get("field", {})
            if isinstance(field, dict):
                field_name = field.get("display_name", "Unknown")
            else:
                field_name = str(field) if field else "Unknown"

            field_counts[field_name] = field_counts.get(field_name, 0) + 1

        # Also check concepts as backup
        elif paper.get("concepts"):
            for concept in paper["concepts"][:3]:  # Top 3 concepts
                if concept.get("level", 0) <= 1:  # High-level concepts only
                    name = concept.get("display_name", "Unknown")
                    field_counts[name] = field_counts.get(name, 0) + 1

    return field_counts
