# app_inter_ultra.py
"""
Interdisciplinary Index Analyzer ULTRA

Measures the cross-domain impact of academic research using 4 complementary metrics:
- External Diversity, Internal Diversity, Reference Diversity, and Bridge Score.

ULTRA version analyzes top 10 most-cited papers for interdisciplinarity analysis.
"""

import gradio as gr
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
import logging
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from collections import Counter, defaultdict
from scipy.stats import gaussian_kde, entropy
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile
import uuid
import shutil
import httpx
from reproducible_cache import (
    is_cached, load_author_cache, build_author_cache, fetch_paper_by_id,
    get_cache_timestamp, clear_author_cache, fetch_with_retry,
    OPENALEX_BASE_URL, HEADERS, REQUEST_TIMEOUT
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MODEL_NAME = 'minishlab/potion-base-32M'
    DEVICE = "cpu"
    TOP_N_PAPERS = 10  # Analyze top 10 most-cited papers
    CITATIONS_PER_PAPER = 10
    REFERENCES_PER_PAPER = 20
    MAX_PAPERS_FETCH = 2000
    INDEX_THRESHOLDS = {"low": 20, "moderate": 50, "high": 80}
    KDE_BANDWIDTH = 0.2
    MAX_CONCURRENT_REQUESTS = 5
    REQUEST_TIMEOUT = 30

    # Minimal black & white palette
    COLORS = {
        "primary": "#000000",
        "secondary": "#374151",
        "accent": "#6B7280",
        "success": "#000000",
        "warning": "#000000",
        "gradient_start": "#000000",
        "gradient_end": "#374151",
    }

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

def create_session_cache_dir() -> str:
    session_id = str(uuid.uuid4())[:8]
    cache_dir = os.path.join(tempfile.gettempdir(), f"interdisciplinary_pro_{session_id}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def cleanup_session_cache(cache_dir: str):
    if cache_dir and os.path.exists(cache_dir) and cache_dir.startswith(tempfile.gettempdir()):
        try:
            shutil.rmtree(cache_dir)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def categorize_index(pct: float) -> str:
    if pct <= Config.INDEX_THRESHOLDS["low"]:
        return "Low"
    elif pct <= Config.INDEX_THRESHOLDS["moderate"]:
        return "Moderate"
    elif pct <= Config.INDEX_THRESHOLDS["high"]:
        return "High"
    return "Very High"

def reconstruct_abstract(inverted_index: dict | None) -> str | None:
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

async def search_authors(author_query: str, limit: int = 5) -> list[dict]:
    async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
        try:
            r = await client.get(
                f"{OPENALEX_BASE_URL}/authors",
                params={"search": author_query, "per_page": limit},
                headers=HEADERS
            )
            r.raise_for_status()
            data = r.json()
            candidates = []
            for result in data.get("results", []):
                institution = result.get("last_known_institution", {}).get("display_name", "") if result.get("last_known_institution") else ""
                candidates.append({
                    "id": result["id"].split("/")[-1],
                    "name": result["display_name"],
                    "works_count": result.get("works_count", 0),
                    "cited_by_count": result.get("cited_by_count", 0),
                    "institution": institution,
                })
            return candidates
        except Exception as e:
            logger.error(f"Author search error: {e}")
            return []

# ============================================================================
# METRIC CALCULATIONS
# ============================================================================

def calculate_embedding_dispersion(abstracts: list[str]) -> dict:
    if len(abstracts) < 2:
        return {"dispersion_score": 0, "n_clusters": 1, "cluster_labels": [0] * len(abstracts), "embeddings": None}

    with torch.no_grad():
        embeddings = sentence_model.encode(abstracts, convert_to_tensor=True, device=Config.DEVICE)
        embeddings_np = embeddings.cpu().numpy()

    pairwise_distances = pdist(embeddings_np, metric='cosine')
    avg_distance = np.mean(pairwise_distances)
    dispersion_score = min(100, (avg_distance / 0.8) * 100)

    n_clusters, cluster_labels = 1, [0] * len(abstracts)
    if len(abstracts) >= 3:
        best_silhouette = -1
        for k in range(2, min(6, len(abstracts))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings_np)
                score = silhouette_score(embeddings_np, labels)
                if score > best_silhouette:
                    best_silhouette, n_clusters, cluster_labels = score, k, labels.tolist()
            except:
                continue

    return {"dispersion_score": float(dispersion_score), "n_clusters": n_clusters, "cluster_labels": cluster_labels, "embeddings": embeddings_np}

async def fetch_paper_references(paper_id: str, client: httpx.AsyncClient, limit: int = 20) -> list[dict]:
    data = await fetch_with_retry(client, f"{OPENALEX_BASE_URL}/works/{paper_id}", params={"select": "referenced_works"})
    if not data or not data.get("referenced_works"):
        return []
    references = []
    for ref_url in data["referenced_works"][:limit]:
        ref_id = ref_url.split("/")[-1]
        ref_data = await fetch_with_retry(client, f"{OPENALEX_BASE_URL}/works/{ref_id}", params={"select": "id,title,primary_topic,concepts"})
        if ref_data:
            references.append(ref_data)
    return references

def extract_field_from_paper(paper: dict) -> str:
    if paper.get("primary_topic"):
        topic = paper["primary_topic"]
        if isinstance(topic, dict):
            field = topic.get("field", {})
            if isinstance(field, dict) and field.get("display_name"):
                return field["display_name"]
            domain = topic.get("domain", {})
            if isinstance(domain, dict) and domain.get("display_name"):
                return domain["display_name"]
    if paper.get("concepts"):
        for concept in paper["concepts"][:1]:
            if concept.get("level", 0) <= 1 and concept.get("display_name"):
                return concept["display_name"]
    return "Unknown"

async def calculate_reference_diversity(author_id: str, papers: list[dict], client: httpx.AsyncClient) -> dict:
    all_field_counts = defaultdict(int)
    for paper in papers[:Config.TOP_N_PAPERS]:
        paper_id = paper.get("id", "").split("/")[-1] if "/" in str(paper.get("id", "")) else paper.get("id", "")
        if not paper_id:
            continue
        references = await fetch_paper_references(paper_id, client, Config.REFERENCES_PER_PAPER)
        for ref in references:
            field = extract_field_from_paper(ref)
            if field != "Unknown":
                all_field_counts[field] += 1

    if not all_field_counts:
        return {"field_counts": {}, "entropy_score": 0, "diversity_index": 0, "unique_fields": 0}

    total = sum(all_field_counts.values())
    probabilities = [count / total for count in all_field_counts.values()]
    entropy_score = entropy(probabilities, base=2)
    max_entropy = np.log2(len(all_field_counts)) if len(all_field_counts) > 1 else 1
    diversity_index = (entropy_score / max_entropy) * 100 if max_entropy > 0 else 0

    return {"field_counts": dict(all_field_counts), "entropy_score": float(entropy_score), "diversity_index": float(diversity_index), "unique_fields": len(all_field_counts)}

async def fetch_citing_paper_fields(paper_id: str, client: httpx.AsyncClient, limit: int = 20) -> list[str]:
    data = await fetch_with_retry(client, f"{OPENALEX_BASE_URL}/works", params={"filter": f"cites:{paper_id}", "per_page": limit, "select": "id,primary_topic,concepts"})
    if not data or not data.get("results"):
        return []
    return [extract_field_from_paper(p) for p in data["results"] if extract_field_from_paper(p) != "Unknown"]

async def calculate_bridge_score(author_id: str, papers: list[dict], client: httpx.AsyncClient) -> dict:
    source_field_counts, audience_field_counts = defaultdict(int), defaultdict(int)

    for paper in papers[:Config.TOP_N_PAPERS]:
        paper_id = paper.get("id", "").split("/")[-1] if "/" in str(paper.get("id", "")) else paper.get("id", "")
        if not paper_id:
            continue
        for ref in await fetch_paper_references(paper_id, client, 10):
            field = extract_field_from_paper(ref)
            if field != "Unknown":
                source_field_counts[field] += 1
        for field in await fetch_citing_paper_fields(paper_id, client, 10):
            audience_field_counts[field] += 1

    if not source_field_counts or not audience_field_counts:
        return {"source_fields": {}, "audience_fields": {}, "bridge_score": 0, "bridged_fields": [], "common_fields": []}

    source_set, audience_set = set(source_field_counts.keys()), set(audience_field_counts.keys())
    bridged_fields = audience_set - source_set
    total_unique = len(source_set | audience_set)
    bridge_score = (len(bridged_fields) / total_unique) * 100 if total_unique > 0 else 0

    return {"source_fields": dict(source_field_counts), "audience_fields": dict(audience_field_counts), "bridge_score": float(bridge_score), "bridged_fields": list(bridged_fields), "common_fields": list(source_set & audience_set)}

def calculate_similarity_and_index(original_abstract: str, citing_abstracts: list[str]) -> tuple:
    with torch.no_grad():
        e_orig = sentence_model.encode(original_abstract, convert_to_tensor=True, device=Config.DEVICE)
        e_cite = sentence_model.encode(citing_abstracts, convert_to_tensor=True, device=Config.DEVICE)
        sims = util.cos_sim(e_orig, e_cite)[0].cpu().tolist()
        sims = [max(0.0, s) for s in sims]
        avg_sim = sum(sims) / len(sims)
        return avg_sim, 1.0 - avg_sim, len(sims), sims

async def fetch_citing_abstracts_parallel(citing_ids: list[str], client: httpx.AsyncClient) -> list[str]:
    if not citing_ids:
        return []
    tasks = [fetch_paper_by_id(cid, client) for cid in citing_ids]
    papers = await asyncio.gather(*tasks, return_exceptions=True)
    return [reconstruct_abstract(p.get("abstract_inverted_index")) for p in papers if isinstance(p, dict) and reconstruct_abstract(p.get("abstract_inverted_index"))]

# ============================================================================
# VISUALIZATION FUNCTIONS (Fresh Modern Theme)
# ============================================================================

def get_chart_layout(title: str, height: int = 350, width: int = None) -> dict:
    """Common chart layout with minimal, Japanese-inspired styling"""
    layout = {
        "title": dict(
            text=title,
            font=dict(size=13, color="#111827", family="Zen Kaku Gothic New, sans-serif"),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top"
        ),
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FAFAFA",
        "font": dict(family="Zen Kaku Gothic New, sans-serif", color="#374151", size=11),
        "margin": dict(t=45, b=50, l=60, r=80),
        "height": height,
        "autosize": True,
        "modebar": dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
        "hoverlabel": dict(
            bgcolor="white",
            font_size=12,
            font_family="Zen Kaku Gothic New, sans-serif",
            font_color="#111827",
            bordercolor="#E5E7EB"
        )
    }
    if width:
        layout["width"] = width
    return layout

def create_dispersion_chart(dispersion_data: dict, paper_titles: list[str]) -> go.Figure:
    """Create a heatmap showing pairwise cosine similarity between papers."""
    embeddings = dispersion_data.get("embeddings")
    if embeddings is None or len(embeddings) < 2:
        return None

    # Compute pairwise cosine similarity (same metric used for dispersion score)
    sim_matrix = cosine_similarity(embeddings)
    n_papers = len(embeddings)

    # Create short labels for axes
    short_titles = []
    for i, title in enumerate(paper_titles):
        short = title[:25] + "..." if len(title) > 25 else title
        short_titles.append(f"P{i+1}: {short}")

    # Create hover text with full information
    hover_text = []
    for i in range(n_papers):
        row = []
        for j in range(n_papers):
            title_i = paper_titles[i][:40] + "..." if len(paper_titles[i]) > 40 else paper_titles[i]
            title_j = paper_titles[j][:40] + "..." if len(paper_titles[j]) > 40 else paper_titles[j]
            sim = sim_matrix[i, j]
            if i == j:
                row.append(f"<b>Paper {i+1}</b><br>{title_i}<br><br>Self-similarity: 1.00")
            else:
                row.append(f"<b>Paper {i+1} ↔ Paper {j+1}</b><br><br>{title_i}<br><b>vs</b><br>{title_j}<br><br>Cosine Similarity: <b>{sim:.3f}</b>")
        hover_text.append(row)

    # Calculate average similarity (excluding diagonal) for annotation
    upper_tri = sim_matrix[np.triu_indices(n_papers, k=1)]
    avg_sim = np.mean(upper_tri)
    dispersion_score = dispersion_data.get("dispersion_score", 0)

    # Convert numpy array to list for proper HTML export serialization
    z_data = sim_matrix.tolist()

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[f"P{i+1}" for i in range(n_papers)],
        y=[f"P{i+1}" for i in range(n_papers)],
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale=[
            [0.0, '#F3F4F6'],   # Light gray for low similarity
            [0.5, '#93C5FD'],   # Light blue for medium
            [0.75, '#3B82F6'],  # Blue for high
            [1.0, '#1E3A8A'],   # Dark blue for very high
        ],
        zmin=0,
        zmax=1,
        colorbar=dict(
            title=dict(
                text="Cosine<br>Similarity",
                font=dict(size=11, color="#374151")
            ),
            tickfont=dict(size=10, color="#374151"),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["0 (Different)", "0.25", "0.5", "0.75", "1 (Identical)"],
            len=0.8,
            thickness=15,
            x=1.02
        )
    ))

    fig.update_layout(**get_chart_layout("Paper Similarity Matrix", 500))
    fig.update_layout(
        autosize=True,
        xaxis=dict(
            title="",
            tickfont=dict(size=9, color="#374151"),
            tickangle=0,
            side="bottom",
            type="category",  # Force categorical axis for P1, P2, etc.
            categoryorder="array",
            categoryarray=[f"P{i+1}" for i in range(n_papers)]
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=9, color="#374151"),
            autorange="reversed",  # So P1 is at top
            type="category",  # Force categorical axis for P1, P2, etc.
            categoryorder="array",
            categoryarray=[f"P{i+1}" for i in range(n_papers)]
        ),
        margin=dict(l=50, r=100, t=60, b=100),
        annotations=[
            dict(
                x=0.5, y=-0.15, xref='paper', yref='paper',
                text=f"Average pairwise similarity: <b>{avg_sim:.3f}</b> | Dispersion score: <b>{dispersion_score:.0f}%</b><br><i>Lower similarity = higher dispersion = more interdisciplinary research</i>",
                showarrow=False,
                font=dict(size=11, color='#374151'),
                xanchor='center',
                align='center'
            )
        ]
    )

    return fig

def create_reference_diversity_chart(ref_diversity: dict) -> go.Figure:
    field_counts = ref_diversity.get("field_counts", {})
    if not field_counts:
        return None

    sorted_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:10][::-1]
    labels, values = [f[0] for f in sorted_fields], [f[1] for f in sorted_fields]

    fig = go.Figure(go.Bar(y=labels, x=values, orientation='h',
        marker=dict(color='#000000', line=dict(width=0)),
        text=values, textposition='outside', textfont=dict(size=10, color='#000000')))

    fig.update_layout(**get_chart_layout("Fields You Reference"))
    fig.update_xaxes(title="Count", title_font=dict(color="#374151", size=11), showgrid=True, gridcolor='#E5E7EB')
    fig.update_yaxes(tickfont=dict(size=10, color="#374151"))
    fig.update_layout(margin=dict(l=200, r=80))
    return fig

def create_bridge_chart(bridge_data: dict) -> go.Figure:
    source_fields, audience_fields = bridge_data.get("source_fields", {}), bridge_data.get("audience_fields", {})
    if not source_fields or not audience_fields:
        return None

    sorted_sources = sorted(source_fields.items(), key=lambda x: x[1], reverse=True)[:6]
    sorted_audience = sorted(audience_fields.items(), key=lambda x: x[1], reverse=True)[:6]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='You cite', y=[f[0] for f in sorted_sources[::-1]], x=[-f[1] for f in sorted_sources[::-1]],
        orientation='h', marker_color='#000000', text=[f[1] for f in sorted_sources[::-1]], textposition='inside', textfont=dict(color='white', size=9)))
    fig.add_trace(go.Bar(name='Cite you', y=[f[0] for f in sorted_audience[::-1]], x=[f[1] for f in sorted_audience[::-1]],
        orientation='h', marker_color='#9CA3AF', text=[f[1] for f in sorted_audience[::-1]], textposition='inside', textfont=dict(color='#000000', size=9)))

    fig.update_layout(**get_chart_layout("Knowledge Flow"))
    fig.update_layout(barmode='overlay', legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center', font=dict(color="#374151", size=10)))
    fig.update_xaxes(title="Count", title_font=dict(color="#374151", size=11), zeroline=True, zerolinecolor='#9CA3AF', zerolinewidth=2)
    fig.update_yaxes(tickfont=dict(size=10, color="#374151"))
    fig.update_layout(margin=dict(l=180, r=80))
    return fig

def create_citation_fields_chart(audience_fields: dict) -> go.Figure:
    if not audience_fields:
        return None

    sorted_fields = sorted(audience_fields.items(), key=lambda x: x[1], reverse=True)[:10][::-1]
    labels, values = [f[0] for f in sorted_fields], [f[1] for f in sorted_fields]
    total = sum(values)

    fig = go.Figure(go.Bar(y=labels, x=values, orientation='h',
        marker=dict(color='#6B7280', line=dict(width=0)),
        text=[f"{v} ({v/total*100:.0f}%)" for v in values], textposition='outside', textfont=dict(size=9, color='#000000')))

    fig.update_layout(**get_chart_layout("Who Cites Your Work"))
    fig.update_xaxes(title="Citations", title_font=dict(color="#374151", size=11), showgrid=True, gridcolor='#E5E7EB')
    fig.update_yaxes(tickfont=dict(size=10, color="#374151"))
    fig.update_layout(margin=dict(l=200, r=100))
    return fig

def create_metrics_summary_chart(metrics: dict) -> go.Figure:
    metric_config = [
        {'name': 'External Diversity', 'key': 'citation_index', 'color': '#000000'},
        {'name': 'Internal Diversity', 'key': 'dispersion_score', 'color': '#374151'},
        {'name': 'Reference Diversity', 'key': 'reference_diversity', 'color': '#6B7280'},
        {'name': 'Bridge Score', 'key': 'bridge_score', 'color': '#9CA3AF'},
    ]

    fig = go.Figure()

    # Background zones for 4 metrics
    for i in range(4):
        y = 3 - i
        fig.add_shape(type="rect", x0=0, x1=20, y0=y-0.35, y1=y+0.35, fillcolor="rgba(156,163,175,0.1)", line_width=0, layer="below")
        fig.add_shape(type="rect", x0=20, x1=50, y0=y-0.35, y1=y+0.35, fillcolor="rgba(156,163,175,0.2)", line_width=0, layer="below")
        fig.add_shape(type="rect", x0=50, x1=80, y0=y-0.35, y1=y+0.35, fillcolor="rgba(156,163,175,0.3)", line_width=0, layer="below")
        fig.add_shape(type="rect", x0=80, x1=100, y0=y-0.35, y1=y+0.35, fillcolor="rgba(156,163,175,0.4)", line_width=0, layer="below")

    for i, cfg in enumerate(metric_config):
        value = metrics.get(cfg['key'], 0)
        y = 3 - i
        fig.add_trace(go.Bar(x=[value], y=[y], orientation='h', marker_color=cfg['color'], width=0.5,
            text=[f"{value:.0f}%"], textposition='outside', textfont=dict(size=12, color='#111827'),
            hovertemplate=f"<b>{cfg['name']}</b>: {value:.1f}%<extra></extra>", showlegend=False))

    fig.update_layout(**get_chart_layout("Metrics Overview", 280))
    fig.update_xaxes(range=[0, 120], tickvals=[0, 20, 50, 80, 100], ticktext=['0', '20', '50', '80', '100'], tickfont=dict(color="#374151", size=10))
    fig.update_yaxes(tickvals=[0,1,2,3], ticktext=[c['name'] for c in reversed(metric_config)], tickfont=dict(size=11, color="#374151"))
    fig.update_layout(margin=dict(l=160, r=60))
    return fig

def create_scatter_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df['year'].astype(int).tolist(),
        y=df['paper_index'].tolist(),
        mode='markers',
        marker=dict(size=10, color='#000000', line=dict(width=1.5, color='white')),
        text=df['title'].tolist(),
        hovertemplate="<b>%{text}</b><br>Year: %{x}<br>Index: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(**get_chart_layout("External Diversity by Year"))
    fig.update_xaxes(title="Year", title_font=dict(color="#374151", size=11), showgrid=True, gridcolor='#E5E7EB', dtick=1, tickfont=dict(size=10))
    fig.update_yaxes(title="Index (%)", title_font=dict(color="#374151", size=11), showgrid=True, gridcolor='#E5E7EB', range=[0, 100], tickfont=dict(size=10))
    return fig

def create_kde_chart(similarities: list) -> go.Figure:
    if not similarities or len(similarities) < 2:
        return None

    try:
        sims = [float(s) for s in similarities if s is not None]
        if len(sims) < 2:
            return None
        kde = gaussian_kde(sims, bw_method=0.2)
        x_vals = np.linspace(min(sims), max(sims), 200)
        y_vals = kde(x_vals)

        fig = go.Figure(go.Scatter(
            x=x_vals.tolist(),
            y=y_vals.tolist(),
            fill='tozeroy',
            mode='lines',
            line=dict(color='#000000', width=2),
            fillcolor='rgba(0, 0, 0, 0.1)',
            hovertemplate="Similarity: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>"
        ))

        fig.update_layout(**get_chart_layout("Similarity Distribution"))
        fig.update_xaxes(title="Cosine Similarity", title_font=dict(color="#374151", size=11), showgrid=True, gridcolor='#E5E7EB', tickfont=dict(size=10))
        fig.update_yaxes(title="Density", title_font=dict(color="#374151", size=11), showgrid=True, gridcolor='#E5E7EB', tickfont=dict(size=10))
        return fig
    except Exception as e:
        logger.warning(f"Could not create KDE chart: {e}")
        return None

def create_keywords_chart(keyword_counts: list) -> go.Figure:
    if not keyword_counts:
        return None

    top_keywords = keyword_counts[:10][::-1]
    labels, values = [kw[0] for kw in top_keywords], [kw[1] for kw in top_keywords]

    fig = go.Figure(go.Bar(y=labels, x=values, orientation='h',
        marker=dict(color='#374151', line=dict(width=0)),
        text=values, textposition='outside', textfont=dict(size=10, color='#000000')))

    fig.update_layout(**get_chart_layout("Top Keywords"))
    fig.update_xaxes(title="Frequency", title_font=dict(color="#374151", size=11), showgrid=True, gridcolor='#E5E7EB', tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10, color="#374151"))
    fig.update_layout(margin=dict(l=160, r=80))
    return fig

# ============================================================================
# HTML EXPORT (Interactive Charts)
# ============================================================================

def generate_html_report(author_name: str, df: pd.DataFrame, metrics: dict, composite_score: float,
                         scatter_fig, kde_fig, dispersion_fig, ref_div_fig, bridge_fig,
                         field_breakdown_fig, bullet_fig, keyword_fig) -> str:
    tempdir = tempfile.gettempdir()
    html_path = os.path.join(tempdir, f"interdisciplinary_{author_name.replace(' ', '_')}.html")

    # Convert figures to HTML divs
    charts_html = []
    chart_configs = [
        ("Metrics Overview", bullet_fig),
        ("Paper Similarity Matrix", dispersion_fig),
        ("External Diversity by Year", scatter_fig),
        ("Similarity Distribution", kde_fig),
        ("Fields Referenced", ref_div_fig),
        ("Knowledge Flow", bridge_fig),
        ("Citing Fields", field_breakdown_fig),
        ("Top Keywords", keyword_fig),
    ]

    for title, fig in chart_configs:
        if fig is not None:
            try:
                chart_html = fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': True, 'responsive': True})
                charts_html.append(f'<div class="chart-container"><h3>{title}</h3>{chart_html}</div>')
            except Exception as e:
                logger.warning(f"Could not convert {title} chart: {e}")

    # Build papers table with all columns
    papers_table = ""
    if not df.empty:
        papers_table = '''<table class="papers-table">
            <thead><tr><th style="width:40px">#</th><th>Title</th><th style="width:60px">Year</th><th style="width:90px">Diversity %</th><th style="width:80px">Citations</th></tr></thead>
            <tbody>'''
        for idx, row in df.iterrows():
            title_text = row["Title"][:120] + "..." if len(str(row["Title"])) > 120 else row["Title"]
            papers_table += f'<tr><td style="text-align:center">{idx+1}</td><td>{title_text}</td><td style="text-align:center">{row["Year"]}</td><td style="text-align:center">{row["Index (%)"]:.1f}</td><td style="text-align:center">{row.get("citation_count", "N/A")}</td></tr>'
        papers_table += '</tbody></table>'

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interdisciplinary Index Report - {author_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700;900&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Zen Kaku Gothic New', sans-serif; -webkit-font-smoothing: antialiased; }}
        body {{ background: #FAFAFA; min-height: 100vh; padding: 48px 24px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: #000000; padding: 48px 40px; border-radius: 20px; text-align: center; color: white; margin-bottom: 32px; }}
        .header h1 {{ font-size: 2.25rem; font-weight: 700; margin-bottom: 8px; letter-spacing: -0.03em; color: #FFFFFF; }}
        .header p {{ font-size: 1rem; color: rgba(255, 255, 255, 0.8); }}
        .metrics-card {{ background: white; border-radius: 16px; padding: 32px; margin-bottom: 24px; box-shadow: 0 1px 3px rgb(0 0 0 / 0.08); border: 1px solid #E5E7EB; }}
        .metrics-card h2 {{ color: #111827; margin-bottom: 16px; font-size: 1.1rem; font-weight: 600; }}
        .composite-score {{ font-size: 4rem; font-weight: 900; color: #111827; text-align: center; margin: 24px 0; letter-spacing: -0.03em; }}
        .score-category {{ display: inline-block; background: #F3F4F6; color: #374151; padding: 6px 16px; border-radius: 100px; font-size: 0.8rem; font-weight: 600; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 24px; }}
        .metric-item {{ background: #FAFAFA; padding: 20px 16px; border-radius: 12px; text-align: center; border: 1px solid #E5E7EB; transition: all 0.2s; }}
        .metric-item:hover {{ background: white; box-shadow: 0 1px 3px rgb(0 0 0 / 0.08); }}
        .metric-item .label {{ font-size: 0.8125rem; color: #374151; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
        .metric-item .value {{ font-size: 1.75rem; font-weight: 700; color: #111827; }}
        .charts-grid {{ display: flex; flex-direction: column; gap: 24px; margin-bottom: 24px; }}
        .chart-container {{ background: white; border-radius: 16px; padding: 24px; box-shadow: 0 1px 3px rgb(0 0 0 / 0.08); border: 1px solid #E5E7EB; width: 100%; }}
        .chart-container h3 {{ font-size: 1rem; font-weight: 600; color: #111827; margin-bottom: 16px; }}
        .chart-container .plotly-graph-div {{ width: 100% !important; min-height: 400px; }}
        .chart-container .js-plotly-plot {{ width: 100% !important; }}
        .table-wrapper {{ max-height: 400px; overflow-y: auto; overflow-x: auto; border: 1px solid #E5E7EB; border-radius: 12px; }}
        .table-wrapper::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        .table-wrapper::-webkit-scrollbar-track {{ background: #F3F4F6; }}
        .table-wrapper::-webkit-scrollbar-thumb {{ background: #9CA3AF; border-radius: 4px; }}
        .table-wrapper::-webkit-scrollbar-thumb:hover {{ background: #6B7280; }}
        .papers-table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; table-layout: fixed; font-family: 'Zen Kaku Gothic New', sans-serif; }}
        .papers-table th {{ background: #000000; color: white; padding: 14px 16px; text-align: left; font-size: 0.75rem; font-weight: 500; position: sticky; top: 0; z-index: 10; text-transform: uppercase; letter-spacing: 0.5px; font-family: 'Zen Kaku Gothic New', sans-serif; }}
        .papers-table th:nth-child(1) {{ width: 50px; text-align: center; }}
        .papers-table th:nth-child(2) {{ width: 45%; }}
        .papers-table th:nth-child(3) {{ width: 80px; text-align: center; }}
        .papers-table th:nth-child(4) {{ width: 100px; text-align: center; }}
        .papers-table th:nth-child(5) {{ width: 100px; text-align: center; }}
        .papers-table td {{ padding: 14px 16px; border-bottom: 1px solid #F3F4F6; color: #374151; font-family: 'Zen Kaku Gothic New', sans-serif; }}
        .papers-table td:nth-child(2) {{ white-space: normal; word-wrap: break-word; color: #111827; font-weight: 500; }}
        .papers-table tr:hover td {{ background: #F9FAFB; }}
        .footer {{ text-align: center; color: #6B7280; font-size: 0.85rem; margin-top: 48px; padding: 24px; border-top: 1px solid #E5E7EB; }}
        @media (max-width: 900px) {{
            .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
        @media print {{
            .chart-container {{ break-inside: avoid; page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Interdisciplinary Index Report</h1>
            <p>{author_name} &bull; Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>

        <div class="metrics-card">
            <h2>Composite Interdisciplinary Score</h2>
            <div class="composite-score">{composite_score:.1f}%</div>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="label">External Diversity</div>
                    <div class="value">{metrics.get('citation_index', 0):.1f}%</div>
                </div>
                <div class="metric-item">
                    <div class="label">Internal Diversity</div>
                    <div class="value">{metrics.get('dispersion_score', 0):.1f}%</div>
                </div>
                <div class="metric-item">
                    <div class="label">Reference Diversity</div>
                    <div class="value">{metrics.get('reference_diversity', 0):.1f}%</div>
                </div>
                <div class="metric-item">
                    <div class="label">Bridge Score</div>
                    <div class="value">{metrics.get('bridge_score', 0):.1f}%</div>
                </div>
            </div>
        </div>

        <div class="metrics-card">
            <h2>Top Papers Analyzed</h2>
            <div class="table-wrapper">
                {papers_table}
            </div>
        </div>

        <div class="charts-grid">
            {''.join(charts_html)}
        </div>

        <div class="footer">
            <p>Generated by Interdisciplinary Index Analyzer</p>
        </div>
    </div>
    <script>
        // Resize all Plotly charts on window resize
        window.addEventListener('resize', function() {{
            document.querySelectorAll('.js-plotly-plot').forEach(function(plot) {{
                Plotly.Plots.resize(plot);
            }});
        }});
        // Initial resize to fit container
        window.addEventListener('load', function() {{
            setTimeout(function() {{
                document.querySelectorAll('.js-plotly-plot').forEach(function(plot) {{
                    Plotly.Plots.resize(plot);
                }});
            }}, 100);
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

async def analyze_author_pro(author_id: str, author_name: str, cache_dir: str = None):
    logger.info(f"Starting analysis for: {author_name} ({author_id})")
    start_time = datetime.now()

    if not is_cached(author_id, cache_dir):
        logger.info("Building cache (first time)...")
        await build_author_cache(author_id, top_n=Config.TOP_N_PAPERS, citations_per_paper=Config.CITATIONS_PER_PAPER, cache_dir=cache_dir)

    logger.info("Loading data...")
    data = load_author_cache(author_id, cache_dir)
    top_papers = data["top_papers"]

    results, all_keywords, similarities_all, all_citing_texts = [], [], [], []

    async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
        total = len(top_papers)
        for idx, paper in enumerate(top_papers):
            logger.info(f"Analyzing paper {idx+1}/{total}...")
            abs_orig = reconstruct_abstract(paper.get("abstract_inverted_index"))
            if not abs_orig:
                continue
            citing_texts = await fetch_citing_abstracts_parallel(paper.get("citing_ids", []), client)
            if citing_texts:
                avg_sim, idx_val, count, sims = calculate_similarity_and_index(abs_orig, citing_texts)
                results.append({"title": paper["title"], "year": paper["year"], "paper_index": idx_val * 100, "citation_count": paper.get("citation_count", 0), "abstract": abs_orig})
                similarities_all.extend(sims)
                all_citing_texts.extend(citing_texts)

        logger.info("Analyzing reference diversity...")
        ref_diversity = await calculate_reference_diversity(author_id, top_papers, client)

        logger.info("Computing bridge score...")
        bridge_data = await calculate_bridge_score(author_id, top_papers, client)

    logger.info("Extracting keywords...")
    for text in all_citing_texts[:50]:
        try:
            kws = kw_model.extract_keywords(text, top_n=3, stop_words='english')
            all_keywords.extend([kw[0] for kw in kws])
        except:
            pass

    logger.info("Creating visualizations...")
    # Sort results by paper_index (External Diversity) descending - this is the table order
    df = pd.DataFrame(results).dropna().sort_values(by="paper_index", ascending=False).reset_index(drop=True)
    citation_index = df['paper_index'].mean() if not df.empty else 0

    # Extract sorted abstracts and titles (matching table order: P1 = highest diversity)
    sorted_abstracts = df['abstract'].tolist() if 'abstract' in df.columns else []
    sorted_titles = df['title'].tolist() if 'title' in df.columns else []

    logger.info("Computing embedding dispersion...")
    dispersion_data = calculate_embedding_dispersion(sorted_abstracts)

    # 4 metrics with weights: 35% + 25% + 30% + 10% = 100%
    all_metrics = {
        'citation_index': citation_index,
        'dispersion_score': dispersion_data['dispersion_score'],
        'reference_diversity': ref_diversity['diversity_index'],
        'bridge_score': bridge_data['bridge_score'],
    }
    composite_score = (
        citation_index * 0.35 +
        dispersion_data['dispersion_score'] * 0.25 +
        ref_diversity['diversity_index'] * 0.30 +
        bridge_data['bridge_score'] * 0.10
    )

    scatter = create_scatter_chart(df) if not df.empty else None
    kde_fig = create_kde_chart(similarities_all)
    # Pass sorted titles so P1 in heatmap = #1 in table (highest diversity)
    dispersion_chart = create_dispersion_chart(dispersion_data, sorted_titles)
    ref_diversity_chart = create_reference_diversity_chart(ref_diversity)
    bridge_chart = create_bridge_chart(bridge_data)
    field_breakdown_chart = create_citation_fields_chart(bridge_data.get('audience_fields', {}))
    bullet_chart = create_metrics_summary_chart(all_metrics)
    keyword_chart = create_keywords_chart(Counter(all_keywords).most_common(10))

    logger.info("Generating report...")
    # Create top papers table (already sorted by citation diversity descending)
    df_top_papers = df[["title", "year", "paper_index", "citation_count"]].copy()
    df_top_papers.insert(0, "Rank", range(1, len(df_top_papers) + 1))
    df_top_papers["paper_index"] = df_top_papers["paper_index"].round(1)
    df_top_papers = df_top_papers.rename(columns={
        "Rank": "#",
        "title": "Paper Title",
        "year": "Year",
        "paper_index": "Diversity %",
        "citation_count": "Citations"
    })

    df_report = df.rename(columns={"title": "Title", "year": "Year", "paper_index": "Index (%)"})
    html_path = generate_html_report(author_name, df_report, all_metrics, composite_score, scatter, kde_fig, dispersion_chart, ref_diversity_chart, bridge_chart, field_breakdown_chart, bullet_chart, keyword_chart)

    category = categorize_index(composite_score)

    results_html = f"""
    <div class="results-card">
        <div class="score-display">
            <div class="score-label">Composite Interdisciplinary Score</div>
            <div class="score-value">{composite_score:.0f}%</div>
            <div class="score-category">{category}</div>
        </div>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">External Diversity</div>
                <div class="metric-value">{citation_index:.0f}%</div>
                <div class="metric-detail">{len(results)} papers analyzed</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Dispersion</div>
                <div class="metric-value">{dispersion_data['dispersion_score']:.0f}%</div>
                <div class="metric-detail">{dispersion_data['n_clusters']} clusters</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Ref. Diversity</div>
                <div class="metric-value">{ref_diversity['diversity_index']:.0f}%</div>
                <div class="metric-detail">{ref_diversity['unique_fields']} fields</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Bridge Score</div>
                <div class="metric-value">{bridge_data['bridge_score']:.0f}%</div>
                <div class="metric-detail">{len(bridge_data.get('bridged_fields', []))} bridged</div>
            </div>
        </div>
    </div>
    """

    explanation = """### How It Works

**Composite Score** is a weighted combination of four metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| External Diversity | 35% | How different are papers that cite you from your own work |
| Internal Diversity | 25% | How spread out your research topics are |
| Reference Diversity | 30% | Variety of fields you draw knowledge from |
| Bridge Score | 10% | Fields that cite you but you don't cite back |

**Interpretation**: Low (0-20%) → Moderate (20-50%) → High (50-80%) → Very High (80-100%)
"""

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Analysis completed in {elapsed:.1f}s")

    return (df_report, results_html, bullet_chart, df_top_papers, scatter, kde_fig, dispersion_chart, ref_diversity_chart, bridge_chart, field_breakdown_chart, keyword_chart, explanation, html_path)

# ============================================================================
# GRADIO UI - PROFESSIONAL DESIGN SYSTEM
# ============================================================================

custom_css = """
/* Professional UI Design - Focus on readability and visual hierarchy */

@import url('https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700;900&display=swap');

/* Core variables - Black & White minimal theme */
:root {
    --primary: #000000;
    --primary-hover: #1F1F1F;
    --bg-light: #FAFAFA;
    --text-dark: #000000;
    --text-medium: #374151;
    --text-light: #6B7280;
    --border: #E5E7EB;
    --border-light: #F3F4F6;
    --success: #000000;
    --warning: #000000;
}

/* Global container and typography */
.gradio-container {
    max-width: 1400px !important;
    font-family: 'Zen Kaku Gothic New', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Improve Gradio native elements */
.gradio-container label {
    font-weight: 600 !important;
    color: var(--text-dark) !important;
    font-size: 0.875rem !important;
    margin-bottom: 8px !important;
}

.gradio-container input[type="text"],
.gradio-container textarea {
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    padding: 10px 14px !important;
    transition: all 0.2s ease !important;
}

.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
    border-color: #000000 !important;
    box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.08) !important;
    outline: none !important;
}

/* Button improvements - WHITE TEXT BY DEFAULT */
.gradio-container button {
    font-weight: 500 !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    border: none !important;
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
}

/* Primary buttons - always black background with white text */
.gradio-container button.primary-btn,
.gradio-container button[variant="primary"],
.gradio-container .primary,
.gradio-container button.lg,
.gradio-container button.primary,
.gradio-container .btn-primary,
.gradio-container button[class*="primary"],
.gradio-container button[class*="lg"],
.gradio-container .gr-button-primary,
.gradio-container .gr-button-lg {
    background: #000000 !important;
    color: #FFFFFF !important;
    padding: 10px 24px !important;
}

/* Ensure ALL text inside buttons is white - comprehensive override */
.gradio-container button.primary-btn,
.gradio-container button.primary-btn *,
.gradio-container button.primary-btn span,
.gradio-container button.primary-btn svg,
.gradio-container button[variant="primary"],
.gradio-container button[variant="primary"] *,
.gradio-container button[variant="primary"] span,
.gradio-container .primary,
.gradio-container .primary *,
.gradio-container .primary span,
.gradio-container button.lg,
.gradio-container button.lg *,
.gradio-container button.lg span,
.gradio-container button.primary,
.gradio-container button.primary *,
.gradio-container button.primary span,
.gradio-container .btn-primary,
.gradio-container .btn-primary *,
.gradio-container .btn-primary span,
.gradio-container button[class*="primary"],
.gradio-container button[class*="primary"] *,
.gradio-container button[class*="primary"] span,
.gradio-container button[class*="lg"],
.gradio-container button[class*="lg"] *,
.gradio-container .gr-button-primary,
.gradio-container .gr-button-primary *,
.gradio-container .gr-button-lg,
.gradio-container .gr-button-lg * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* Hover state */
.gradio-container button.primary-btn:hover,
.gradio-container button[variant="primary"]:hover,
.gradio-container .primary:hover,
.gradio-container button.lg:hover,
.gradio-container button.primary:hover {
    background: #1F1F1F !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    color: #FFFFFF !important;
}

/* Gradio 4.x/5.x specific button selectors - FORCE white text on ALL black buttons */
.gradio-container .svelte-1ipelgc,
.gradio-container .svelte-1ipelgc *,
.gradio-container [data-testid="textbox"] + button,
.gradio-container [data-testid="textbox"] + button * {
    background: #000000 !important;
    color: #FFFFFF !important;
}

/* NUCLEAR OPTION: Force ALL buttons with elem_classes to have white text */
button[class*="primary-btn"],
button[class*="primary-btn"] * {
    background: #000000 !important;
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* Target Gradio's internal button structure */
.gradio-container button[class*="svelte"] {
    color: #FFFFFF !important;
}

.gradio-container button[class*="svelte"] span,
.gradio-container button[class*="svelte"] div,
.gradio-container button[class*="svelte"] p {
    color: #FFFFFF !important;
}

/* Target buttons by their background color */
.gradio-container button[style*="background"],
.gradio-container button {
    --button-text-color: #FFFFFF !important;
}

/* Override Gradio's button text specifically */
.gradio-container .gr-button,
.gradio-container .gr-button *,
.gradio-container [class*="button"],
.gradio-container [class*="button"] * {
    color: inherit;
}

/* Ensure primary-btn class always has white text */
.primary-btn,
.primary-btn *,
.primary-btn span,
.primary-btn div {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* Header section - Pure black/white minimal */
.header-section {
    background: #000000;
    padding: 48px 40px;
    border-radius: 16px;
    margin-bottom: 32px;
    text-align: center;
    color: white;
    position: relative;
    overflow: hidden;
}

.header-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.15);
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.75rem;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
    color: #FFFFFF;
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 12px;
    letter-spacing: -0.02em;
    line-height: 1.2;
    color: #FFFFFF;
}

.header-subtitle {
    font-size: 1.05rem;
    line-height: 1.6;
    max-width: 800px;
    margin: 0 auto;
    color: rgba(255, 255, 255, 0.8);
}

/* Results card - the key display area */
.results-card {
    background: white;
    padding: 32px;
    border-radius: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    margin-bottom: 24px;
    border: 1px solid var(--border);
}

.score-display {
    text-align: center;
    padding: 32px 24px;
    background: linear-gradient(135deg, #F9FAFB 0%, #FFFFFF 100%);
    border-radius: 16px;
    margin-bottom: 24px;
    border: 1px solid var(--border);
}

.score-label {
    font-size: 0.875rem;
    color: var(--text-medium);
    text-transform: uppercase;
    margin-bottom: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
}

.score-value {
    font-size: 4rem;
    font-weight: 900;
    color: var(--text-dark);
    line-height: 1;
    letter-spacing: -0.03em;
}

.score-category {
    display: inline-block;
    margin-top: 16px;
    padding: 8px 20px;
    background: var(--border-light);
    color: var(--text-dark);
    border-radius: 24px;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Metrics grid - improved for 4 items */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}

.metric-card {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid var(--border);
    transition: all 0.2s ease;
}

.metric-card:hover {
    border-color: var(--text-light);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.metric-label {
    font-size: 0.8125rem;
    color: var(--text-medium);
    text-transform: uppercase;
    margin-bottom: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-dark);
    line-height: 1;
    margin-bottom: 6px;
}

.metric-detail {
    font-size: 0.8125rem;
    color: var(--text-medium);
    margin-top: 6px;
    font-weight: 500;
}

/* Progress indicator */
.progress-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px;
    background: white;
    border-radius: 16px;
    margin: 24px 0;
    border: 1px solid var(--border);
}

.progress-spinner {
    width: 48px;
    height: 48px;
    border: 4px solid var(--border-light);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin-bottom: 20px;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.progress-text {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-dark);
    margin-bottom: 4px;
}

.progress-subtext {
    font-size: 0.9rem;
    color: var(--text-light);
    margin-top: 4px;
}

/* Dataframe/Table improvements - comprehensive font override */
.gradio-container .dataframe,
.gradio-container .dataframe *,
.gradio-container table,
.gradio-container table *,
.gradio-container [class*="table"],
.gradio-container [class*="table"] *,
.gradio-container .wrap,
.gradio-container .wrap *,
.gradio-container .svelte-1gfkn6j,
.gradio-container .svelte-1gfkn6j * {
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
}

/* Table container - scrollable */
.gradio-container .dataframe,
.gradio-container table,
.gradio-container [data-testid="dataframe"],
.gradio-container .table-wrap {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: auto !important;
    max-height: 500px !important;
}

/* Table wrapper scrollbar styling */
.gradio-container .table-wrap::-webkit-scrollbar,
.gradio-container [data-testid="dataframe"]::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

.gradio-container .table-wrap::-webkit-scrollbar-track,
.gradio-container [data-testid="dataframe"]::-webkit-scrollbar-track {
    background: #F3F4F6;
    border-radius: 4px;
}

.gradio-container .table-wrap::-webkit-scrollbar-thumb,
.gradio-container [data-testid="dataframe"]::-webkit-scrollbar-thumb {
    background: #9CA3AF;
    border-radius: 4px;
}

.gradio-container .table-wrap::-webkit-scrollbar-thumb:hover,
.gradio-container [data-testid="dataframe"]::-webkit-scrollbar-thumb:hover {
    background: #6B7280;
}

/* Table header - BLACK background with WHITE text */
.gradio-container .dataframe thead,
.gradio-container table thead,
.gradio-container [data-testid="dataframe"] thead {
    background: #000000 !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 10 !important;
}

.gradio-container .dataframe th,
.gradio-container table th,
.gradio-container thead th,
.gradio-container [class*="header"] th,
.gradio-container [data-testid="dataframe"] th,
.gradio-container .table-wrap th {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    padding: 14px 16px !important;
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
    background: #000000 !important;
    border-bottom: none !important;
}

/* Ensure header cell text is white */
.gradio-container th *,
.gradio-container thead *,
.gradio-container [data-testid="dataframe"] th *,
.gradio-container .table-wrap th * {
    color: #FFFFFF !important;
}

.gradio-container .dataframe td,
.gradio-container table td,
.gradio-container tbody td,
.gradio-container [data-testid="dataframe"] td {
    padding: 12px 16px !important;
    border-bottom: 1px solid var(--border-light) !important;
    font-size: 0.875rem !important;
    color: #374151 !important;
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
    background: #FFFFFF !important;
}

/* Ensure ALL text inside table cells is visible - dark gray text */
.gradio-container .dataframe td *,
.gradio-container table td *,
.gradio-container tbody td *,
.gradio-container [data-testid="dataframe"] td *,
.gradio-container [data-testid="dataframe"] td span,
.gradio-container [data-testid="dataframe"] td div,
.gradio-container [data-testid="dataframe"] td p,
.gradio-container .table-wrap td *,
.gradio-container .table-wrap td span,
.gradio-container .table-wrap td div {
    color: #374151 !important;
    background: transparent !important;
}

.gradio-container .dataframe tr:hover td,
.gradio-container table tr:hover td,
.gradio-container [data-testid="dataframe"] tr:hover td {
    background: #FAFAFA !important;
}

.gradio-container .dataframe tr:hover td *,
.gradio-container [data-testid="dataframe"] tr:hover td * {
    background: transparent !important;
}

/* Gradio DataFrame specific - target all internal elements */
.gradio-container [data-testid="dataframe"] *,
.gradio-container [data-testid="table"] *,
.gradio-container .table-wrap *,
.gradio-container .overflow-hidden * {
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
}

/* Force table body text color */
.gradio-container tbody,
.gradio-container tbody *,
.gradio-container [data-testid="dataframe"] tbody,
.gradio-container [data-testid="dataframe"] tbody * {
    color: #374151 !important;
}

/* Accordion improvements */
.gradio-container .accordion {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-top: 16px !important;
}

.gradio-container .accordion button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: var(--text-dark) !important;
}

/* Tab improvements */
.gradio-container .tabs {
    border-bottom: 2px solid var(--border) !important;
    margin-bottom: 24px !important;
}

.gradio-container .tabs button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: var(--text-medium) !important;
    padding: 12px 20px !important;
}

.gradio-container .tabs button.selected {
    color: #000000 !important;
    border-bottom: 3px solid #000000 !important;
}

/* Markdown text improvements */
.gradio-container .markdown-text {
    line-height: 1.7 !important;
    color: var(--text-medium) !important;
}

.gradio-container .markdown-text h2 {
    color: var(--text-dark) !important;
    font-weight: 700 !important;
    margin-top: 24px !important;
    margin-bottom: 12px !important;
}

.gradio-container .markdown-text h3 {
    color: var(--text-dark) !important;
    font-weight: 600 !important;
    margin-top: 20px !important;
    margin-bottom: 10px !important;
}

.gradio-container .markdown-text table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 16px 0 !important;
}

.gradio-container .markdown-text th {
    background: var(--bg-light) !important;
    padding: 10px 14px !important;
    text-align: left !important;
    font-weight: 600 !important;
    border: 1px solid var(--border) !important;
}

.gradio-container .markdown-text td {
    padding: 10px 14px !important;
    border: 1px solid var(--border) !important;
}

/* Responsive design */
@media (max-width: 1200px) {
    .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
    }
}

@media (max-width: 768px) {
    .metrics-grid {
        grid-template-columns: 1fr;
    }

    .header-title {
        font-size: 1.875rem;
    }

    .header-subtitle {
        font-size: 0.95rem;
    }

    .score-value {
        font-size: 3rem;
    }

    .results-card {
        padding: 20px;
    }
}

@media (max-width: 480px) {
    .header-section {
        padding: 32px 20px;
    }

    .header-title {
        font-size: 1.5rem;
    }
}

/* Footer styling - match "Built with Gradio" gray color */
footer,
footer a,
footer span,
footer button,
footer button span,
footer div,
footer p,
footer a[href*="settings"],
footer [class*="settings"],
footer [class*="Settings"],
a[href*="settings"],
button[class*="settings"],
span[class*="settings"],
.footer-link,
.settings-link {
    color: #6B7280 !important;
}

footer svg,
footer button svg,
footer a svg,
[class*="settings"] svg {
    fill: #6B7280 !important;
    color: #6B7280 !important;
}
"""

def create_pro_interface():
    # Force light mode only - no dark mode option
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.gray,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
    )

    # JavaScript to hide Display Theme section and style Settings link
    hide_theme_js = """
    function() {
        function styleFooter() {
            // Style Settings link and all footer text to match "Built with Gradio"
            document.querySelectorAll('a, button, span, div').forEach(function(el) {
                if (el.textContent && el.textContent.trim() === 'Settings') {
                    el.style.setProperty('color', '#6B7280', 'important');
                    el.querySelectorAll('*').forEach(function(child) {
                        child.style.setProperty('color', '#6B7280', 'important');
                    });
                }
            });
            const footer = document.querySelector('footer');
            if (footer) {
                footer.querySelectorAll('*').forEach(function(el) {
                    if (el.tagName !== 'SVG' && el.tagName !== 'PATH') {
                        el.style.setProperty('color', '#6B7280', 'important');
                    }
                });
            }
        }

        function hideTheme() {
            document.querySelectorAll('h2, h3, h4, span, p').forEach(function(el) {
                if (el.textContent && el.textContent.trim() === 'Display Theme') {
                    let section = el.parentElement;
                    while (section && section.tagName !== 'BODY') {
                        if (section.querySelector('button')) {
                            section.style.display = 'none';
                            break;
                        }
                        section = section.parentElement;
                    }
                }
            });
            document.querySelectorAll('button').forEach(function(btn) {
                const t = btn.textContent?.trim();
                if (t === 'Light' || t === 'Dark' || t === 'System') {
                    let p = btn.parentElement;
                    while (p && p.children.length <= 3) {
                        p.style.display = 'none';
                        p = p.parentElement;
                    }
                }
            });
        }

        // Run immediately
        styleFooter();
        hideTheme();

        // Run on any DOM changes
        const observer = new MutationObserver(function() {
            styleFooter();
            hideTheme();
        });
        observer.observe(document.body, { childList: true, subtree: true });

        // Also run after delays to catch late-loading elements
        setTimeout(styleFooter, 500);
        setTimeout(styleFooter, 1000);
        setTimeout(styleFooter, 2000);
    }
    """

    with gr.Blocks(css=custom_css, title="Interdisciplinary Index Analyzer", theme=theme, js=hide_theme_js) as demo:
        cache_dir = gr.State(value=create_session_cache_dir)
        selected_author_id = gr.State(value="")

        gr.HTML("""
        <div class="header-section">
            <div class="header-content">
                <div class="header-badge">Research Analytics Platform</div>
                <h1 class="header-title">Interdisciplinary Index Analyzer</h1>
                <p class="header-subtitle">Measure the cross-domain impact of academic research through advanced citation analysis and semantic modeling</p>
            </div>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("Analyze"):
                with gr.Row():
                    with gr.Column(scale=4):
                        author_input = gr.Textbox(label="Search for a researcher", placeholder="Enter name (e.g., 'Yoshua Bengio')")
                    with gr.Column(scale=1, min_width=140):
                        search_btn = gr.Button("Search", elem_classes=["primary-btn"])

                author_dropdown = gr.Dropdown(label="Select author", choices=[], interactive=True, visible=False)
                analyze_btn = gr.Button("Run Analysis", elem_classes=["primary-btn"], visible=False, size="lg")

                # Custom animated progress indicator
                progress_html = gr.HTML(visible=False, elem_id="progress-indicator")

                results_html = gr.HTML(visible=False)

                with gr.Row():
                    bullet_chart = gr.Plot(label=None, visible=False)

                # Top papers table sorted by citation diversity
                top_papers_table = gr.Dataframe(
                    label="Top 10 Papers by External Diversity Score",
                    visible=False,
                    wrap=True
                )

                # Full width dispersion
                dispersion_plot = gr.Plot(label=None, visible=False)

                with gr.Accordion("Detailed Charts", open=True, visible=False) as charts_accordion:
                    # Each chart on its own row for full width and better readability
                    scatter_plot = gr.Plot(label=None)
                    kde_plot = gr.Plot(label=None)
                    ref_diversity_plot = gr.Plot(label=None)
                    bridge_plot = gr.Plot(label=None)
                    field_breakdown_plot = gr.Plot(label=None)
                    keywords_plot = gr.Plot(label=None)

                with gr.Accordion("Papers Analyzed", open=False, visible=False) as papers_accordion:
                    papers_table = gr.Dataframe()

                html_download = gr.File(label="Download Interactive Report (HTML)", visible=False)
                methodology_md = gr.Markdown(visible=False)

            with gr.Tab("About"):
                gr.Markdown("""
## What is the Interdisciplinary Index?

This tool measures how research crosses disciplinary boundaries. A higher score means your work has broader impact across different fields.

---

## The Four Metrics Explained

### 1. External Diversity (35% of composite score)

**What it measures:** How different are the papers citing your work from your original papers?

**Direction:** External — looks at how *others* engage with your work

**How it works:**
- We take your top 10 most-cited papers and look at who cites them
- Using AI (neural embeddings), we convert each paper's abstract into a mathematical representation
- We measure how "far apart" your papers are from the papers that cite them
- If papers from very different topics cite you, your score is higher

**Example:** If you write about genetics and get cited by ecology, public health, and computer science papers, your external diversity is high — your work reaches outside your own field.

---

### 2. Internal Diversity (25% of composite score)

**What it measures:** How spread out are your own research topics?

**Direction:** Internal — looks at diversity *within* your own body of work

**How it works:**
- We take all your paper abstracts and convert them to embeddings (mathematical representations)
- We compute pairwise cosine similarity between all your papers
- Lower average similarity = papers are more different from each other = higher diversity
- The heatmap visualization shows this similarity matrix (P1 = highest external diversity paper from the table)

**Example:** A researcher who publishes in both "machine learning" and "neuroscience" will have lower similarity between papers (higher internal diversity) than someone who only publishes variations of "machine learning applications."

---

### 3. Reference Diversity (30% of composite score)

**What it measures:** How many different fields do you draw knowledge from?

**How it works:**
- We look at the references in your papers (who do you cite?)
- Each reference is tagged with its academic field (e.g., Biology, Computer Science, Economics)
- We calculate Shannon entropy - a measure of diversity from information theory
- Higher entropy = more evenly distributed across fields = more diverse

**Example:** If you cite 50 biology papers and 50 economics papers, your entropy is higher than citing 95 biology papers and 5 economics papers.

---

### 4. Bridge Score (10% of composite score)

**What it measures:** Does your work connect fields that don't normally interact?

**How it works:**
- We identify which fields cite your work (your "audience")
- We identify which fields you cite (your "sources")
- Bridge fields are those that cite you but you don't cite back
- More bridge fields = your work is reaching new communities

**Example:** If you study computational biology and cite biology/CS papers, but physicists and social scientists start citing you, those are "bridge" fields - you're connecting communities!

---

## Composite Score

The final score combines all four metrics:

| Metric | Weight | Why this weight? |
|--------|--------|------------------|
| External Diversity | 35% | How others engage with your work (external impact) |
| Internal Diversity | 25% | Breadth within your own research (internal scope) |
| Reference Diversity | 30% | Shows breadth of your knowledge sources |
| Bridge Score | 10% | Captures unexpected connections |

**Interpretation:**
- **0-20%**: Low interdisciplinarity - focused within one field
- **20-50%**: Moderate - some cross-field activity
- **50-80%**: High - significant interdisciplinary impact
- **80-100%**: Very High - exceptional boundary-crossing research

---

## Data & Methodology

**Data Source:** [OpenAlex](https://openalex.org/) - an open catalog of 250M+ scholarly works

**Process:**
1. Fetch author's top 10 most-cited papers
2. For each paper, retrieve up to 10 recent citing papers
3. Extract abstracts and compute neural embeddings (using Sentence Transformers)
4. Analyze field distributions using OpenAlex's topic classifications
5. Calculate all four metrics and combine into composite score

**Limitations:**
- Only analyzes top 10 papers (most impactful work)
- Relies on OpenAlex's field classifications
- Embedding similarity is an approximation of semantic distance
                """)

        async def search_and_show_authors(query):
            if not query.strip():
                return gr.update(choices=[], visible=False), gr.update(visible=False), ""
            candidates = await search_authors(query, limit=5)
            if not candidates:
                return gr.update(choices=[], visible=False), gr.update(visible=False), ""
            choices = [(f"{c['name']} ({c['institution']}) - {c['works_count']} works" if c['institution'] else f"{c['name']} - {c['works_count']} works", c['id']) for c in candidates]
            return gr.update(choices=choices, value=choices[0][1], visible=True), gr.update(visible=True), candidates[0]['id']

        def create_progress_html(message: str) -> str:
            return f"""
            <div class="progress-container">
                <div class="progress-spinner"></div>
                <div class="progress-text">{message}</div>
                <div class="progress-subtext">This may take a minute for new authors</div>
            </div>
            """

        def show_progress():
            """Show progress indicator when analysis starts"""
            return gr.update(value=create_progress_html("Analyzing"), visible=True)

        async def run_analysis(author_id, cache_dir_val):
            if not author_id:
                return [gr.update()] * 16
            async with httpx.AsyncClient(timeout=30) as client:
                data = await fetch_with_retry(client, f"{OPENALEX_BASE_URL}/authors/{author_id}")
                author_name = data.get("display_name", "Unknown") if data else "Unknown"
            results = await analyze_author_pro(author_id, author_name, cache_dir_val)
            # results order: df_report, results_html, bullet_chart, df_top_papers, scatter, kde_fig, dispersion_chart,
            #                ref_diversity_chart, bridge_chart, field_breakdown_chart, keyword_chart, explanation, html_path
            return [
                gr.update(visible=False),  # progress_html - hide when done
                results[0],  # papers_table (df_report)
                gr.update(value=results[1], visible=True),  # results_html
                gr.update(value=results[2], visible=True),  # bullet_chart
                gr.update(value=results[3], visible=True),  # top_papers_table (df_top_papers)
                results[4],  # scatter_plot
                results[5],  # kde_plot
                gr.update(value=results[6], visible=True),  # dispersion_plot
                results[7],  # ref_diversity_plot
                results[8],  # bridge_plot
                results[9],  # field_breakdown_plot
                results[10], # keywords_plot
                gr.update(value=results[11], visible=True),  # methodology_md
                gr.update(value=results[12], visible=True),  # html_download
                gr.update(visible=True),  # charts_accordion
                gr.update(visible=True),  # papers_accordion
            ]

        search_btn.click(search_and_show_authors, [author_input], [author_dropdown, analyze_btn, selected_author_id])
        author_input.submit(search_and_show_authors, [author_input], [author_dropdown, analyze_btn, selected_author_id])
        author_dropdown.change(lambda x: x, [author_dropdown], [selected_author_id])

        # Show progress indicator first, then run analysis
        analyze_btn.click(show_progress, None, [progress_html])
        analyze_btn.click(run_analysis, [selected_author_id, cache_dir],
            [progress_html, papers_table, results_html, bullet_chart, top_papers_table, scatter_plot, kde_plot, dispersion_plot,
             ref_diversity_plot, bridge_plot, field_breakdown_plot, keywords_plot, methodology_md, html_download,
             charts_accordion, papers_accordion])

    return demo

if __name__ == "__main__":
    demo = create_pro_interface()
    demo.launch()
