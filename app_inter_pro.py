# app_inter_pro.py
"""
Interdisciplinary Index Analyzer PRO

Measures the cross-domain impact of academic research using 4 complementary metrics:
- Citation Diversity, Cluster Dispersion, Reference Diversity, and Bridge Score.
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
from sklearn.decomposition import PCA
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
    TOP_N_PAPERS = 10
    CITATIONS_PER_PAPER = 10
    REFERENCES_PER_PAPER = 20
    MAX_PAPERS_FETCH = 2000
    INDEX_THRESHOLDS = {"low": 20, "moderate": 50, "high": 80}
    KDE_BANDWIDTH = 0.2
    MAX_CONCURRENT_REQUESTS = 5
    REQUEST_TIMEOUT = 30

    # Fresh, vibrant color palette
    COLORS = {
        "primary": "#6C5CE7",      # Vibrant purple
        "secondary": "#00CEC9",    # Teal
        "accent": "#FD79A8",       # Pink
        "success": "#00B894",      # Mint green
        "warning": "#FDCB6E",      # Yellow
        "gradient_start": "#6C5CE7",
        "gradient_end": "#A29BFE",
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

def get_chart_layout(title: str, height: int = 400, width: int = None) -> dict:
    """Common chart layout with fresh, readable styling"""
    layout = {
        "title": dict(text=title, font=dict(size=18, color="#2D3436", family="Inter, sans-serif"), x=0.5, y=0.95),
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FAFAFA",
        "font": dict(family="Inter, sans-serif", color="#2D3436"),
        "margin": dict(t=80, b=50, l=60, r=40),
        "height": height,
        "autosize": True,
        "modebar": dict(orientation="v")
    }
    if width:
        layout["width"] = width
    return layout

def create_dispersion_chart(dispersion_data: dict, paper_titles: list[str]) -> go.Figure:
    embeddings = dispersion_data.get("embeddings")
    if embeddings is None or len(embeddings) < 2:
        return None

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    cluster_labels = dispersion_data.get("cluster_labels", [0] * len(coords))

    # Vibrant colors for clusters
    cluster_colors = ['#6C5CE7', '#00CEC9', '#FD79A8', '#FDCB6E', '#00B894', '#E17055']

    fig = go.Figure()
    for i, (x, y) in enumerate(coords):
        title = paper_titles[i] if i < len(paper_titles) else f"Paper {i+1}"
        cluster_idx = cluster_labels[i] % len(cluster_colors)
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=20, color=cluster_colors[cluster_idx], line=dict(width=2, color='white'), opacity=0.9),
            text=[title], hovertemplate=f"<b>{title[:50]}...</b><br>Cluster {cluster_labels[i]}<extra></extra>",
            showlegend=False
        ))

    fig.update_layout(**get_chart_layout(f"Research Landscape (Dispersion: {dispersion_data['dispersion_score']:.0f}%)", 500))
    fig.update_xaxes(title="Dimension 1", showgrid=True, gridcolor='#E8E8E8', zeroline=False)
    fig.update_yaxes(title="Dimension 2", showgrid=True, gridcolor='#E8E8E8', zeroline=False)
    return fig

def create_reference_diversity_chart(ref_diversity: dict) -> go.Figure:
    field_counts = ref_diversity.get("field_counts", {})
    if not field_counts:
        return None

    sorted_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:10][::-1]
    labels, values = [f[0] for f in sorted_fields], [f[1] for f in sorted_fields]

    fig = go.Figure(go.Bar(y=labels, x=values, orientation='h',
        marker=dict(color='#00CEC9', line=dict(width=0)),
        text=values, textposition='outside', textfont=dict(size=12, color='#2D3436')))

    fig.update_layout(**get_chart_layout(f"Fields You Reference (Diversity: {ref_diversity['diversity_index']:.0f}%)"))
    fig.update_xaxes(title="Count", showgrid=True, gridcolor='#E8E8E8')
    fig.update_yaxes(tickfont=dict(size=11))
    fig.update_layout(margin=dict(l=200))
    return fig

def create_bridge_chart(bridge_data: dict) -> go.Figure:
    source_fields, audience_fields = bridge_data.get("source_fields", {}), bridge_data.get("audience_fields", {})
    if not source_fields or not audience_fields:
        return None

    sorted_sources = sorted(source_fields.items(), key=lambda x: x[1], reverse=True)[:6]
    sorted_audience = sorted(audience_fields.items(), key=lambda x: x[1], reverse=True)[:6]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='You cite', y=[f[0] for f in sorted_sources[::-1]], x=[-f[1] for f in sorted_sources[::-1]],
        orientation='h', marker_color='#6C5CE7', text=[f[1] for f in sorted_sources[::-1]], textposition='inside'))
    fig.add_trace(go.Bar(name='Cite you', y=[f[0] for f in sorted_audience[::-1]], x=[f[1] for f in sorted_audience[::-1]],
        orientation='h', marker_color='#00B894', text=[f[1] for f in sorted_audience[::-1]], textposition='inside'))

    fig.update_layout(**get_chart_layout(f"Knowledge Flow (Bridge: {bridge_data['bridge_score']:.0f}%)"))
    fig.update_layout(barmode='overlay', legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center'))
    fig.update_xaxes(title="Count", zeroline=True, zerolinecolor='#CBD5E0', zerolinewidth=2)
    fig.update_layout(margin=dict(l=180))
    return fig

def create_citation_fields_chart(audience_fields: dict) -> go.Figure:
    if not audience_fields:
        return None

    sorted_fields = sorted(audience_fields.items(), key=lambda x: x[1], reverse=True)[:10][::-1]
    labels, values = [f[0] for f in sorted_fields], [f[1] for f in sorted_fields]
    total = sum(values)

    fig = go.Figure(go.Bar(y=labels, x=values, orientation='h',
        marker=dict(color='#FD79A8', line=dict(width=0)),
        text=[f"{v} ({v/total*100:.0f}%)" for v in values], textposition='outside', textfont=dict(size=11)))

    fig.update_layout(**get_chart_layout("Who Cites Your Work"))
    fig.update_xaxes(title="Citations", showgrid=True, gridcolor='#E8E8E8')
    fig.update_layout(margin=dict(l=200))
    return fig

def create_metrics_summary_chart(metrics: dict) -> go.Figure:
    metric_config = [
        {'name': 'Citation Diversity', 'key': 'citation_index', 'color': '#6C5CE7'},
        {'name': 'Cluster Dispersion', 'key': 'dispersion_score', 'color': '#00CEC9'},
        {'name': 'Reference Diversity', 'key': 'reference_diversity', 'color': '#FD79A8'},
        {'name': 'Bridge Score', 'key': 'bridge_score', 'color': '#00B894'},
    ]

    fig = go.Figure()

    # Background zones
    for i in range(4):
        y = 3 - i
        fig.add_shape(type="rect", x0=0, x1=20, y0=y-0.35, y1=y+0.35, fillcolor="rgba(200,200,200,0.15)", line_width=0, layer="below")
        fig.add_shape(type="rect", x0=20, x1=50, y0=y-0.35, y1=y+0.35, fillcolor="rgba(200,200,200,0.25)", line_width=0, layer="below")
        fig.add_shape(type="rect", x0=50, x1=80, y0=y-0.35, y1=y+0.35, fillcolor="rgba(200,200,200,0.35)", line_width=0, layer="below")
        fig.add_shape(type="rect", x0=80, x1=100, y0=y-0.35, y1=y+0.35, fillcolor="rgba(200,200,200,0.45)", line_width=0, layer="below")

    for i, cfg in enumerate(metric_config):
        value = metrics.get(cfg['key'], 0)
        y = 3 - i
        fig.add_trace(go.Bar(x=[value], y=[y], orientation='h', marker_color=cfg['color'], width=0.5,
            text=[f"{value:.0f}%"], textposition='outside', textfont=dict(size=14, color='#2D3436'),
            hovertemplate=f"<b>{cfg['name']}</b>: {value:.1f}%<extra></extra>", showlegend=False))

    fig.update_layout(**get_chart_layout("Interdisciplinarity Profile", 280))
    fig.update_xaxes(range=[0, 110], tickvals=[0, 20, 50, 80, 100], ticktext=['0', '20 (Low)', '50 (Mod)', '80 (High)', '100'])
    fig.update_yaxes(tickvals=[0,1,2,3], ticktext=[c['name'] for c in reversed(metric_config)], tickfont=dict(size=12))
    fig.update_layout(margin=dict(l=150, r=60))
    return fig

def create_scatter_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df['year'].astype(int).tolist(),
        y=df['paper_index'].tolist(),
        mode='markers',
        marker=dict(size=14, color='#6C5CE7', line=dict(width=2, color='white')),
        text=df['title'].tolist(),
        hovertemplate="<b>%{text}</b><br>Year: %{x}<br>Index: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(**get_chart_layout("Citation Diversity by Publication Year"))
    fig.update_xaxes(title="Year", showgrid=True, gridcolor='#E8E8E8', dtick=1)
    fig.update_yaxes(title="Interdisciplinary Index (%)", showgrid=True, gridcolor='#E8E8E8', range=[0, 100])
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
            line=dict(color='#00CEC9', width=3),
            fillcolor='rgba(0, 206, 201, 0.2)',
            hovertemplate="Similarity: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>"
        ))

        fig.update_layout(**get_chart_layout("Similarity Distribution (KDE)"))
        fig.update_xaxes(title="Cosine Similarity", showgrid=True, gridcolor='#E8E8E8')
        fig.update_yaxes(title="Density", showgrid=True, gridcolor='#E8E8E8')
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
        marker=dict(color='#FDCB6E', line=dict(width=0)),
        text=values, textposition='outside', textfont=dict(size=12)))

    fig.update_layout(**get_chart_layout("Top Keywords from Citing Papers"))
    fig.update_xaxes(title="Frequency", showgrid=True, gridcolor='#E8E8E8')
    fig.update_layout(margin=dict(l=160))
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
        ("Metrics Overview", bullet_fig, False),
        ("Research Landscape", dispersion_fig, True),  # Full width
        ("Citation Diversity by Year", scatter_fig, False),
        ("Similarity Distribution", kde_fig, False),
        ("Fields Referenced", ref_div_fig, False),
        ("Knowledge Flow", bridge_fig, False),
        ("Citing Fields", field_breakdown_fig, False),
        ("Top Keywords", keyword_fig, False),
    ]

    for title, fig, full_width in chart_configs:
        if fig is not None:
            try:
                chart_html = fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': True, 'responsive': True})
                css_class = "chart-container full-width" if full_width else "chart-container"
                charts_html.append(f'<div class="{css_class}"><h3>{title}</h3>{chart_html}</div>')
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
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }}
        body {{ background: linear-gradient(180deg, #F8F9FA 0%, #FFFFFF 100%); min-height: 100vh; padding: 40px 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 50%, #74B9FF 100%); padding: 40px; border-radius: 24px; text-align: center; color: white; margin-bottom: 32px; box-shadow: 0 20px 60px rgba(108, 92, 231, 0.3); }}
        .header h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 8px; }}
        .header p {{ opacity: 0.9; font-size: 1rem; }}
        .metrics-card {{ background: white; border-radius: 16px; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
        .metrics-card h2 {{ color: #2D3436; margin-bottom: 16px; font-size: 1.2rem; }}
        .composite-score {{ font-size: 3rem; font-weight: 700; color: #6C5CE7; text-align: center; margin: 16px 0; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 20px; }}
        .metric-item {{ background: #F8F9FA; padding: 16px; border-radius: 12px; text-align: center; }}
        .metric-item .label {{ font-size: 0.8rem; color: #636E72; margin-bottom: 4px; }}
        .metric-item .value {{ font-size: 1.5rem; font-weight: 600; color: #2D3436; }}
        .charts-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px; margin-bottom: 24px; }}
        .chart-container {{ background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
        .chart-container.full-width {{ grid-column: 1 / -1; }}
        .chart-container h3 {{ color: #2D3436; margin-bottom: 16px; font-size: 1.1rem; font-weight: 600; }}
        .chart-container .plotly-graph-div {{ width: 100% !important; }}
        .table-wrapper {{ max-height: 400px; overflow-y: auto; overflow-x: auto; border: 1px solid #E8E8E8; border-radius: 12px; }}
        .table-wrapper::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        .table-wrapper::-webkit-scrollbar-track {{ background: #F0F0F0; border-radius: 4px; }}
        .table-wrapper::-webkit-scrollbar-thumb {{ background: #C0C0C0; border-radius: 4px; }}
        .table-wrapper::-webkit-scrollbar-thumb:hover {{ background: #A0A0A0; }}
        .papers-table {{ width: 100%; border-collapse: collapse; font-size: 13px; table-layout: fixed; }}
        .papers-table th {{ background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 100%); color: white; padding: 12px 10px; text-align: left; font-size: 12px; font-weight: 600; position: sticky; top: 0; z-index: 10; }}
        .papers-table th:nth-child(1) {{ width: 40px; text-align: center; }}
        .papers-table th:nth-child(2) {{ width: 45%; }}
        .papers-table th:nth-child(3) {{ width: 70px; text-align: center; }}
        .papers-table th:nth-child(4) {{ width: 100px; text-align: center; }}
        .papers-table th:nth-child(5) {{ width: 80px; text-align: center; }}
        .papers-table td {{ padding: 10px 12px; border-bottom: 1px solid #E8E8E8; white-space: nowrap; }}
        .papers-table td:nth-child(2) {{ white-space: normal; word-wrap: break-word; }}
        .papers-table tr:nth-child(even) td {{ background: #F8F9FA; }}
        .papers-table tr:hover td {{ background: #EEF0FF; }}
        .footer {{ text-align: center; color: #636E72; font-size: 0.85rem; margin-top: 40px; padding: 20px; }}
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
                    <div class="label">Citation Diversity</div>
                    <div class="value">{metrics.get('citation_index', 0):.1f}%</div>
                </div>
                <div class="metric-item">
                    <div class="label">Cluster Dispersion</div>
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
</body>
</html>'''

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return html_path

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

async def analyze_author_pro(author_id: str, author_name: str, cache_dir: str = None, progress_callback=None):
    logger.info(f"Starting analysis for: {author_name} ({author_id})")
    start_time = datetime.now()

    # Progress helper that safely handles None
    def update_progress(value, desc=""):
        if progress_callback is not None:
            try:
                progress_callback(value, desc=desc)
            except Exception:
                pass

    update_progress(0.05, "Checking cache...")
    if not is_cached(author_id, cache_dir):
        update_progress(0.08, "Building cache (first time)...")
        await build_author_cache(author_id, cache_dir=cache_dir)

    update_progress(0.15, "Loading data...")
    data = load_author_cache(author_id, cache_dir)
    top_papers = data["top_papers"]

    results, all_keywords, similarities_all, all_citing_texts, all_abstracts, paper_titles = [], [], [], [], [], []

    async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
        total = len(top_papers)
        for idx, paper in enumerate(top_papers):
            update_progress(0.15 + (0.25 * idx / total), f"Analyzing paper {idx+1}/{total}...")
            abs_orig = reconstruct_abstract(paper.get("abstract_inverted_index"))
            if not abs_orig:
                continue
            all_abstracts.append(abs_orig)
            paper_titles.append(paper.get("title", f"Paper {idx+1}"))
            citing_texts = await fetch_citing_abstracts_parallel(paper.get("citing_ids", []), client)
            if citing_texts:
                avg_sim, idx_val, count, sims = calculate_similarity_and_index(abs_orig, citing_texts)
                results.append({"title": paper["title"], "year": paper["year"], "paper_index": idx_val * 100, "citation_count": paper.get("citation_count", 0)})
                similarities_all.extend(sims)
                all_citing_texts.extend(citing_texts)

        update_progress(0.45, "Computing embedding dispersion...")
        dispersion_data = calculate_embedding_dispersion(all_abstracts)

        update_progress(0.55, "Analyzing reference diversity...")
        ref_diversity = await calculate_reference_diversity(author_id, top_papers, client)

        update_progress(0.70, "Computing bridge score...")
        bridge_data = await calculate_bridge_score(author_id, top_papers, client)

    update_progress(0.82, "Extracting keywords...")
    for text in all_citing_texts[:50]:
        try:
            kws = kw_model.extract_keywords(text, top_n=3, stop_words='english')
            all_keywords.extend([kw[0] for kw in kws])
        except:
            pass

    update_progress(0.88, "Creating visualizations...")
    df = pd.DataFrame(results).dropna().sort_values(by="paper_index", ascending=False)
    citation_index = df['paper_index'].mean() if not df.empty else 0

    all_metrics = {'citation_index': citation_index, 'dispersion_score': dispersion_data['dispersion_score'],
                   'reference_diversity': ref_diversity['diversity_index'], 'bridge_score': bridge_data['bridge_score']}
    composite_score = citation_index * 0.30 + dispersion_data['dispersion_score'] * 0.25 + ref_diversity['diversity_index'] * 0.25 + bridge_data['bridge_score'] * 0.20

    scatter = create_scatter_chart(df) if not df.empty else None
    kde_fig = create_kde_chart(similarities_all)
    dispersion_chart = create_dispersion_chart(dispersion_data, paper_titles)
    ref_diversity_chart = create_reference_diversity_chart(ref_diversity)
    bridge_chart = create_bridge_chart(bridge_data)
    field_breakdown_chart = create_citation_fields_chart(bridge_data.get('audience_fields', {}))
    bullet_chart = create_metrics_summary_chart(all_metrics)
    keyword_chart = create_keywords_chart(Counter(all_keywords).most_common(10))

    update_progress(0.94, "Generating PDF...")
    # Create top papers table sorted by citation diversity (descending)
    df_top_papers = df[["title", "year", "paper_index", "citation_count"]].copy()
    df_top_papers = df_top_papers.sort_values(by="paper_index", ascending=False).reset_index(drop=True)
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
    <div style='background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 100%); padding: 28px; border-radius: 20px; color: white; margin: 16px 0; box-shadow: 0 10px 40px rgba(108, 92, 231, 0.3);'>
        <h2 style='margin: 0 0 20px 0; font-size: 1.5em; font-weight: 600; text-align: center;'>Analysis Complete</h2>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px;'>
            <div style='background: rgba(255,255,255,0.2); padding: 20px; border-radius: 16px; text-align: center; backdrop-filter: blur(10px);'>
                <div style='font-size: 0.8em; opacity: 0.9; margin-bottom: 6px;'>Composite Score</div>
                <div style='font-size: 2.8em; font-weight: 700;'>{composite_score:.0f}%</div>
                <div style='font-size: 0.9em; background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; display: inline-block; margin-top: 6px;'>{category}</div>
            </div>
            <div style='background: rgba(255,255,255,0.15); padding: 20px; border-radius: 16px; text-align: center;'>
                <div style='font-size: 0.8em; opacity: 0.9; margin-bottom: 6px;'>Citation Diversity</div>
                <div style='font-size: 2.2em; font-weight: 600;'>{citation_index:.0f}%</div>
            </div>
        </div>
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;'>
            <div style='background: rgba(255,255,255,0.1); padding: 14px; border-radius: 12px; text-align: center;'>
                <div style='font-size: 0.75em; opacity: 0.85;'>Dispersion</div>
                <div style='font-size: 1.5em; font-weight: 600;'>{dispersion_data['dispersion_score']:.0f}%</div>
                <div style='font-size: 0.7em; opacity: 0.7;'>{dispersion_data['n_clusters']} clusters</div>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 14px; border-radius: 12px; text-align: center;'>
                <div style='font-size: 0.75em; opacity: 0.85;'>Ref. Diversity</div>
                <div style='font-size: 1.5em; font-weight: 600;'>{ref_diversity['diversity_index']:.0f}%</div>
                <div style='font-size: 0.7em; opacity: 0.7;'>{ref_diversity['unique_fields']} fields</div>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 14px; border-radius: 12px; text-align: center;'>
                <div style='font-size: 0.75em; opacity: 0.85;'>Bridge Score</div>
                <div style='font-size: 1.5em; font-weight: 600;'>{bridge_data['bridge_score']:.0f}%</div>
                <div style='font-size: 0.7em; opacity: 0.7;'>{len(bridge_data.get('bridged_fields', []))} bridged</div>
            </div>
        </div>
    </div>
    """

    explanation = """### How It Works

**Composite Score** is a weighted combination of four metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| Citation Diversity | 30% | How different are papers that cite you from your own work |
| Cluster Dispersion | 25% | How spread out your research topics are |
| Reference Diversity | 25% | Variety of fields you draw knowledge from |
| Bridge Score | 20% | Fields that cite you but you don't cite back |

**Interpretation**: Low (0-20%) → Moderate (20-50%) → High (50-80%) → Very High (80-100%)
"""

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Analysis completed in {elapsed:.1f}s")
    update_progress(1.0, "Done!")

    return (df_report, results_html, bullet_chart, df_top_papers, scatter, kde_fig, dispersion_chart, ref_diversity_chart, bridge_chart, field_breakdown_chart, keyword_chart, explanation, html_path)

# ============================================================================
# GRADIO UI - FRESH MODERN THEME
# ============================================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

.gradio-container {
    background: linear-gradient(180deg, #F8F9FA 0%, #FFFFFF 100%) !important;
    max-width: 1600px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
}

.header-card {
    background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 50%, #74B9FF 100%);
    padding: 40px;
    border-radius: 24px;
    text-align: center;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 20px 60px rgba(108, 92, 231, 0.3);
}

.header-card h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
}

.header-card p {
    font-size: 1rem;
    opacity: 0.95;
    margin: 0;
    font-weight: 400;
}

/* Inputs */
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background: white !important;
    border: 2px solid #E8E8E8 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    font-size: 15px !important;
    transition: all 0.2s ease !important;
}

.gradio-container input:focus, .gradio-container textarea:focus {
    border-color: #6C5CE7 !important;
    box-shadow: 0 0 0 4px rgba(108, 92, 231, 0.1) !important;
}

/* Labels */
.gradio-container label {
    font-weight: 500 !important;
    color: #2D3436 !important;
    font-size: 0.9rem !important;
    margin-bottom: 6px !important;
}

/* Primary Button */
.primary-btn, .gradio-container button.primary {
    background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px 32px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(108, 92, 231, 0.4) !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

.primary-btn:hover, .gradio-container button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(108, 92, 231, 0.5) !important;
}

/* Accordion */
.gradio-container .accordion {
    background: white !important;
    border: 1px solid #E8E8E8 !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04) !important;
}

/* Top Papers Table - Scrollable with Fixed Header */
#top-papers-table {
    font-size: 13px !important;
    max-height: 450px !important;
    overflow-y: auto !important;
    overflow-x: auto !important;
    border: 1px solid #E8E8E8 !important;
    border-radius: 12px !important;
}

#top-papers-table table {
    width: 100% !important;
    border-collapse: collapse !important;
    table-layout: fixed !important;
}

/* Fixed column widths for alignment */
#top-papers-table th:nth-child(1),
#top-papers-table td:nth-child(1) { width: 40px !important; min-width: 40px !important; }
#top-papers-table th:nth-child(2),
#top-papers-table td:nth-child(2) { width: 45% !important; min-width: 200px !important; }
#top-papers-table th:nth-child(3),
#top-papers-table td:nth-child(3) { width: 70px !important; min-width: 70px !important; }
#top-papers-table th:nth-child(4),
#top-papers-table td:nth-child(4) { width: 100px !important; min-width: 100px !important; }
#top-papers-table th:nth-child(5),
#top-papers-table td:nth-child(5) { width: 90px !important; min-width: 90px !important; }

#top-papers-table thead tr th {
    background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 10px !important;
    font-size: 12px !important;
    white-space: normal !important;
    word-wrap: break-word !important;
    text-align: left !important;
    border: none !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 10 !important;
}

#top-papers-table tbody tr td {
    padding: 10px 14px !important;
    border-bottom: 1px solid #E8E8E8 !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    vertical-align: middle !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Paper title column - allow wrapping */
#top-papers-table tbody tr td:nth-child(2) {
    white-space: normal !important;
    word-wrap: break-word !important;
}

/* Center align numeric columns */
#top-papers-table th:nth-child(1),
#top-papers-table td:nth-child(1),
#top-papers-table th:nth-child(3),
#top-papers-table td:nth-child(3),
#top-papers-table th:nth-child(4),
#top-papers-table td:nth-child(4),
#top-papers-table th:nth-child(5),
#top-papers-table td:nth-child(5) {
    text-align: center !important;
}

#top-papers-table tbody tr:nth-child(even) td { background: #F8F9FA !important; }
#top-papers-table tbody tr:hover td { background: #EEF0FF !important; }

/* Custom scrollbar */
#top-papers-table::-webkit-scrollbar { width: 8px; height: 8px; }
#top-papers-table::-webkit-scrollbar-track { background: #F0F0F0; border-radius: 4px; }
#top-papers-table::-webkit-scrollbar-thumb { background: #C0C0C0; border-radius: 4px; }
#top-papers-table::-webkit-scrollbar-thumb:hover { background: #A0A0A0; }

/* Other tables default styling */
.gradio-container table { border-radius: 8px !important; overflow: hidden !important; }
.gradio-container th { background: #F8F9FA !important; color: #2D3436 !important; font-weight: 600 !important; padding: 10px !important; }
.gradio-container td { padding: 8px 10px !important; border-bottom: 1px solid #F0F0F0 !important; }

/* Tabs */
.gradio-container .tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    color: #636E72 !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    transition: all 0.2s !important;
}

.gradio-container .tab-nav button.selected {
    color: #6C5CE7 !important;
    border-bottom-color: #6C5CE7 !important;
}

/* Markdown */
.gradio-container .markdown { color: #2D3436 !important; line-height: 1.7 !important; }
.gradio-container .markdown h1, .gradio-container .markdown h2, .gradio-container .markdown h3 { color: #2D3436 !important; font-weight: 600 !important; }
.gradio-container .markdown code { background: #F0F0F0 !important; color: #6C5CE7 !important; padding: 2px 8px !important; border-radius: 6px !important; }

/* File download */
.gradio-container .file-preview { background: white !important; border: 2px dashed #E8E8E8 !important; border-radius: 12px !important; }

/* Animated Progress Indicator with Dots */
.progress-container {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    background: linear-gradient(135deg, #F8F9FA 0%, #FFFFFF 100%);
    border-radius: 16px;
    border: 1px solid #E8E8E8;
    margin: 16px 0;
}

.progress-text {
    font-size: 1rem;
    font-weight: 500;
    color: #6C5CE7;
    margin-right: 4px;
}

.dots {
    display: inline-flex;
    gap: 4px;
}

.dot {
    width: 6px;
    height: 6px;
    background-color: #6C5CE7;
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
}

.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }
.dot:nth-child(3) { animation-delay: 0s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

"""

def create_pro_interface():
    with gr.Blocks(css=custom_css, title="Interdisciplinary Index Analyzer", theme=gr.themes.Soft()) as demo:
        cache_dir = gr.State(value=create_session_cache_dir)
        selected_author_id = gr.State(value="")

        gr.HTML("""
        <div class="header-card">
            <h1>Interdisciplinary Index Analyzer</h1>
            <p>Measure the cross-domain impact of academic research through citation and semantic analysis</p>
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
                    bullet_chart = gr.Plot(label="Metrics Overview", visible=False)

                # Top papers table sorted by citation diversity
                top_papers_table = gr.Dataframe(
                    label="Top 10 Papers by Citation Diversity Score",
                    visible=False,
                    wrap=True,
                    elem_id="top-papers-table"
                )

                # Full width dispersion
                dispersion_plot = gr.Plot(label="Research Landscape", visible=False)

                with gr.Accordion("Detailed Charts", open=True, visible=False) as charts_accordion:
                    with gr.Row():
                        scatter_plot = gr.Plot(label="Index by Year")
                        kde_plot = gr.Plot(label="Similarity Distribution")
                    with gr.Row():
                        ref_diversity_plot = gr.Plot(label="Fields Referenced")
                        bridge_plot = gr.Plot(label="Knowledge Flow")
                    with gr.Row():
                        field_breakdown_plot = gr.Plot(label="Citing Fields")
                        keywords_plot = gr.Plot(label="Keywords")

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

### 1. Citation Diversity (30% of composite score)

**What it measures:** How different are the papers citing your work from your original papers?

**How it works:**
- We take your top 10 most-cited papers and look at who cites them
- Using AI (neural embeddings), we convert each paper's abstract into a mathematical representation
- We measure how "far apart" your papers are from the papers that cite them
- If papers from very different topics cite you, your score is higher

**Example:** If you write about genetics and get cited by ecology, public health, and computer science papers, your citation diversity is high.

---

### 2. Cluster Dispersion (25% of composite score)

**What it measures:** How spread out are your own research topics?

**How it works:**
- We take all your paper abstracts and convert them to embeddings
- Using clustering (K-means algorithm), we group similar papers together
- We then measure how spread apart these clusters are using PCA (a dimensionality reduction technique)
- More spread = higher dispersion = more interdisciplinary

**Example:** A researcher who publishes in both "machine learning" and "neuroscience" clusters will have higher dispersion than someone who only publishes in "machine learning applications."

---

### 3. Reference Diversity (25% of composite score)

**What it measures:** How many different fields do you draw knowledge from?

**How it works:**
- We look at the references in your papers (who do you cite?)
- Each reference is tagged with its academic field (e.g., Biology, Computer Science, Economics)
- We calculate Shannon entropy - a measure of diversity from information theory
- Higher entropy = more evenly distributed across fields = more diverse

**Example:** If you cite 50 biology papers and 50 economics papers, your entropy is higher than citing 95 biology papers and 5 economics papers.

---

### 4. Bridge Score (20% of composite score)

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
| Citation Diversity | 30% | Most direct measure of cross-field impact |
| Cluster Dispersion | 25% | Shows breadth of your own research |
| Reference Diversity | 25% | Shows breadth of your knowledge sources |
| Bridge Score | 20% | Captures unexpected connections |

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
                <span class="progress-text">{message}</span>
                <span class="dots">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </span>
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
            results = await analyze_author_pro(author_id, author_name, cache_dir_val, progress_callback=None)
            # results order: df_report, results_html, bullet_chart, df_top_papers, scatter, kde_fig, dispersion_chart,
            #                ref_diversity_chart, bridge_chart, field_breakdown_chart, keyword_chart, explanation, pdf_path
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
