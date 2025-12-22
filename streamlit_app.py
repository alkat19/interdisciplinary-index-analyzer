"""
Streamlit version of the Interdisciplinary Index Analyzer.
This is a wrapper around the core analysis logic for deployment on Streamlit Cloud.
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
import logging
import os
import tempfile
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from collections import Counter
from scipy.stats import gaussian_kde
import httpx

from reproducible_cache import (
    is_cached, load_author_cache, build_author_cache, fetch_paper_by_id,
    get_cache_timestamp, clear_author_cache, fetch_citing_papers_with_topics,
    extract_fields_from_papers
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Interdisciplinary Index Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MODEL_NAME = 'minishlab/potion-base-32M'
    DEVICE = "cpu"
    TOP_N_PAPERS = 10
    CITATIONS_PER_PAPER = 10
    INDEX_THRESHOLDS = {"low": 20, "moderate": 50, "high": 80}
    KDE_BANDWIDTH = 0.2
    REQUEST_TIMEOUT = 30
    COLORS = {
        "primary": "#6366f1",
        "secondary": "#8b5cf6",
        "accent": "#06b6d4",
    }

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL LOADING (cached)
# ============================================================================

@st.cache_resource
def load_models():
    """Load ML models once and cache them"""
    logger.info(f"Loading model: {Config.MODEL_NAME}")
    sentence_model = SentenceTransformer(Config.MODEL_NAME, device=Config.DEVICE)
    kw_model = KeyBERT(sentence_model)
    logger.info("Models loaded successfully")
    return sentence_model, kw_model

sentence_model, kw_model = load_models()

# ============================================================================
# SESSION STATE
# ============================================================================

if 'cache_dir' not in st.session_state:
    session_id = str(uuid.uuid4())[:8]
    st.session_state.cache_dir = os.path.join(tempfile.gettempdir(), f"interdisciplinary_cache_{session_id}")
    os.makedirs(st.session_state.cache_dir, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def categorize_index(pct: float) -> str:
    thresholds = Config.INDEX_THRESHOLDS
    if pct <= thresholds["low"]:
        return "Low"
    elif pct <= thresholds["moderate"]:
        return "Moderate"
    elif pct <= thresholds["high"]:
        return "High"
    else:
        return "Very High"

def get_category_emoji(category: str) -> str:
    return {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Very High": "🔴"}.get(category, "⚪")

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
        logger.error(f"Error reconstructing abstract: {e}")
        return None

def calculate_similarity_and_index(original_abstract: str, citing_abstracts: list[str]) -> tuple:
    with torch.no_grad():
        e_orig = sentence_model.encode(original_abstract, convert_to_tensor=True, device=Config.DEVICE)
        e_cite = sentence_model.encode(citing_abstracts, convert_to_tensor=True, device=Config.DEVICE)
        sims = util.cos_sim(e_orig, e_cite)[0].cpu().tolist()
        sims = [max(0.0, s) for s in sims]
        avg_sim = sum(sims) / len(sims)
        idx = 1.0 - avg_sim
        return avg_sim, idx, len(sims), sims

async def search_authors(author_query: str, limit: int = 5) -> list[dict]:
    async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
        try:
            r = await client.get(
                "https://api.openalex.org/authors",
                params={"search": author_query, "per_page": limit},
                headers={"User-Agent": "InterdisciplinaryIndexApp/1.0 (mailto:research@example.com)"}
            )
            r.raise_for_status()
            data = r.json()

            candidates = []
            for result in data.get("results", []):
                institution = ""
                if result.get("last_known_institution"):
                    institution = result["last_known_institution"].get("display_name", "")
                candidates.append({
                    "id": result["id"].split("/")[-1],
                    "name": result["display_name"],
                    "works_count": result.get("works_count", 0),
                    "cited_by_count": result.get("cited_by_count", 0),
                    "institution": institution,
                })
            return candidates
        except Exception as e:
            logger.error(f"Error searching authors: {e}")
            return []

async def fetch_citing_abstracts_parallel(citing_ids: list[str], client: httpx.AsyncClient) -> list[str]:
    if not citing_ids:
        return []
    tasks = [fetch_paper_by_id(cid, client) for cid in citing_ids]
    papers = await asyncio.gather(*tasks, return_exceptions=True)

    abstracts = []
    for paper in papers:
        if isinstance(paper, dict):
            abstract = reconstruct_abstract(paper.get("abstract_inverted_index"))
            if abstract:
                abstracts.append(abstract)
    return abstracts

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_field_breakdown_chart(field_counts: dict) -> go.Figure:
    """Create a horizontal bar chart showing citation field distribution"""
    if not field_counts:
        return None
    sorted_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    # Reverse for horizontal bar (bottom to top)
    sorted_fields = sorted_fields[::-1]
    labels = [f[0] for f in sorted_fields]
    values = [f[1] for f in sorted_fields]

    # Color gradient from light to dark
    colors = px.colors.sequential.Purples[3:3+len(labels)] if len(labels) <= 7 else px.colors.sequential.Purples[2:]

    fig = go.Figure(data=[go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker_color=colors[:len(labels)] if len(colors) >= len(labels) else '#6366f1',
        text=[str(v) for v in values],
        textposition='outside',
        textfont=dict(size=11, color='#64748b'),
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>"
    )])
    fig.update_layout(
        title=dict(text="Citation Fields Breakdown", font=dict(size=16, color='#1e293b')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=40, l=150, r=60),
        xaxis=dict(title="Number of Citing Papers", showgrid=True, gridcolor='#f1f5f9', zeroline=False),
        yaxis=dict(title="", tickfont=dict(size=11)),
        font=dict(family="Inter")
    )
    return fig

def create_keywords_chart(keyword_counts: list) -> go.Figure:
    if not keyword_counts:
        return None
    top_keywords = keyword_counts[:10][::-1]
    labels = [kw[0] for kw in top_keywords]
    values = [kw[1] for kw in top_keywords]

    fig = go.Figure(data=[go.Bar(
        y=labels, x=values, orientation='h',
        marker_color='#8b5cf6',
        text=[str(v) for v in values],
        textposition='outside'
    )])
    fig.update_layout(
        title="Top Keywords from Citing Papers",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=40, l=150, r=60),
        xaxis=dict(title="Frequency", showgrid=True, gridcolor='#f1f5f9'),
        yaxis=dict(title="")
    )
    return fig

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

async def analyze_author(author_id: str, author_name: str, cache_dir: str, progress_bar, status_text):
    """Main analysis function"""

    status_text.text("Checking cache...")
    progress_bar.progress(0.05)

    was_cached = is_cached(author_id, cache_dir)

    if not was_cached:
        status_text.text("Building author cache (first-time analysis)...")
        progress_bar.progress(0.1)
        await build_author_cache(author_id, cache_dir=cache_dir)

    status_text.text("Loading cached data...")
    progress_bar.progress(0.2)

    data = load_author_cache(author_id, cache_dir)
    top_papers = data["top_papers"]
    cache_time = get_cache_timestamp(author_id, cache_dir)

    results, all_keywords, similarities_all = [], [], []
    all_citing_texts = []
    field_counts = {}

    async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
        total_papers = len(top_papers)

        for paper_idx, paper in enumerate(top_papers):
            progress = 0.2 + (0.5 * paper_idx / total_papers)
            progress_bar.progress(progress)
            status_text.text(f"Processing paper {paper_idx + 1}/{total_papers}...")

            abs_orig = reconstruct_abstract(paper["abstract_inverted_index"])
            if not abs_orig:
                continue

            citing_texts = await fetch_citing_abstracts_parallel(paper["citing_ids"], client)

            # Field breakdown
            citing_papers_with_topics = await fetch_citing_papers_with_topics(paper["citing_ids"], client)
            paper_fields = extract_fields_from_papers(citing_papers_with_topics)
            for field, count in paper_fields.items():
                field_counts[field] = field_counts.get(field, 0) + count

            if citing_texts:
                avg_sim, idx_value, count, sims = calculate_similarity_and_index(abs_orig, citing_texts)

                results.append({
                    "title": paper["title"],
                    "year": paper["year"],
                    "paper_index": idx_value * 100,
                    "citation_count": paper.get("citation_count", 0)
                })

                similarities_all.extend(sims)
                all_citing_texts.extend(citing_texts)

    # Keyword extraction
    status_text.text("Extracting keywords...")
    progress_bar.progress(0.75)

    for text in all_citing_texts:
        try:
            kws = kw_model.extract_keywords(text, top_n=3, stop_words='english')
            all_keywords.extend([kw[0] for kw in kws])
        except:
            pass

    status_text.text("Generating visualizations...")
    progress_bar.progress(0.85)

    df = pd.DataFrame(results).dropna().sort_values(by="paper_index", ascending=False)

    if df.empty:
        return None, "No papers with valid data found for this author."

    author_index = df['paper_index'].mean()
    max_index = df['paper_index'].max()
    min_index = df['paper_index'].min()

    # Scatter plot
    scatter = px.scatter(
        df, x="year", y="paper_index",
        hover_name="title",
        size="citation_count" if "citation_count" in df.columns else None,
        color_discrete_sequence=[Config.COLORS["primary"]]
    )
    scatter.update_layout(
        title="Interdisciplinary Index by Publication Year",
        xaxis_title="Year",
        yaxis_title="Index (%)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # KDE plot
    if len(similarities_all) > 1:
        kde = gaussian_kde(similarities_all, bw_method=Config.KDE_BANDWIDTH)
        x_vals = np.linspace(min(similarities_all), max(similarities_all), 200)
        y_vals = kde(x_vals)
        kde_fig = go.Figure()
        kde_fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, fill='tozeroy', mode='lines',
            line=dict(color='#8b5cf6', width=3),
            fillcolor='rgba(139, 92, 246, 0.2)'
        ))
        kde_fig.update_layout(
            title="Similarity Distribution",
            xaxis_title="Cosine Similarity",
            yaxis_title="Density",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    else:
        kde_fig = go.Figure()

    # Field chart
    field_chart = create_field_breakdown_chart(field_counts)

    # Keywords chart
    keyword_counts = Counter(all_keywords).most_common(10)
    keyword_chart = create_keywords_chart(keyword_counts)

    progress_bar.progress(1.0)
    status_text.text("Analysis complete!")

    return {
        "df": df,
        "author_index": author_index,
        "max_index": max_index,
        "min_index": min_index,
        "category": categorize_index(author_index),
        "scatter": scatter,
        "kde_fig": kde_fig,
        "field_chart": field_chart,
        "keyword_chart": keyword_chart,
        "was_cached": was_cached,
        "cache_time": cache_time
    }, None

# ============================================================================
# STREAMLIT UI
# ============================================================================

def display_author_results(result, author_name):
    """Display analysis results for a single author"""
    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Author Index",
            f"{result['author_index']:.1f}%",
            help="Average interdisciplinary index across all papers"
        )
        st.caption(f"{get_category_emoji(result['category'])} {result['category']}")
    with col2:
        st.metric(
            "Highest Paper",
            f"{result['max_index']:.1f}%",
            help="Most interdisciplinary paper"
        )
    with col3:
        st.metric(
            "Papers Analyzed",
            len(result['df']),
            help="Number of papers included in analysis"
        )

    # Cache status
    if result['was_cached'] and result['cache_time']:
        st.info(f"📦 Using cached data from {result['cache_time'].strftime('%b %d, %Y')}")
    else:
        st.success("✨ Fresh analysis completed")

    st.markdown("---")

    # Charts in 2x2 grid
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(result['scatter'], use_container_width=True)
    with col2:
        st.plotly_chart(result['kde_fig'], use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if result['field_chart']:
            st.plotly_chart(result['field_chart'], use_container_width=True)
    with col4:
        if result['keyword_chart']:
            st.plotly_chart(result['keyword_chart'], use_container_width=True)

    # Data table
    st.markdown("### 📄 Paper Details")
    display_df = result['df'].rename(columns={
        "title": "Title",
        "year": "Year",
        "paper_index": "Index (%)",
        "citation_count": "Citations"
    })
    st.dataframe(display_df, use_container_width=True)

    # Download CSV
    csv = display_df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        csv,
        f"interdisciplinary_{author_name.replace(' ', '_')}.csv",
        "text/csv"
    )


def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .stDataFrame {
        border-radius: 0.5rem;
    }
    .comparison-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .comparison-card-cyan {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Interdisciplinary Index Analyzer</h1>
        <p>Measure the cross-domain impact of academic research by analyzing citation patterns and semantic similarity</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔬 Single Author", "⚖️ Compare Authors", "📘 About"])

    # ==================== TAB 1: SINGLE AUTHOR ====================
    with tab1:
        st.markdown("### Search for Researcher")
        col1, col2 = st.columns([3, 1])
        with col1:
            author_name = st.text_input("Enter author name", placeholder="e.g., Geoffrey Hinton", key="single_author")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_button = st.button("🔬 Analyze", type="primary", use_container_width=True, key="single_search")

        # Quick examples
        st.markdown("**Quick examples:**")
        example_cols = st.columns(4)
        example_authors = ["Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Fei-Fei Li"]
        for i, example in enumerate(example_authors):
            with example_cols[i]:
                if st.button(example, key=f"example_{example}", use_container_width=True):
                    st.session_state.single_author = example
                    st.rerun()

        if search_button and author_name:
            # Search for author
            with st.spinner("Searching for author..."):
                candidates = asyncio.run(search_authors(author_name, limit=5))

            if not candidates:
                st.error(f"Could not find author '{author_name}' in OpenAlex.")
            else:
                # Show disambiguation if multiple results
                if len(candidates) > 1:
                    selected_idx = st.selectbox(
                        "Multiple authors found. Please select:",
                        range(len(candidates)),
                        format_func=lambda i: f"{candidates[i]['name']} | {candidates[i]['institution']} | {candidates[i]['works_count']} works"
                    )
                    selected_author = candidates[selected_idx]
                else:
                    selected_author = candidates[0]

                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Run analysis
                result, error = asyncio.run(analyze_author(
                    selected_author['id'],
                    selected_author['name'],
                    st.session_state.cache_dir,
                    progress_bar,
                    status_text
                ))

                if error:
                    st.error(error)
                else:
                    st.markdown("---")
                    display_author_results(result, selected_author['name'])

    # ==================== TAB 2: COMPARE AUTHORS ====================
    with tab2:
        st.markdown("### Compare Two Researchers")
        st.markdown("Analyze and compare the interdisciplinary profiles of two researchers side by side.")

        col1, col2 = st.columns(2)
        with col1:
            author1_name = st.text_input("First Author", placeholder="e.g., Geoffrey Hinton", key="author1")
        with col2:
            author2_name = st.text_input("Second Author", placeholder="e.g., Yann LeCun", key="author2")

        compare_button = st.button("⚖️ Compare Authors", type="primary", use_container_width=True)

        if compare_button and author1_name and author2_name:
            # Search for both authors
            with st.spinner("Searching for authors..."):
                candidates1 = asyncio.run(search_authors(author1_name, limit=1))
                candidates2 = asyncio.run(search_authors(author2_name, limit=1))

            if not candidates1:
                st.error(f"Could not find author '{author1_name}' in OpenAlex.")
            elif not candidates2:
                st.error(f"Could not find author '{author2_name}' in OpenAlex.")
            else:
                author1 = candidates1[0]
                author2 = candidates2[0]

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Analyze first author
                status_text.text(f"Analyzing {author1['name']}...")
                result1, error1 = asyncio.run(analyze_author(
                    author1['id'], author1['name'],
                    st.session_state.cache_dir,
                    progress_bar, status_text
                ))

                if error1:
                    st.error(f"Error analyzing {author1['name']}: {error1}")
                else:
                    # Reset progress for second author
                    progress_bar.progress(0)
                    status_text.text(f"Analyzing {author2['name']}...")

                    result2, error2 = asyncio.run(analyze_author(
                        author2['id'], author2['name'],
                        st.session_state.cache_dir,
                        progress_bar, status_text
                    ))

                    if error2:
                        st.error(f"Error analyzing {author2['name']}: {error2}")
                    else:
                        status_text.text("Comparison complete!")
                        progress_bar.progress(1.0)

                        st.markdown("---")

                        # Side by side comparison cards
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(f"""
                            <div class="comparison-card">
                                <h3>{author1['name']}</h3>
                                <h1>{result1['author_index']:.1f}%</h1>
                                <p>{get_category_emoji(result1['category'])} {result1['category']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.metric("Max Index", f"{result1['max_index']:.1f}%")
                            st.metric("Papers", len(result1['df']))

                        with col2:
                            st.markdown(f"""
                            <div class="comparison-card-cyan">
                                <h3>{author2['name']}</h3>
                                <h1>{result2['author_index']:.1f}%</h1>
                                <p>{get_category_emoji(result2['category'])} {result2['category']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.metric("Max Index", f"{result2['max_index']:.1f}%")
                            st.metric("Papers", len(result2['df']))

                        # Comparison summary
                        diff = abs(result1['author_index'] - result2['author_index'])
                        more_interdisciplinary = author1['name'] if result1['author_index'] > result2['author_index'] else author2['name']
                        st.info(f"**Difference:** {diff:.1f} percentage points | **More Interdisciplinary:** {more_interdisciplinary}")

                        st.markdown("---")

                        # Scatter plots side by side
                        st.markdown("### Publication Timeline")
                        col1, col2 = st.columns(2)
                        with col1:
                            result1['scatter'].update_layout(title=f"{author1['name']} - Index Over Time")
                            st.plotly_chart(result1['scatter'], use_container_width=True)
                        with col2:
                            result2['scatter'].update_layout(title=f"{author2['name']} - Index Over Time")
                            st.plotly_chart(result2['scatter'], use_container_width=True)

                        # Field breakdown side by side
                        st.markdown("### Citation Field Distribution")
                        col1, col2 = st.columns(2)
                        with col1:
                            if result1['field_chart']:
                                result1['field_chart'].update_layout(title=f"{author1['name']} - Citing Fields")
                                st.plotly_chart(result1['field_chart'], use_container_width=True)
                        with col2:
                            if result2['field_chart']:
                                result2['field_chart'].update_layout(title=f"{author2['name']} - Citing Fields")
                                st.plotly_chart(result2['field_chart'], use_container_width=True)

                        # Keywords side by side
                        st.markdown("### Top Keywords")
                        col1, col2 = st.columns(2)
                        with col1:
                            if result1['keyword_chart']:
                                result1['keyword_chart'].update_layout(title=f"{author1['name']} - Keywords")
                                st.plotly_chart(result1['keyword_chart'], use_container_width=True)
                        with col2:
                            if result2['keyword_chart']:
                                result2['keyword_chart'].update_layout(title=f"{author2['name']} - Keywords")
                                st.plotly_chart(result2['keyword_chart'], use_container_width=True)

    # ==================== TAB 3: ABOUT ====================
    with tab3:
        st.markdown("""
        ## About the Interdisciplinary Index

        This tool measures how **interdisciplinary** a researcher's work is by analyzing the semantic
        similarity between their papers and the papers that cite them.

        ### How It Works

        1. **Fetch Top Papers**: Retrieves the author's top 10 most-cited papers from OpenAlex
        2. **Analyze Citations**: For each paper, fetches 10 recent citing papers
        3. **Compute Similarity**: Uses Model2Vec embeddings to calculate semantic similarity between original and citing abstracts
        4. **Calculate Index**: Index = (1 - average similarity) × 100

        ### Interpretation

        | Index Range | Category | Meaning |
        |-------------|----------|---------|
        | 0-20% | 🟢 Low | Focused research within specific domain |
        | 21-50% | 🟡 Moderate | Moderate cross-disciplinary engagement |
        | 51-80% | 🟠 High | Significant interdisciplinary impact |
        | 81-100% | 🔴 Very High | Highly interdisciplinary work |

        ### Technology Stack

        - **Data Source**: [OpenAlex](https://openalex.org/) - Open academic data
        - **Embeddings**: [Model2Vec](https://github.com/MinishLab/model2vec) (minishlab/potion-base-32M)
        - **Keywords**: [KeyBERT](https://github.com/MaartenGr/KeyBERT)
        - **Frontend**: [Streamlit](https://streamlit.io/)

        ### Limitations

        - Only analyzes papers with abstracts available in OpenAlex
        - Semantic similarity is an approximation of disciplinary distance
        - Results depend on citation patterns which may have biases

        ---

        **Source Code**: [GitHub](https://github.com/alkat19/interdisciplinary-index-analyzer)
        """)

if __name__ == "__main__":
    main()
