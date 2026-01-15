# Interdisciplinary Index Analyzer

A web application that measures the cross-domain impact of academic research by analyzing citation patterns and semantic similarity.

## What It Does

This tool calculates a **Composite Interdisciplinary Score** for researchers using four equally-weighted metrics (25% each):

### 1. External Diversity (25%)

**What it measures:** How different are the papers that cite your work from your own research?

**How it works:**
- Fetches your top 10 most-cited papers
- For each paper, retrieves the 50 most recent citing papers
- Converts abstracts into numerical vectors using ML embeddings
- Calculates cosine distance between your papers and their citations
- Higher distance = higher diversity score

**Example:** If a biology paper gets cited by economics and physics papers, those citations will have high semantic distance from the original, resulting in a high diversity score.

### 2. Internal Diversity (25%)

**What it measures:** How spread out are your research topics?

**How it works:**
- Takes all 10 paper embeddings and computes pairwise cosine distance
- Score = average pairwise distance × 100
- Higher distance = papers cover more different topics

**Example:** A researcher working on both "machine learning" and "climate policy" would have high distance between papers, indicating diverse research areas.

### 3. Reference Diversity (25%)

**What it measures:** How many different fields do you draw knowledge from?

**How it works:**
- For each of your 10 papers, analyzes up to 50 references
- Extracts the academic field of each referenced paper
- Calculates Shannon entropy across field distribution
- More fields with balanced representation = higher score

**Example:** If your references span Medicine (40%), Mathematics (30%), and Computer Science (30%), you'd score higher than someone citing only Medicine (100%).

### 4. Bridge Score (25%)

**What it measures:** Are you connecting fields that don't usually talk to each other?

**How it works:**
- Compares fields you cite (50 refs per paper) vs. fields that cite you (50 most recent per paper)
- Identifies "bridged" fields: those citing your work that you don't cite back
- Bridge score = bridged fields / total unique fields × 100

**Example:** If you publish in Biology but get cited by Sociology researchers (whom you don't cite), you're bridging knowledge between disconnected fields.

## Analysis Scope

| Parameter | Value |
|-----------|-------|
| Papers analyzed | Top 10 most-cited |
| Citing papers per paper | 50 (most recent) |
| References per paper | 50 |

## Interpretation

| Score Range | Category | Meaning |
|-------------|----------|---------|
| 0-20% | Low | Focused research within specific domain |
| 21-50% | Moderate | Moderate cross-disciplinary engagement |
| 51-80% | High | Significant interdisciplinary impact |
| 81-100% | Very High | Highly interdisciplinary work |

## Features

- **Author Search**: Search for any researcher in OpenAlex database
- **Disambiguation**: Select from multiple matching authors
- **Top Papers Table**: Ranked by External Diversity score with scrollable view
- **Visualizations** (8 interactive charts):
  - Metrics Overview (bullet chart)
  - Paper Similarity Heatmap (pairwise cosine similarity matrix)
  - External Diversity by Year (scatter plot)
  - Similarity Distribution (KDE)
  - Fields Referenced (bar chart)
  - Knowledge Flow (bridge analysis)
  - Citing Fields Breakdown (bar chart)
  - Top Keywords from citing papers
- **Interactive HTML Export**: Download full report with all charts preserved
- **Caching**: Session-based caching for faster repeated queries

## Performance Optimizations

- **Batched API Calls**: Uses OpenAlex's pipe-separated ID filter to fetch 50 references in 1 call instead of 50 individual calls (96% reduction in API requests)
- **Parallel Fetching**: Reference Diversity and Bridge Score fetch all papers simultaneously via `asyncio.gather` (5-10x faster)
- **Rate Limit Detection**: Gracefully handles OpenAlex's 100,000 daily request limit with user-friendly error message
- **Progress Logging**: Step-by-step console output shows analysis progress

## Tech Stack

- **Frontend**: Gradio
- **Data Source**: OpenAlex API
- **ML Model**: Sentence Transformers (minishlab/potion-base-32M)
- **Similarity**: scipy (cosine distance), scikit-learn (cosine similarity for heatmap)
- **Visualization**: Plotly
- **Keyword Extraction**: KeyBERT

## Local Development

```bash
# Clone the repo
git clone https://github.com/alkat19/interdisciplinary-index-analyzer.git
cd interdisciplinary-index-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
python interdisciplinary_app.py
```

The app will launch at `http://localhost:7860`

## Acknowledgments

- [OpenAlex](https://openalex.org/) for the open academic data API
- [Sentence Transformers](https://www.sbert.net/) for efficient embeddings
- [KeyBERT](https://github.com/MaartenGr/KeyBERT) for keyword extraction
