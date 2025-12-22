# Interdisciplinary Index Analyzer

A web application that measures the cross-domain impact of academic research by analyzing citation patterns and semantic similarity.

## What It Does

This tool calculates a **Composite Interdisciplinary Score** for researchers using four metrics:

### 1. Citation Diversity (30%)

**What it measures:** How different are the papers that cite your work from your own research?

**How it works:**
- Fetches your top 10 most-cited papers and their citing papers
- Converts abstracts into numerical vectors using ML embeddings (Model2Vec)
- Calculates cosine similarity between your papers and their citations
- Lower similarity = higher diversity score

**Example:** If a biology paper gets cited by economics and physics papers, those citations will have low semantic similarity to the original, resulting in a high diversity score.

### 2. Cluster Dispersion (25%)

**What it measures:** How spread out are your research topics?

**How it works:**
- Takes all paper embeddings and reduces them to 2D using PCA
- Groups papers into clusters using K-means algorithm
- Measures how far apart these clusters are from each other
- More spread = higher dispersion score

**Example:** A researcher working on both "machine learning" and "climate policy" would have distant clusters, indicating diverse research areas.

### 3. Reference Diversity (25%)

**What it measures:** How many different fields do you draw knowledge from?

**How it works:**
- Analyzes the academic fields of papers you reference
- Calculates Shannon entropy across field distribution
- More fields with balanced representation = higher score

**Example:** If your references span Medicine (40%), Mathematics (30%), and Computer Science (30%), you'd score higher than someone citing only Medicine (100%).

### 4. Bridge Score (20%)

**What it measures:** Are you connecting fields that don't usually talk to each other?

**How it works:**
- Compares fields you cite vs. fields that cite you
- Identifies "bridged" fields: those citing your work that you don't cite back
- More bridged fields = higher bridge score

**Example:** If you publish in Biology but get cited by Sociology researchers (whom you don't cite), you're bridging knowledge between disconnected fields.

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
- **Top Papers Table**: Ranked by citation diversity score with scrollable view
- **Visualizations** (8 interactive charts):
  - Metrics Overview (bullet chart)
  - Research Landscape (PCA scatter with clusters)
  - Citation Diversity by Year
  - Similarity Distribution (KDE)
  - Fields Referenced
  - Knowledge Flow (bridge analysis)
  - Citing Fields Breakdown
  - Top Keywords from citing papers
- **Interactive HTML Export**: Download full report with all charts preserved
- **Caching**: Session-based caching for faster repeated queries

## Tech Stack

- **Frontend**: Gradio
- **Data Source**: OpenAlex API
- **ML Model**: Model2Vec (minishlab/potion-base-32M)
- **Clustering**: scikit-learn (PCA, K-means)
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
python app_inter_pro.py
```

The app will launch at `http://localhost:7860`

## Acknowledgments

- [OpenAlex](https://openalex.org/) for the open academic data API
- [Model2Vec](https://github.com/MinishLab/model2vec) for efficient embeddings
- [KeyBERT](https://github.com/MaartenGr/KeyBERT) for keyword extraction
