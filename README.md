# Interdisciplinary Index Analyzer

A web application that measures the cross-domain impact of academic research by analyzing citation patterns and semantic similarity.

## What It Does

This tool calculates a **Composite Interdisciplinary Score** for researchers using four metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| **Citation Diversity** | 30% | Semantic distance between author's papers and citing papers |
| **Cluster Dispersion** | 25% | How spread out research topics are (PCA + K-means) |
| **Reference Diversity** | 25% | Variety of fields in references (Shannon entropy) |
| **Bridge Score** | 20% | Fields that cite you but you don't cite back |

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
