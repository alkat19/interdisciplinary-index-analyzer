# Interdisciplinary Index Analyzer

A web application that measures the cross-domain impact of academic research by analyzing citation patterns and semantic similarity.

## What It Does

This tool calculates an **Interdisciplinary Index** for researchers by:

1. Fetching the author's top 10 most-cited papers from OpenAlex
2. For each paper, retrieving 10 recent citing papers
3. Computing semantic similarity between original and citing abstracts using Model2Vec embeddings
4. Index = (1 - average similarity) × 100
5. Higher index indicates citations from more diverse/different fields

## Interpretation

| Index Range | Category | Meaning |
|-------------|----------|---------|
| 0-20% | Low | Focused research within specific domain |
| 21-50% | Moderate | Moderate cross-disciplinary engagement |
| 51-80% | High | Significant interdisciplinary impact |
| 81-100% | Very High | Highly interdisciplinary work |

## Features

- **Author Search**: Search for any researcher in OpenAlex database
- **Disambiguation**: Select from multiple matching authors
- **Visualizations**:
  - Index by publication year (scatter plot)
  - Similarity distribution (KDE plot)
  - Citation fields breakdown (pie chart)
  - Top keywords from citing papers (bar chart)
- **Export**: Download results as CSV
- **Caching**: Session-based caching for faster repeated queries

## Tech Stack

- **Frontend**: Streamlit
- **Data Source**: OpenAlex API
- **ML Model**: Model2Vec (minishlab/potion-base-32M)
- **Keyword Extraction**: KeyBERT
- **Visualization**: Plotly

## Local Development

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/interdisciplinary-index-analyzer.git
cd interdisciplinary-index-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## Deployment

This app is designed for deployment on [Streamlit Cloud](https://streamlit.io/cloud):

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo and `streamlit_app.py` as the main file
5. Deploy!

## License

MIT License

## Acknowledgments

- [OpenAlex](https://openalex.org/) for the open academic data API
- [Model2Vec](https://github.com/MinishLab/model2vec) for efficient embeddings
- [KeyBERT](https://github.com/MaartenGr/KeyBERT) for keyword extraction
