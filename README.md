# CineMatch — End-to-End Movie Recommender System

Hybrid recommendation engine combining **Content-Based Filtering** (TF-IDF & Cosine Similarity) and **Deep Collaborative Filtering** via neural embeddings (**RecommenderNet** in TensorFlow/Keras) benchmarked on the industry-standard **MovieLens 100K** dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1INLUzym1GanvzJL8NU3YgXJeYVfl0K8N?usp=sharing)
[![Live Demo on Vercel](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=flat&logo=vercel)](https://movie-recommender-system-mu-sable.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset: MovieLens 100K](https://img.shields.io/badge/Dataset-MovieLens%20100K-4B8BBE.svg)](https://grouplens.org/datasets/movielens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Live Demos

- **Production Web Application**: [https://movie-recommender-system-mu-sable.vercel.app/](https://movie-recommender-system-mu-sable.vercel.app/)
- **Executable Research Notebook**: [Open in Google Colab](https://colab.research.google.com/drive/1INLUzym1GanvzJL8NU3YgXJeYVfl0K8N?usp=sharing)

## Architecture & Recommendation Engines

```
                        +----------------------------+
                        |  MovieLens 100K (100k pts) |
                        +-------------+--------------+
                                      |
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
       [ Content-Based Engine ]            [ Collaborative Filtering ]
     TF-IDF Genre & Synopsis Vectors      User & Item Latent Embeddings (50-dim)
            |                                   |
     Cosine Similarity Matrix              Dot Product + Dropout + Sigmoid
            |                                   |
     Precision@10: 100%                   RMSE: 0.9796 | MAE: 0.7494
```

### 1. Content-Based Filtering
- Represents movie metadata (genres, textual attributes) into high-dimensional TF-IDF vectors.
- Computes pairwise Cosine Similarity matrices to evaluate item proximity.
- Robust against item cold-start issues, yielding **100% Precision@10** on thematic genre similarity queries.

### 2. Deep Collaborative Filtering (RecommenderNet)
- Maps user IDs and movie IDs into dense 50-dimensional latent embedding spaces.
- Computes user-item interactions via dot-product matching with bias vectors.
- Applies Dropout regularization and custom learning rate schedules with Adam optimizer to predict user affinity scores.

## Empirical Benchmark & Performance Metrics

Evaluated on the 100,000-rating benchmark split:

| Paradigm | Architecture / Algorithm | Primary Metric | Quantitative Score | Characteristics |
| :--- | :--- | :---: | :---: | :--- |
| **Content-Based** | TF-IDF + Cosine Similarity | **Precision@10** | **100.00%** | Zero cold-start latency for new catalog items. |
| **Collaborative** | RecommenderNet (50-dim Keras) | **RMSE**<br>**MAE** | **0.9796**<br>**0.7494** | Uncovers latent cross-genre serendipity across user communities. |

## Interactive Web Showcase

The production web client provides instant browser-based recommendation querying:
- **Real-Time Content Querying**: Instant lookup across 9,737 movies with sub-millisecond client-side Cosine Similarity scoring.
- **Persona Simulations**: Interactive evaluation comparing user rating history against neural predictions.
- **Zero-Latency Static Hosting**: Deployed on Vercel Edge Network.

## Repository Structure

```
├── index.html                             # Single-page web showcase client
├── style.css                              # Production styling
├── app.js                                 # Client-side inference simulator
├── vercel.json                            # Edge caching & security configuration
├── data/                                  # Precomputed vector indices
│   ├── movies.json                        # 9,737 movie catalog & metadata
│   └── cf_users.json                      # Sample user rating profiles
├── proyek_akhir_sistem_rekomendasi.ipynb  # End-to-end research notebook
├── proyek_akhir_sistem_rekomendasi.py     # Standalone Python script
└── requirements.txt                       # Python dependencies
```

## Quickstart

### Running the Notebook

Open [`proyek_akhir_sistem_rekomendasi.ipynb`](proyek_akhir_sistem_rekomendasi.ipynb) locally or launch on [Google Colab](https://colab.research.google.com/drive/1INLUzym1GanvzJL8NU3YgXJeYVfl0K8N?usp=sharing).

### Running Locally

```bash
# Clone the repository
git clone https://github.com/zakski-bit/movie-recommender-system.git
cd movie-recommender-system

# Serve web client locally
python -m http.server 8000
```
Navigate to `http://localhost:8000` to interact with the recommendation engine.

## License

MIT License. See [LICENSE](LICENSE) for details.
