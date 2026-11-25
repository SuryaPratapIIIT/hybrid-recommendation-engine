---
title: Hybrid Recommendation Engine
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Hybrid Recommendation System

An end-to-end machine learning recommendation platform combining **Content-Based Filtering (TF-IDF)**, **Bayesian Personalized Ranking (BPR) Collaborative Filtering**, **Popularity Ranking**, and **Cold-Start Handling**, served via **FastAPI** and an interactive **Streamlit** dashboard.

---

## Overview

This system routes each request to the most appropriate recommendation strategy based on user and item history:

- **Warm Users** (present in both Content-Based and BPR models) receive **Hybrid** recommendations combining normalized scores:
  - `40%` Content-Based Filtering (TF-IDF item features & user profiles)
  - `40%` BPR Collaborative Filtering (PyTorch latent embeddings)
  - `20%` Popularity Baseline (interaction-strength weighted)
- **Cold Users** (unseen or lacking sufficient history) automatically fall back to **Popularity** (`cold_user_popularity`).
- **Cold Items** (items absent from training interactions) are ranked by **Category Overlap** against the user's learned category preferences (`cold_item_category`).
- **Seen-Item Filtering**: Items the user has already interacted with in `train.parquet` (plus any explicitly excluded items) are automatically filtered out.

---

## Architecture

```text
User ──▶ Streamlit UI (Port 8501)
             │
             ▼
         FastAPI Service (Port 8000)
             │
             ▼
       Unified Recommender (src/inference/recommender.py)
             │
             ├──▶ Warm User ──▶ Hybrid Ranker (0.4 Content + 0.4 BPR + 0.2 Popularity)
             ├──▶ Cold User ──▶ Popularity Fallback
             └──▶ Cold Items ─▶ Category-Overlap Ranker
```

---

## Offline Evaluation (`@10`)

Evaluated on the held-out test set (`models/evaluation/final_test_results.csv`):

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 |
| :--- | :---: | :---: | :---: | :---: |
| **Hybrid (40/40/20)** | **0.007518** | 0.016739 | **0.015367** | **0.055319** |
| Content-Based (TF-IDF) | 0.005532 | **0.016859** | 0.011305 | 0.043972 |
| BPR Collaborative Filtering | 0.002411 | 0.003171 | 0.004455 | 0.018440 |
| Popularity Baseline | 0.001560 | 0.000167 | 0.002273 | 0.012766 |

---

## Project Structure

```text
Recommendation-System/
├── data/
│   └── processed/                  # Inference data (train.parquet, item_categories.parquet, bpr_eval_users.npy)
├── models/
│   ├── bpr_model.pt                # Trained BPR embeddings and user/item index mappings
│   ├── product_tfidf.npz           # Sparse TF-IDF item feature matrix
│   ├── user_profiles.npz           # Sparse user preference profiles
│   ├── content_vectorizer.joblib   # Fitted TF-IDF vectorizer
│   ├── product_ids.npy             # Item ID index for content matrix
│   ├── warm_user_ids.npy           # User ID index for content profiles
│   ├── popularity_scores.csv       # Precomputed item popularity scores
│   ├── cold_start/                 # Cold-item IDs and category matrices
│   └── evaluation/                 # Offline evaluation CSVs and hybrid config
├── notebooks/                      # End-to-end EDA, training, and evaluation notebooks (01–07)
├── reports/
│   └── model_results.csv           # Baseline popularity evaluation across K = 5, 10, 20
├── src/
│   ├── api/
│   │   └── app.py                  # FastAPI application and endpoints
│   ├── evaluation/
│   │   └── metrics.py              # Precision@K, Recall@K, NDCG@K, HitRate@K
│   ├── inference/
│   │   └── recommender.py          # Unified Recommender routing & seen-item filtering
│   ├── models/
│   │   ├── bpr.py                  # BPRRecommender
│   │   ├── cold_item.py            # ColdItemCategoryRecommender
│   │   ├── content_based.py        # ContentBasedRecommender
│   │   ├── hybrid.py               # HybridRecommender
│   │   ├── popularity.py           # PopularityRecommender
│   │   └── save_popularity.py      # Script to regenerate popularity_scores.csv
│   └── ui/
│       └── app.py                  # Streamlit interactive frontend
├── tests/
│   └── test_api.py                 # Automated FastAPI TestClient suite
├── .gitignore
├── .python-version
└── requirements.txt
```

---

## Getting Started

### 1. Clone the Repository

```powershell
git clone https://github.com/SuryaPratapIIIT/hybrid-recommendation-engine.git
cd hybrid-recommendation-engine
```

### 2. Set Up Virtual Environment & Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Start the FastAPI Backend

```powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

### 4. Start the Streamlit UI

In a second terminal (with the virtual environment activated):

```powershell
python -m streamlit run src/ui/app.py
```

Open `http://localhost:8501` in your browser. You can override the backend URL with the `API_URL` environment variable if needed.

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health check |
| `GET` | `/recommend/{user_id}?k=10` | Top-K recommendations (`hybrid` for warm users, `cold_user_popularity` for cold users) |
| `GET` | `/recommend/{user_id}/exclude?k=10&exclude=1,2` | Top-K recommendations excluding comma-separated item IDs |
| `GET` | `/recommend/{user_id}/cold-items?k=10` | Top-K cold-item recommendations via category overlap |
| `GET` | `/users/{user_id}/strategy` | Inspect routing strategy and warm-user status for a user |

**Example Request (PowerShell):**

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/recommend/172?k=10"
```

---

## Running Tests

Run the automated API test suite with `pytest`:

```powershell
python -m pytest tests -v
```

---

## Tech Stack

- **ML & Data**: Python, PyTorch, NumPy, Pandas, SciPy, PyArrow
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit, Requests
- **Testing**: Pytest, HTTPX
