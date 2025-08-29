# AI-Powered Movie Recommender

A hybrid recommender blending collaborative filtering (stats-based correlations) and deep learning embeddings for semantic matches using MovieLens data. Demonstrates explainable AI with p-value insights (p as probability of data assuming null hypothesis of no correlation—low p<0.05 rejects chance for significant similarity).

## Features
- Collaborative: Pearson r on user ratings (filtered n>50 for low variance).
- Content: Genre embeddings via Sentence-Transformers (cosine sim >0.6 rejects null of unrelated content).
- Fuzzy Queries: Partial inputs (e.g., "toy story" → "Toy Story (1995)").
- Streamlit UI: Interactive with stats sidebar.

## Prerequisites
- macOS (tested on MacBook Pro).
- Homebrew installed (for Python 3.12): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`.
- Git cloned repo: `git clone https://github.com/ktech7moon/movie_recommender.git; cd movie_recommender`.

## Setup Steps
1. **Install Python 3.12** (Stable for ML deps):  
   `brew install python@3.12` (Verify: `python3.12 --version` ~3.12.11).

2. **Create/Activate Virtual Env**:  
   `python3.12 -m venv recommender_env_312`  
   `source recommender_env_312/bin/activate` (Deactivate: `deactivate`). Prompt changes to ((recommender_env_312)).

3. **Install Dependencies**:  
   `pip install --upgrade pip`  
   `pip install pandas matplotlib seaborn scikit-learn streamlit torch==2.2.2 transformers==4.41.2 sentence-transformers==3.0.1`  
   (If NumPy warning: `pip install numpy==1.26.4` for compatibility).

4. **Download Dataset**:  
   `curl -O https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`  
   `unzip ml-latest-small.zip`.

## Running the Project
1. **Core Script** (CLI test/recs):  
   `python recommender.py --movie "Toy Story (1995)"` (Outputs exploration, matrix shape, recs).

2. **Web App** (Interactive):  
   `streamlit run app.py` (Open http://localhost:8501; test "toy story"—fuzzy matches, shows hybrid recs).

## Deployment
- Streamlit Cloud: Sign up at streamlit.io/cloud, connect GitHub repo, select branch/app.py for live URL.

## Stats/AI Notes
- P-value: Probability of observing data (or extreme) assuming null (no similarity); low p<0.05 = evidence against chance.
- Embeddings: Cosine sim measures vector closeness (high sim rejects null of random genres).

## Troubleshooting
- Env switch: Always activate recommender_env_312 before run.
- Errors: Check pip list for deps; reinstall if missing.

License: MIT