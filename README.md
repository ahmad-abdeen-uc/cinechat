# CineChat

CineChat is a local, conversational movie recommender. Ask for a genre, a mood,
a similar title, or a specific filter. The Streamlit chat shows verified movie
details, TMDB audience ratings when available, and short explanations.

## Run locally on Windows

Use Python 3.12 and the existing `venv` folder from the project root.

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. For a fresh environment, create the
virtual environment with `py -3.12 -m venv venv` and install the pinned
dependencies with `.\venv\Scripts\python.exe -m pip install -r requirements.txt`.

Start Ollama locally and use `ollama list` to confirm `qwen2.5:3b` is installed.
For a new Ollama setup, make the model available with `ollama pull qwen2.5:3b`.
CineChat uses the local Ollama server at `http://localhost:11434`.

The app expects the existing artifacts at `data/processed/cinechat.db` and
`data/vector_store/movies.faiss`. Semantic text queries also need the existing
`sentence-transformers/all-MiniLM-L6-v2` model cached locally. If the cache is
empty on a new machine, download that same model once into the default cache.

```powershell
.\venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"
```

## Run locally on macOS

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ollama            # or download the app from https://ollama.com/download
ollama serve &                 # skip if the Ollama app is already running
ollama pull qwen2.5:3b
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"
streamlit run app.py
```

## Measure accuracy

`evaluation/gold_set.json` holds 30 realistic requests with the filters and topics each should
produce. The script reports intent field accuracy, constraint satisfaction, relevance@5 and
answer rate.

```bash
python evaluation/evaluate.py                 # full pipeline (needs Ollama)
python evaluation/evaluate.py --rules-only    # rule layer only
```

See `CHANGES.md` for the accuracy improvements and before/after results.

## Using the chat

- Select an example or type your own request. Each uses the same local recommendation pipeline.
- Use **New Conversation** to clear the visible chat while keeping the local search resources warm.
- Restate the filters you want to carry forward. Chat history is displayed for the current browser session. Retrieval interprets each request independently.

## Catalog and limitations

The local Kaggle Movies Dataset catalog contains 45,433 movies. FAISS contains
45,198 movie embeddings, so some catalog entries cannot appear in semantic
results. Ratings are TMDB aggregate audience ratings. This historical dataset
does not provide live streaming availability or verified critic scores.

SQLite and FAISS are used read-only. Search warnings, unavailable local services,
and responses that need clarification appear in the chat.

## UI tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_app.py -v
```
