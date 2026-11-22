#!/bin/bash
# One-step macOS setup and launch for CineChat.
# Installs what is missing (Python packages, Ollama, Qwen2.5 3B), then starts the app at http://localhost:8501.
# Progress goes to setup.log in this folder. Safe to run again: finished steps are skipped.
set -u
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:$HOME/.local/bin:$PATH"
step() { echo; echo "=== $(date '+%H:%M:%S') $*"; }

step "Finding Python 3.10-3.13"
PY=""
for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
      PY="$(command -v "$candidate")"; break
    fi
  fi
done
if [ -z "$PY" ]; then echo "ERROR: no Python 3.10-3.13 found. Install Python 3.12 from python.org."; exit 1; fi
echo "Using $PY ($("$PY" --version))"

step "Creating virtual environment"
[ -x venv/bin/python ] || "$PY" -m venv venv
VPY="$PWD/venv/bin/python"
"$VPY" -m pip install --quiet --upgrade pip

step "Installing Python packages"
if ! "$VPY" -c "import streamlit, faiss, sentence_transformers, ollama, pydantic, rapidfuzz" 2>/dev/null; then
  if ! "$VPY" -m pip install --quiet -r requirements.txt; then
    echo "Pinned requirements failed on this Mac; installing current versions of the core packages instead."
    "$VPY" -m pip install --quiet streamlit faiss-cpu sentence-transformers ollama pydantic rapidfuzz numpy pandas
  fi
fi
"$VPY" -c "import streamlit, faiss, sentence_transformers, ollama, pydantic, rapidfuzz; print('Python packages OK')" || exit 1

step "Finding or installing Ollama"
OLLAMA="$(command -v ollama || true)"
for app in /Applications/Ollama.app "$HOME/Applications/Ollama.app"; do
  [ -z "$OLLAMA" ] && [ -x "$app/Contents/Resources/ollama" ] && OLLAMA="$app/Contents/Resources/ollama"
done
if [ -z "$OLLAMA" ]; then
  mkdir -p "$HOME/Applications"
  curl -fsSL https://ollama.com/download/Ollama-darwin.zip -o /tmp/Ollama-darwin.zip
  ditto -x -k /tmp/Ollama-darwin.zip "$HOME/Applications/"
  OLLAMA="$HOME/Applications/Ollama.app/Contents/Resources/ollama"
fi
echo "Using $OLLAMA"
if ! curl -fs http://127.0.0.1:11434/api/version >/dev/null; then
  nohup "$OLLAMA" serve > ollama.log 2>&1 < /dev/null &
  disown
  for i in $(seq 1 30); do curl -fs http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
fi
curl -fs http://127.0.0.1:11434/api/version && echo " Ollama is running"

step "Downloading Qwen2.5 3B (about 2 GB, first time only)"
"$OLLAMA" list | grep -q "qwen2.5:3b" || "$OLLAMA" pull qwen2.5:3b
"$OLLAMA" list | grep "qwen2.5:3b" || { echo "ERROR: model download failed"; exit 1; }

step "Downloading the MiniLM embedding model (first time only)"
"$VPY" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'); print('MiniLM OK')" || exit 1

# PyTorch and FAISS each bundle OpenMP on macOS; loading both can crash Python silently.
# These settings are the standard workaround.
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1

step "Smoke test: one real search (LLM + FAISS)"
"$VPY" - <<'PY'
from src.movie_search import MovieSearch
search = MovieSearch()
result = search.search("Recommend sci-fi movies about space exploration")
print("status:", result.status, "| strategy:", result.strategy)
print("movies:", [f"{m.title} ({m.release_year})" for m in result.movies])
search.close()
PY
code=$?
echo "Smoke test exit code: $code"
if [ $code -ne 0 ]; then echo "ERROR: the search crashed (exit code $code). Send this log to Claude."; exit 1; fi

step "Starting CineChat"
if curl -fs http://localhost:8501/_stcore/health >/dev/null; then
  echo "CineChat is already running"
else
  # Detached from this Terminal window, so closing the window does not stop the app.
  # If Streamlit ever stops, its exit code is written at the end of streamlit.log.
  nohup bash -c '"$0" -m streamlit run app.py --server.headless true --server.port 8501; echo "Streamlit stopped with exit code $?"' \
    "$VPY" > streamlit.log 2>&1 < /dev/null &
  disown
  for i in $(seq 1 60); do curl -fs http://localhost:8501/_stcore/health >/dev/null && break; sleep 1; done
fi
curl -fs http://localhost:8501/_stcore/health && echo " CineChat is ready at http://localhost:8501"
echo "SETUP FINISHED"
