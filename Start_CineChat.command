#!/bin/bash
# Double-click in Finder to set up (first time) and start CineChat at http://localhost:8501
cd "$(dirname "$0")"
bash ./run_cinechat.sh 2>&1 | tee setup.log
echo
echo "You can close this window; CineChat keeps running in the background."
