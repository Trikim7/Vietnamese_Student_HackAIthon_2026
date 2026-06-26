#!/bin/bash
# inference.sh - Official execution script for VSDS 2026 Track C Innovator

if [ -f "/code/src/predict.py" ]; then
    python3 /code/src/predict.py
elif [ -f "/agent_src/src/predict.py" ]; then
    python3 /agent_src/src/predict.py
elif [ -f "src/predict.py" ]; then
    python3 src/predict.py
else
    python3 predict.py
fi
