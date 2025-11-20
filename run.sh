#!/bin/bash
cd "$(dirname "$0")"
python3 -m streamlit run app/main_v2.py --server.port=8501 --server.address=0.0.0.0

