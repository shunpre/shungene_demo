#!/bin/bash
cd /home/ubuntu/swipe_lp_analyzer
python3 -m streamlit run app/main_v2.py --server.port=8501 --server.address=0.0.0.0

