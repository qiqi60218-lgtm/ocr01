@echo off

pip install -r requirements.txt
streamlit run docvision_ai.py --server.address 0.0.0.0 --server.port 8502

pause