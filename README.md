# Mini Smart Home Energy Saver

**Course:** AI Fundamentals (Final Project)  
**Author:** Tazhkeyev Alim  

## Overview
An interactive AI prototype that reduces household electricity waste by combining three core AI paradigms in a single decision pipeline:
1. **Hill Climbing (Local Search):** Finds the lowest-cost appliance operating schedule among 5 candidate time slots.
2. **Bayesian Inference (Probabilistic Reasoning):** Updates room occupancy probability from noisy motion sensor readings.
3. **Markov Decision Process / MDP (Sequential Decision Making):** Evaluates expected utility to decide between `Keep On` and `Turn Off`.

## Repository Structure
- `algorithms.py` — Core implementations of Hill Climbing, Bayesian Inference, and MDP.
- `app.py` — Interactive Streamlit web dashboard.
- `requirements.txt` — Python dependencies (`streamlit`, `pandas`).
- `docs/` — Technical Report, 7-slide presentation, AI Usage Declaration, and Team Contribution Statement.

## How to Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py