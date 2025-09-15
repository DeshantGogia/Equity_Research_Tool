# Equity Research Tool

A local GenAI-powered equity research helper. Includes FAISS retrieval and FinBERT sentiment.

## Structure
- app.py: main app entry
- faiss_store_finbert/: FAISS index and metadata
- workflow_divided_example/: notebooks and sample datasets

## Getting started
`ash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
`

## Notes
- Consider using Git LFS for large artifacts (FAISS, .pkl).
- Notebooks checkpoints and venv are ignored via .gitignore.
