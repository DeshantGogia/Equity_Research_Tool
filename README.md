# Equity Research Tool

A local GenAI-powered equity research helper. Includes FAISS retrieval and FinBERT sentiment.

A QnA tool where you upload the number of web URL's to the streamlit UI --> the llama 3:8b model retrieves the vector indexes created using FINBERT of the text in the URL through RAG  -->  ask a question related to the uploaded URL --> answer's the question based on the URL's --> setting model parameters like temperature to generate answers.
<img width="1857" height="758" alt="Screenshot 2025-09-15 190726" src="https://github.com/user-attachments/assets/163458fd-ec66-4d99-88fd-f0a8cb633701" />

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
