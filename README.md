# Equity Research Tool

A local GenAI-powered equity research helper. Includes FAISS retrieval and FinBERT sentiment.

A QnA tool created using Langchain where user upload's the number of web URL's to the streamlit UI --> the llama 3:8b model retrieves the vector indexes created using FINBERT of the text in the URL through RAG --> user asks a question related to the uploaded URL --> answer's the question based on the URL's --> setting model parameters like temperature to generate answers.

As the name suggests "Eqyuity Research Tool", It works best with finance-focused URLs, delivering more accurate and domain-specific answers. as the word embedding model used in this tool is "FINBERT" also know as "Financial BERT" model which is a specialized version of the BERT model fine-tuned for financial text and terminology.

<img width="1857" height="758" alt="Screenshot 2025-09-15 190726" src="https://github.com/user-attachments/assets/163458fd-ec66-4d99-88fd-f0a8cb633701" />



Yes, ChatGPT and many other commercial AI models can perform similarly. However, the key advantage of this approach is that it extracts only the most relevant chunks of data from the uploaded web URLs and provides them to the LLM based on the user’s query through FAISS. This not only improves contextual accuracy but also helps optimize token usage, which is especially important when working with commercial models like ChatGPT that have token limitations.

<img width="759" height="239" alt="Screenshot 2025-09-16 130653" src="https://github.com/user-attachments/assets/a17fa723-00f3-4f95-adf5-4aacba53ccf4" />

<img width="757" height="228" alt="Screenshot 2025-09-16 130727" src="https://github.com/user-attachments/assets/aa8fc881-9b2a-48c2-969b-1d692d2c0a34" />

<img width="796" height="70" alt="Screenshot 2025-09-06 051221" src="https://github.com/user-attachments/assets/4c60f705-b003-4c21-9411-10004a4e79eb" />

<img width="960" height="170" alt="Screenshot 2025-09-16 130756" src="https://github.com/user-attachments/assets/484b8398-8d38-4d43-b149-387505ca42d1" />

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
