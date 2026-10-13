# AskMyPDF - Chat with your PDFs (RAG)

Ask questions about your PDFs and get grounded answers with page-level citations.
Built with LangChain, FAISS, HuggingFace embeddings, Groq LLM and Streamlit.

**Live demo:** <YOUR_STREAMLIT_LINK>

## Why I built this
I wanted a simple way to ask questions about a PDF without reading all of it.
While building it, I learned how a chatbot finds the right part of a document and then answers from it.
I also hit a real error when the AI model I was using was removed, so I made the model name a setting that can be changed easily.

## Features
- Multi-PDF upload with page-level source citations
- Follow-up questions via history-aware query rewriting
- Strict grounded prompt: answers "not found" when the context has no answer
- MMR retrieval for diverse, non-redundant chunks
- Evaluation script and unit tests

## Architecture
```
PDF -> pages -> chunks (800 chars, 120 overlap) -> MiniLM embeddings -> FAISS index

Question -> (rewrite using chat history) -> MMR top-4 chunks -> Groq LLM -> answer + sources
```
## Results (own test set, 12 questions)
| chunk_size | top_k | Retrieval hit rate | Answer match |
|-----------:|------:|-------------------:|-------------:|
| 800        | 4     | 11/12 (92%)        | 10/12 (83%)  |
## Design decisions
- **FAISS:** fast, local, no external service needed for a small corpus.
- **MMR:** plain top-k often returns near-duplicate chunks; MMR balances relevance and diversity.
- **temperature=0 + strict prompt:** reduces made-up answers.
- **Limitations:** scanned PDFs (no OCR), index is in memory, small LLM can miss details.

## Project structure
```
app.py            Streamlit UI
src/config.py     Settings
src/ingest.py     PDF loading, chunking, indexing
src/chain.py      Retrieval and answer generation
eval/             Evaluation script and test questions
tests/            Unit tests
```

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env      # then add your GROQ_API_KEY
streamlit run app.py
```

## Tests and evaluation
```bash
python -m pytest
python -m eval.run_eval --pdf data/sample.pdf
```
