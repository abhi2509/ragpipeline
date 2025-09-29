# RAG Pipeline for Financial Data

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, an in-memory vector store, and OpenAI's free API for LLM integration. The UI is built with Streamlit.

## Features
- Upload and embed financial data (CSV)
- Query financial data using LLM
- In-memory vector store for fast retrieval
- Simple Streamlit UI

## Folder Structure
```
ragpipeline/
├── src/            # Core pipeline code
├── data/           # Sample financial data files
├── ui/             # Streamlit UI code
├── requirements.txt
└── README.md
```

## Installation
1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd ragpipeline
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Place your financial data CSV files in the `data/` folder.
2. Run the Streamlit UI:
   ```sh
   streamlit run ui/app.py
   ```
3. Enter your OpenAI API key in the UI when prompted.

## How it Works
- Financial data is embedded using Sentence Transformers.
- Embeddings are stored in-memory for fast retrieval.
- LangChain handles retrieval and LLM integration.
- Streamlit provides a simple interface for uploading data and asking questions.

## Notes
- Only free OpenAI endpoints are used (e.g., GPT-3.5-turbo).
- No external vector database required.

## License
MIT
