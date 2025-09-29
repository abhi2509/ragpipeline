
# Financial Data RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline for financial data using LangChain, FAISS (in-memory vector store), and OpenAI LLMs. The UI is built with Streamlit and supports advanced features for usability and performance.

## Features
- Upload and embed financial data (CSV, Excel, JSON, PDF)
- Query financial data using OpenAI LLM (GPT-3.5-turbo)
- In-memory FAISS vector store for fast retrieval
- Modular codebase (embedding, retrieval, LLM, data utils)
- Advanced prompt engineering
- Data validation and cleaning
- Caching for repeated queries
- Query performance metrics
- Context display for transparency
- Interactive charts for numeric columns
- Customizable dark/light mode UI
- Full logging and error handling
- Test coverage with pytest

## Folder Structure
```
ragpipeline/
├── src/
│   ├── pipeline.py         # Main RAG pipeline
│   ├── config.py           # Config management
│   ├── prompt.py           # Prompt engineering
│   └── modules/
│       ├── embedding.py    # Embedding logic
│       ├── retrieval.py    # Retrieval logic
│       ├── data_utils.py   # Data cleaning/validation
│       └── llm_utils.py    # LLM management
├── ui/
│   └── app.py              # Streamlit UI
├── data/                   # Financial data files
├── tests/                  # Test cases
├── requirements.txt        # Core dependencies
├── requirements-dev.txt    # Dev/test/lint dependencies
├── .env                    # Environment/config variables
└── README.md
```

## Installation
1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd ragpipeline
   ```
2. Create a virtual environment (recommended):
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Usage
1. Place your financial data files (CSV, Excel, JSON, PDF) in the `data/` folder or upload via UI.
2. Set your OpenAI API key in the `.env` file or as an environment variable:
   ```sh
   set OPENAI_API_KEY=your-key-here
   ```
3. Run the Streamlit UI:
   ```sh
   streamlit run ui/app.py
   ```

## How it Works
- Data is parsed, validated, and cleaned (supports CSV, Excel, JSON, PDF)
- Text is chunked and embedded using Sentence Transformers
- Embeddings are stored in FAISS for fast similarity search
- LangChain handles retrieval and LLM integration
- Streamlit UI provides upload, preview, charts, query, and context display

## Advanced Features
- Modular code for maintainability and scalability
- Prompt engineering for financial queries
- Caching and query performance metrics
- Customizable UI (dark/light mode, charts, sidebar)
- Full logging and error handling
- Test coverage with pytest and pytest-cov

## Notes
- Only free OpenAI endpoints are used (e.g., GPT-3.5-turbo)
- No external vector database required
- All config options are managed via `.env` and `config.py`

## Testing
Run all tests with coverage:
```sh
pytest --cov=src tests/
```

## License
MIT
