import os
import logging
import pandas as pd
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL, DATA_PATH
from prompt import build_financial_prompt
from typing import List, Optional, Union
import hashlib
import json
from io import BytesIO
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))  # Configurable chunk size
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))  # Configurable overlap

class FinancialRAGPipeline:
    def __init__(self, data_path=DATA_PATH, openai_api_key=OPENAI_API_KEY):
        if not data_path or not os.path.exists(data_path):
            raise ValueError(f"Invalid data path: {data_path}. File does not exist.")
        self.data_path = data_path
        self.openai_api_key = openai_api_key
        self.embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=LLM_MODEL)
        self.qa_chain = None
        self.cache = {}
        logging.info(f"Initialized pipeline with data: {data_path}")

    def load_and_embed(self):
        try:
            logging.info("Loading financial data for embedding...")
            df = self._parse_file(self.data_path)
            df = self._validate_and_clean_data(df)
            texts = df.astype(str).apply(lambda row: ' '.join(row), axis=1).tolist()
            chunks = self._chunk_texts(texts)
            logging.info(f"Embedding {len(chunks)} chunks of data.")
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
            logging.info("In-memory FAISS vector store created.")
        except Exception as e:
            logging.error(f"Error in load_and_embed: {e}")
            raise

    def similarity_search(self, query: str, k: int = 5) -> List[str]:
        if not self.vector_store:
            raise Exception("Vector store not initialized. Call load_and_embed() first.")
        logging.info(f"Performing similarity search for: {query}")
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def query(self, question: str) -> Optional[str]:
        if not self.vector_store:
            raise Exception("Vector store not initialized. Call load_and_embed() first.")
        cache_key = hashlib.sha256(question.encode()).hexdigest()
        if cache_key in self.cache:
            logging.info("Returning cached result.")
            return self.cache[cache_key]
        try:
            # Retrieve top-k similar chunks
            context_chunks = self.similarity_search(question, k=5)
            context = "\n".join(context_chunks)
            full_prompt = build_financial_prompt(context, question)
            logging.info(f"Querying LLM with context: {full_prompt}")
            time.sleep(2)  # Add delay to reduce request frequency
            answer = self.llm.invoke(full_prompt)
            self.cache[cache_key] = answer
            logging.info(f"LLM response: {answer}")
            return answer
        except Exception as e:
            logging.error(f"Error in query: {e}")
            return None

    def _parse_file(self, file_path: str) -> pd.DataFrame:
        ext = file_path.split('.')[-1].lower()
        if ext == "csv":
            return pd.read_csv(file_path)
        elif ext == "xlsx":
            return pd.read_excel(file_path)
        elif ext == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.json_normalize(data)
        elif ext == "pdf" and PyPDF2:
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            return pd.DataFrame({"text": [text]})
        else:
            raise ValueError(f"Unsupported file type or missing PDF parser for: {file_path}")

    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove duplicates, handle missing values, normalize columns
        df = df.drop_duplicates()
        df = df.fillna('N/A')
        # Add more cleaning as needed
        logging.info(f"Validated and cleaned data: {df.shape[0]} rows.")
        return df

    def _chunk_texts(self, texts: List[str]) -> List[str]:
        # Chunking large texts for efficient embedding
        chunks = []
        for text in texts:
            start = 0
            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))
                chunk = text[start:end]
                chunks.append(chunk)
                start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _build_prompt(self, question: str) -> str:
        # Deprecated: now using build_financial_prompt from prompt.py
        pass
