import os
import logging
import pandas as pd
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from config import OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL, DATA_PATH
from prompt import FINANCIAL_PROMPT
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)

CHUNK_SIZE = 500  # Number of characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks

class FinancialRAGPipeline:
    def __init__(self, data_path=DATA_PATH, openai_api_key=OPENAI_API_KEY):
        self.data_path = data_path
        self.openai_api_key = openai_api_key
        self.embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None
        self.llm = OpenAI(openai_api_key=openai_api_key, model_name=LLM_MODEL)
        self.qa_chain = None
        logging.info(f"Initialized pipeline with data: {data_path}")

    def load_and_embed(self):
        logging.info("Loading financial data for embedding...")
        df = pd.read_csv(self.data_path)
        texts = df.astype(str).apply(lambda row: ' '.join(row), axis=1).tolist()
        chunks = self._chunk_texts(texts)
        logging.info(f"Embedding {len(chunks)} chunks of data.")
        self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        retriever = self.vector_store.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=retriever)
        logging.info("Embedding and retrieval setup complete.")

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

    def query(self, question):
        if not self.qa_chain:
            raise Exception("Pipeline not initialized. Call load_and_embed() first.")
        prompt = self._build_prompt(question)
        logging.info(f"Querying LLM with: {prompt}")
        return self.qa_chain.run(prompt)

    def _build_prompt(self, question):
        # Use prompt from prompt.py
        return f"{FINANCIAL_PROMPT}\nQuestion: {question}\n"
