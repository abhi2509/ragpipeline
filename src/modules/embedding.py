import logging
from langchain_community.embeddings import SentenceTransformerEmbeddings

class EmbeddingManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        logging.info(f"Embedding model initialized: {model_name}")

    def embed_texts(self, texts):
        return self.embeddings.embed_documents(texts)
