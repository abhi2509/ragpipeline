import logging
from langchain_community.vectorstores import FAISS

class RetrievalManager:
    def __init__(self, embeddings):
        self.vector_store = None
        self.embeddings = embeddings

    def build_vector_store(self, texts):
        self.vector_store = FAISS.from_texts(texts, self.embeddings)
        logging.info("Vector store built with FAISS.")

    def similarity_search(self, query, k=5):
        if not self.vector_store:
            raise Exception("Vector store not initialized.")
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
