import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    Configuration manager for the RAG pipeline project.

    Loads environment variables for API keys, model names, data paths, and LLM parameters.
    Provides a validation method to ensure required configuration values are set.
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    DATA_PATH = os.getenv("DATA_PATH", "data/wix_kb_corpus.jsonl")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))  # Lower temp for factual answers
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
    LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
    LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0"))
    LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.0"))

    @staticmethod
    def validate():
        """
        Validate that all required configuration values are present.
        Raises:
            ValueError: If any required config value is missing.
        """
        missing = []
        if not Config.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not Config.DATA_PATH:
            missing.append("DATA_PATH")
        if missing:
            raise ValueError(f"Missing required config values: {', '.join(missing)}")

# Usage: Config.validate() to check config, and access values via Config.OPENAI_API_KEY, etc.
