import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    DATA_PATH = os.getenv("DATA_PATH", "data/financial_data.csv")

    @staticmethod
    def validate():
        missing = []
        if not Config.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not Config.DATA_PATH:
            missing.append("DATA_PATH")
        if missing:
            raise ValueError(f"Missing required config values: {', '.join(missing)}")

# Usage: Config.validate() to check config, and access values via Config.OPENAI_API_KEY, etc.
