import os
import pandas as pd
import pytest
from src.pipeline import FinancialRAGPipeline

TEST_API_KEY = os.getenv("OPENAI_API_KEY", "test-key")
TEST_DATA_PATH = "tests/test_data.csv"

def setup_module(module):
    # Create a small test CSV file
    df = pd.DataFrame({
        "col1": ["A", "B", "C"],
        "col2": [1, 2, 3]
    })
    df.to_csv(TEST_DATA_PATH, index=False)

def teardown_module(module):
    if os.path.exists(TEST_DATA_PATH):
        os.remove(TEST_DATA_PATH)

def test_load_and_embed():
    pipeline = FinancialRAGPipeline(TEST_DATA_PATH, TEST_API_KEY)
    pipeline.load_and_embed()
    assert pipeline.retrieval_manager is not None
    assert pipeline.retrieval_manager.vector_store is not None

def test_chunking():
    pipeline = FinancialRAGPipeline(TEST_DATA_PATH, TEST_API_KEY)
    pipeline.load_and_embed()
    chunks = pipeline._chunk_texts(["A" * 1000])
    assert len(chunks) > 1

def test_query_cache():
    pipeline = FinancialRAGPipeline(TEST_DATA_PATH, TEST_API_KEY)
    pipeline.load_and_embed()
    pipeline.cache["dummy"] = "cached answer"
    assert pipeline.cache["dummy"] == "cached answer"

def test_parse_file_csv():
    pipeline = FinancialRAGPipeline(TEST_DATA_PATH, TEST_API_KEY)
    df = pipeline._parse_file(TEST_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3
