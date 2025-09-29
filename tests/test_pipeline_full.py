import os
import pandas as pd
import pytest
from src.pipeline import FinancialRAGPipeline

TEST_API_KEY = os.getenv("OPENAI_API_KEY", "test-key")
TEST_DATA_PATH = "tests/test_data.csv"
TEST_JSON_PATH = "tests/test_data.json"
TEST_XLSX_PATH = "tests/test_data.xlsx"

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Create test CSV
    df = pd.DataFrame({"col1": ["A", "B", "C"], "col2": [1, 2, 3]})
    df.to_csv(TEST_DATA_PATH, index=False)
    # Create test JSON
    df.to_json(TEST_JSON_PATH, orient="records")
    # Create test XLSX
    df.to_excel(TEST_XLSX_PATH, index=False)
    yield
    for path in [TEST_DATA_PATH, TEST_JSON_PATH, TEST_XLSX_PATH]:
        if os.path.exists(path):
            os.remove(path)

def test_init():
    pipeline = FinancialRAGPipeline(TEST_DATA_PATH, TEST_API_KEY)
    assert pipeline.data_path == TEST_DATA_PATH
    assert pipeline.openai_api_key == TEST_API_KEY

def test_load_and_embed():
    pipeline = FinancialRAGPipeline(TEST_DATA_PATH, TEST_API_KEY)
    pipeline.load_and_embed()
    assert pipeline.vector_store is not None

def test_chunking():
    pipeline = FinancialRAGPipeline(TEST_DATA_PATH, TEST_API_KEY)
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

def test_parse_file_json():
    pipeline = FinancialRAGPipeline(TEST_JSON_PATH, TEST_API_KEY)
    df = pipeline._parse_file(TEST_JSON_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3

def test_parse_file_xlsx():
    pipeline = FinancialRAGPipeline(TEST_XLSX_PATH, TEST_API_KEY)
    df = pipeline._parse_file(TEST_XLSX_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3

def test_validate_and_clean_data():
    pipeline = FinancialRAGPipeline(TEST_DATA_PATH, TEST_API_KEY)
    df = pd.DataFrame({"col1": ["A", "A", None], "col2": [1, 1, None]})
    cleaned = pipeline._validate_and_clean_data(df)
    assert cleaned.isnull().sum().sum() == 0
    assert cleaned.shape[0] == 2

def test_similarity_search():
    pipeline = FinancialRAGPipeline(TEST_DATA_PATH, TEST_API_KEY)
    pipeline.load_and_embed()
    results = pipeline.similarity_search("A", k=2)
    assert isinstance(results, list)

def test_query_method(monkeypatch):
    pipeline = FinancialRAGPipeline(TEST_DATA_PATH, TEST_API_KEY)
    pipeline.load_and_embed()
    monkeypatch.setattr(pipeline, "similarity_search", lambda q, k=5: ["context"])
    monkeypatch.setattr(pipeline.llm, "invoke", lambda prompt: "answer")
    result = pipeline.query("A?")
    assert result == "answer"
