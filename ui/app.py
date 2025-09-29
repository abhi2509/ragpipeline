import streamlit as st
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from pipeline import FinancialRAGPipeline
import pandas as pd
import json
from io import BytesIO
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

"""
Streamlit UI for Financial Data RAG Pipeline

- Allows users to upload financial data in CSV, Excel, JSON, or PDF format
- Parses and normalizes uploaded data
- Initializes and runs the RAG pipeline for question answering
- Reads OpenAI API key from environment variable
- Shows query performance and retrieved context
"""

st.set_page_config(page_title="Financial RAG Pipeline", layout="wide")

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload your financial data (CSV, Excel, JSON, PDF).
    2. Ask questions about your data.
    3. View query performance and retrieved context.
    """)
    st.markdown("---")
    st.markdown("**Tip:** Use specific questions for best results.")

st.title("Financial Data RAG Pipeline")

api_key = st.text_input("Enter your OpenAI API Key:", type="password")

uploaded_file = st.file_uploader(
    "Upload your financial data (CSV, Excel, JSON, PDF)", type=["csv", "xlsx", "json", "pdf"])

def parse_file(uploaded_file):
    """
    Parse uploaded file and return its path and DataFrame.
    Supports CSV, Excel, JSON, and PDF formats.
    """
    if uploaded_file is None:
        return None, None
    filename = uploaded_file.name
    ext = filename.split('.')[-1].lower()
    data_path = os.path.join("data", filename)
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    if ext == "csv":
        df = pd.read_csv(data_path)
    elif ext == "xlsx":
        df = pd.read_excel(data_path)
    elif ext == "json":
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.json_normalize(data)
    elif ext == "pdf" and PyPDF2:
        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getbuffer()))
        text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        df = pd.DataFrame({"text": [text]})
    else:
        st.error("Unsupported file type or missing PDF parser.")
        return None, None
    return data_path, df

if uploaded_file and api_key:
    data_path, df = parse_file(uploaded_file)
    if df is not None:
        st.success(f"Uploaded and parsed {uploaded_file.name}")
        # Show data preview and charts
        st.subheader("Data Preview")
        st.dataframe(df.head(20))
        st.subheader("Data Charts")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 1:
            for col in numeric_cols:
                st.line_chart(df[col], height=200, use_container_width=True)
        else:
            st.info("No numeric columns available for charting.")
        # Save parsed data as CSV for pipeline
        csv_path = data_path if data_path.endswith(".csv") else data_path + ".csv"
        df.to_csv(csv_path, index=False)
        pipeline = FinancialRAGPipeline(csv_path, api_key)
        pipeline.load_and_embed()
        st.session_state["pipeline"] = pipeline

if "pipeline" in st.session_state:
    question = st.text_input("Ask a question about your financial data:")
    if question:
        start_time = time.time()
        # Retrieve context chunks for display
        context_chunks = st.session_state["pipeline"].similarity_search(question, k=5)
        answer = st.session_state["pipeline"].query(question)
        elapsed = time.time() - start_time
        st.markdown(f"**Query Performance:** {elapsed:.2f} seconds")
        with st.expander("Show Retrieved Context"):
            for i, chunk in enumerate(context_chunks, 1):
                st.markdown(f"**Chunk {i}:** {chunk}")
        st.write("**Answer:**", answer)
