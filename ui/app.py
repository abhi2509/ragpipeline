
import streamlit as st
import os
import sys
import time
# Ensure src is in sys.path for absolute imports
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
try:
    from pipeline import FinancialRAGPipeline
    from modules.data_utils import DataUtils
except ImportError as e:
    st.error(f"Import error: {e}. Please check your project structure and PYTHONPATH.")
    raise
import pandas as pd
import json
from io import BytesIO
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

"""
Streamlit UI for Financial Data RAG Pipeline

- Modern light mode UI with custom CSS
- Sidebar instructions and tips
- Data preview and charts
- Query performance and context display
"""

st.set_page_config(page_title="Financial RAG Pipeline", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for light mode and modern look
st.markdown("""
    <style>
    body, .stApp {
        background-color: #18191a;
        color: #f5f6fa;
        font-family: inherit;
    }
    .stApp header {background: #222;}
    .stApp [data-testid="stSidebar"] {
        background-color: #222;
        border-right: 1px solid #333;
        color: #f5f6fa;
    }
    .stApp .stButton>button {
        background-color: #007bff;
        color: #fff;
        border-radius: 6px;
        border: none;
        padding: 0.5em 1.2em;
        font-weight: 500;
        transition: background 0.2s;
    }
    .stApp .stButton>button:hover {
        background-color: #0056b3;
    }
    .stApp .stTextInput>div>input {
        background: #222;
        color: #f5f6fa;
        border-radius: 6px;
        border: 1px solid #444;
        padding: 0.5em;
    }
    .stApp .stDataFrame {
        background: #222;
        color: #f5f6fa;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    .stApp .stExpander {
        background: #23272f;
        border-radius: 8px;
        color: #f5f6fa;
    }
    .stApp .stMarkdown {
        font-size: 1.05em;
        color: #f5f6fa;
    }
    </style>
    """, unsafe_allow_html=True)

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

api_key = os.getenv("OPENAI_API_KEY")

uploaded_file = st.file_uploader(
    "Upload your financial data (CSV, Excel, JSON, PDF)", type=["csv", "xlsx", "json", "pdf"])

from modules.data_utils import DataUtils

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
    df = DataUtils.clean_data(df)
    return data_path, df

if uploaded_file and api_key:
    data_path, df = parse_file(uploaded_file)
    if df is not None:
        st.success(f"Uploaded and parsed {uploaded_file.name}")
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
    if "question" not in st.session_state:
        st.session_state["question"] = ""
    question = st.text_input("Ask a question about your financial data:", value=st.session_state["question"], key="question_input")
    if question:
        start_time = time.time()
        context_chunks = st.session_state["pipeline"].similarity_search(question, k=5)
        answer = st.session_state["pipeline"].query(question)
        elapsed = time.time() - start_time
        st.markdown(f"**Query Performance:** {elapsed:.2f} seconds")
        with st.expander("Show Retrieved Context"):
            for i, chunk in enumerate(context_chunks, 1):
                st.markdown(f"**Chunk {i}:** {chunk}")
        st.write("**Answer:**", answer)
        # Clear input for next question
        st.session_state["question"] = ""
