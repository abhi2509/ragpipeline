import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from pipeline import FinancialRAGPipeline

st.title("Financial Data RAG Pipeline")

api_key = st.text_input("Enter your OpenAI API Key:", type="password")

uploaded_file = st.file_uploader("Upload your financial data (CSV)", type=["csv"])

if uploaded_file and api_key:
    data_path = os.path.join("data", uploaded_file.name)
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name}")
    pipeline = FinancialRAGPipeline(data_path, api_key)
    pipeline.load_and_embed()
    st.session_state["pipeline"] = pipeline

if "pipeline" in st.session_state:
    question = st.text_input("Ask a question about your financial data:")
    if question:
        answer = st.session_state["pipeline"].query(question)
        st.write("**Answer:**", answer)
