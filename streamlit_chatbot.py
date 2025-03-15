import os
import re
import requests
import PyPDF2
import streamlit as st
import torch
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import TFIDFRetriever
from langchain.retrievers import EnsembleRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
import asyncio

nltk.download('punkt')
nltk.download('stopwords')

# Streamlit UI
st.set_page_config(page_title="Financial RAG ChatBot", page_icon="📊", layout="centered")
st.title("📊 Financial RAG ChatBot")
st.markdown("Ask questions related to the last two years' financial statements.")

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))
    
nltk_data_path = "/home/appuser/nltk_data"
if not os.path.exists(nltk_data_path):
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)

nltk.data.path.append(nltk_data_path)

### **1️⃣ Data Collection & Preprocessing** """
def load_financial_statements(directory):
    """Loads financial statements from a directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            try:
                with open(os.path.join(directory, filename), "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text += page.extract_text() or ""  
                    documents.append(text)  # Store extracted text
            except (FileNotFoundError, PyPDF2.errors.PdfReadError) as e:
                st.error(f"Error loading PDF: {e}")
    return documents

### **2️⃣ Text Chunking for Retrieval** """
def chunk_documents(documents, chunk_size=500, overlap=50):
    """Chunks financial documents using optimal chunk size & overlap."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.create_documents(["".join(documents)])
    return chunks

### **3️⃣ FAISS Vector Store Creation** """
def create_vectorstore(chunks):
    """Creates a FAISS vector store using Sentence Transformers."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

### **4️⃣ BM25 Sparse Vector Retrieval** """
def create_bm25_retriever(chunks):
    """Creates a BM25 retriever."""
    tokenized_corpus = [word_tokenize(doc.page_content.lower()) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks

def bm25_search(query, bm25, documents, k=4):
    """Retrieves top-k documents using BM25."""
    query_tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [documents[i] for i in top_indices]

### **5️⃣ Hybrid Search: Combining BM25 + FAISS Results** """
def combine_results(retrieved_faiss, retrieved_bm25):
    """Combines FAISS & BM25 results without duplication."""
    combined = []
    seen = set()
    for doc in retrieved_faiss + retrieved_bm25:
        if doc.page_content not in seen:
            combined.append(doc)
            seen.add(doc.page_content)
    return combined

### **6️⃣ DistilGPT-2 as SLM for Response Generation** """
def load_distilgpt2():
    """Loads the DistilGPT-2 model & tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model

def generate_response_distilgpt2(query, context, tokenizer, model, max_new_tokens=100):
    """Generates response using DistilGPT-2."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input_text = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=5, early_stopping=True, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

### **7️⃣ User Input Guardrails** """
def is_financial_query(query):
    """Filters out non-financial queries using NLP-based keyword validation."""
    financial_keywords = ["revenue", "profit", "loss", "cash flow", "balance sheet", "income statement", "financial", "earnings", "assets", "liabilities", "equity", "reserves", "dividend"]
    query_tokens = word_tokenize(query.lower())
    filtered_tokens = [w for w in query_tokens if w not in stopwords.words('english')]

    return any(keyword in filtered_tokens for keyword in financial_keywords)

### **8️⃣ Output Formatting & Confidence Score Calculation** """
def extract_answer(output):
    """Extracts only the generated answer from response."""
    match = re.search(r"Answer:\s*(.+)", output, re.DOTALL)
    return match.group(1).strip() if match else "Answer not found."

def calculate_confidence(query, retrieved_docs):
    """Assigns a confidence score based on retrieval relevance."""
    return min(len(retrieved_docs) * 10, 100)  # Example formula (scaling factor)

### **9️⃣ RAG Query Execution: Testing & Validation** """
# Load financial documents & create embeddings
documents = load_financial_statements(".")
chunks = chunk_documents(documents)
bm25, bm25_docs = create_bm25_retriever(chunks)
vectorstore = create_vectorstore(chunks)

# Load the Small Language Model (SLM)
tokenizer, model = load_distilgpt2()

""" **🔍 Streamlit UI & Response Handling** """
user_query = st.text_input("🔍 Ask a financial question:")

if st.button("Submit"):
    if user_query:
        if not is_financial_query(user_query):
            st.error("⚠️ This does not appear to be a financial question. Please ask a relevant query.")
        else:
            with st.spinner("🔎 Searching for relevant financial data..."):
                retrieved_faiss = vectorstore.as_retriever(search_kwargs={"k": 4}).get_relevant_documents(user_query)
                retrieved_bm25 = bm25_search(user_query, bm25, bm25_docs, k=4)
                retrieved_docs = combine_results(retrieved_faiss, retrieved_bm25)

                if retrieved_docs:
                    context = [doc.page_content for doc in retrieved_docs[:2]]
                    response = generate_response_distilgpt2(user_query, " ".join(context), tokenizer, model)
                    confidence_score = calculate_confidence(user_query, retrieved_docs)

                    st.success(f"✅ Answer Found (Confidence: {confidence_score:.2f}%)")
                    st.write(extract_answer(response))
                else:
                    st.error("❌ No relevant financial information found.")
    else:
        st.warning("⚠️ Please enter a valid question.")
