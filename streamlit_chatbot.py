import os
import re
import requests
import PyPDF2
import streamlit as st
import torch
import numpy as np
import nltk
import asyncio
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import TFIDFRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi

# ✅ 1️⃣ Fix PyTorch '__path__._path' Issue (Set before importing torch)
os.environ["TORCH_USE_RTLD_GLOBAL"] = "1"

# ✅ 2️⃣ Fix asyncio Conflict in Streamlit
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ✅ 3️⃣ Force NLTK Data Path & Download Fix
NLTK_PATH = os.path.join(os.path.expanduser("~"), "nltk_data")

# Ensure the directory exists
os.makedirs(NLTK_PATH, exist_ok=True)

# Set NLTK path environment variable
os.environ["NLTK_DATA"] = NLTK_PATH
nltk.data.path.append(NLTK_PATH)

# Force-download required nltk resources
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=NLTK_PATH)

# ✅ 4️⃣ Streamlit UI Setup
st.set_page_config(page_title="Financial RAG ChatBot", page_icon="📊", layout="centered")
st.title("📊 Financial RAG ChatBot")
st.markdown("Ask questions related to the last two years' financial statements.")

# ✅ 5️⃣ Load Financial PDFs
def load_financial_statements(directory):
    """Loads financial statements from PDFs in the directory."""
    documents = []
    abs_directory = os.path.abspath(directory)
    
    for filename in os.listdir(abs_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(abs_directory, filename)
            try:
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text = "".join([page.extract_text() or "" for page in reader.pages])
                    documents.append(text)
            except (FileNotFoundError, PyPDF2.errors.PdfReadError) as e:
                st.error(f"Error loading PDF: {e}")
    return documents

# ✅ 6️⃣ Text Chunking for Retrieval
def chunk_documents(documents, chunk_size=500, overlap=50):
    """Splits documents into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.create_documents(["".join(documents)])

# ✅ 7️⃣ FAISS Vector Store Creation
def create_vectorstore(chunks):
    """Creates a FAISS vector store using Sentence Transformers."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    return FAISS.from_documents(chunks, embeddings)

# ✅ 8️⃣ BM25 Sparse Vector Retrieval
def create_bm25_retriever(chunks):
    """Creates a BM25 retriever."""
    tokenized_corpus = [word_tokenize(doc.page_content.lower()) for doc in chunks]
    return BM25Okapi(tokenized_corpus), chunks

def bm25_search(query, bm25, documents, k=4):
    """Retrieves top-k documents using BM25."""
    query_tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [documents[i] for i in top_indices]

# ✅ 9️⃣ Hybrid Search: Combining BM25 + FAISS
def combine_results(retrieved_faiss, retrieved_bm25):
    """Combines FAISS & BM25 results without duplication."""
    seen = set()
    combined = [doc for doc in retrieved_faiss + retrieved_bm25 if doc.page_content not in seen and not seen.add(doc.page_content)]
    return combined

# ✅ 🔟 DistilGPT-2 as Response Generator
def load_distilgpt2():
    """Loads DistilGPT-2 model & tokenizer."""
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
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ 🔟 User Input Guardrails
def is_financial_query(query):
    """Filters out non-financial queries using NLP-based keyword validation."""
    financial_keywords = ["revenue", "profit", "loss", "cash flow", "balance sheet", "income statement", "financial", "earnings", "assets", "liabilities", "equity", "reserves", "dividend"]
    query_tokens = word_tokenize(query.lower())
    return any(word in query_tokens for word in financial_keywords)

# ✅ 🔟 Output Formatting & Confidence Score Calculation
def extract_answer(output):
    """Extracts the generated answer from response."""
    match = re.search(r"Answer:\s*(.+)", output, re.DOTALL)
    return match.group(1).strip() if match else "Answer not found."

def calculate_confidence(query, retrieved_docs):
    """Assigns a confidence score based on retrieval relevance."""
    return min(len(retrieved_docs) * 10, 100)

# ✅ 🔟 RAG Query Execution: Testing & Validation
documents = load_financial_statements(".")
chunks = chunk_documents(documents)
bm25, bm25_docs = create_bm25_retriever(chunks)
vectorstore = create_vectorstore(chunks)

tokenizer, model = load_distilgpt2()

# ✅ 🔍 Streamlit UI & Response Handling
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
