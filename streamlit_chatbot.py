import os
import PyPDF2
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import pipeline
import re
import nltk
from nltk.tokenize import sent_tokenize

# Streamlit UI
st.set_page_config(page_title="Financial RAG ChatBot", page_icon="📊", layout="centered")
st.title("📊 Financial RAG ChatBot")
st.markdown("Ask questions related to the last two years' financial statements.")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load summarization model
summarizer = pipeline("summarization")

# PDF Directory
dataset_folder = "."

# Extract text from PDFs with improved chunking
def extract_text_from_pdfs(pdf_files):
    text_chunks = []
    for pdf_path in pdf_files:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                sentences = sent_tokenize(text)
                chunk = []
                for sentence in sentences:
                    chunk.append(sentence)
                    if len(" ".join(chunk)) > 300:
                        text_chunks.append(" ".join(chunk))
                        chunk = []
                if chunk:
                    text_chunks.append(" ".join(chunk))
    return text_chunks

pdf_files = [os.path.join(dataset_folder, file) for file in os.listdir(dataset_folder) if file.endswith(".pdf")]
text_chunks = extract_text_from_pdfs(pdf_files)
tokenized_corpus = [chunk.split() for chunk in text_chunks]

# Initialize BM25
bm25 = BM25Okapi(tokenized_corpus)

# Encode text using embeddings
embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Guard Rails - Validate user query with LLM-based filtering
def validate_query(query):
    prohibited_patterns = [
        r'\b(violence|hate speech|illegal activity|politics|religion|stupid|dumb|bad|useless)\b',
        r'[^a-zA-Z0-9\s?]',
        r'\b(capital of|who is the president of|weather in|history of)\b'
    ]
    if any(re.search(pattern, query, re.IGNORECASE) for pattern in prohibited_patterns):
        return False
    return True

# Hybrid Retrieval with Re-ranking
def hybrid_retrieval(query, top_k=5):
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_k_indices = np.argsort(bm25_scores)[-top_k:][::-1]

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, dense_top_k_indices = index.search(query_embedding, top_k)
    dense_scores = 1 / (1 + distances)

    retrieved_texts = {}
    for i, idx in enumerate(bm25_top_k_indices):
        retrieved_texts[idx] = bm25_scores[idx] * 2.0
    for i, idx in enumerate(dense_top_k_indices[0]):
        retrieved_texts[idx] = retrieved_texts.get(idx, 0) + dense_scores[0][i] * 1.5

    sorted_indices = sorted(retrieved_texts, key=retrieved_texts.get, reverse=True)

    # Re-ranking with CrossEncoder
    if sorted_indices:
        query_doc_pairs = [[query, text_chunks[idx]] for idx in sorted_indices[:top_k]]
        rerank_scores = reranker.predict(query_doc_pairs)
        sorted_indices = [idx for _, idx in sorted(zip(rerank_scores, sorted_indices), reverse=True)]

    max_confidence_score = retrieved_texts[sorted_indices[0]] if sorted_indices else 0
    return [text_chunks[idx] for idx in sorted_indices[:top_k]], max_confidence_score

# Hybrid Retrieval with Summarization
def hybrid_retrieval_with_summary(query, top_k=3):
    retrieved_texts, confidence_score = hybrid_retrieval(query, top_k)
    summarized_texts = []
    for text in retrieved_texts:
        if len(text) > 50:
            summary = summarizer(text, max_length=100, min_length=50, do_sample=False)
            summarized_texts.append(summary[0]["summary_text"])
        else:
            summarized_texts.append(text)
    return summarized_texts, confidence_score

# Add a text input field for user questions
user_query = st.text_input("🔍 Ask a financial question:", "")

# Display results when the user clicks submit
if st.button("Submit"):
    if user_query:
        if not validate_query(user_query):
            st.error("⚠️ Your query is not relevant to financial reports. Please enter a valid financial question.")
        else:
            with st.spinner("Searching and summarizing..."):
                summarized_response, confidence_score = hybrid_retrieval_with_summary(user_query)
                if summarized_response:
                    st.success(f"✅ Answer Found (Confidence: {confidence_score:.2f})")
                    st.write("\n\n".join(summarized_response))
                else:
                    st.error("❌ No relevant information found.")
    else:
        st.warning("⚠️ Please enter a question.")
