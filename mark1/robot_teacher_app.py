import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Load local LLM model
llm = Llama(model_path="models\mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf", n_ctx=4096 ) #n_ctx=32768

# Initialize embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to chunk text
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# Function to embed chunks
def embed_chunk(chunk):
    return embed_model.encode(chunk)

# Build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# Generate answer from LLM
def generate_answer(context, question):
    prompt = f"""
    You are a helpful teacher AI. Answer the question using only the following context.

    Context:
    {context}

    Question:
    {question}

    Answer:"""

    output = llm(prompt, max_tokens=512)
    return output['choices'][0]['text']

# Streamlit UI
st.title("📚 Robot Teacher 🤖")

uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_pdf:
    st.success("PDF uploaded successfully!")
    text = extract_text(uploaded_pdf)
    chunks = chunk_text(text)
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    index = build_faiss_index(embeddings)

    question = st.text_input("Ask me a question based on this PDF:")

    if question:
        question_embedding = embed_model.encode(question)
        D, I = index.search(np.array([question_embedding]), k=5)
        top_chunks = [chunks[i] for i in I[0]]
        context = "\n".join(top_chunks)
        answer = generate_answer(context, question)
        st.write("**Answer:**", answer)
