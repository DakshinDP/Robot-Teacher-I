import streamlit as st
import llama_cpp
import os
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ----- Settings -----
MODEL_PATH = "models\mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf"  # CHANGE to your model path
N_CTX = 32768
CHUNK_SIZE = 500  # tokens per chunk
TOP_K = 3  # number of chunks to send as context

# ----- Load LLaMA model -----
llm = llama_cpp.Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    embedding=True,  # enable embeddings
    verbose=False
)

# ----- PDF reading -----

def read_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# ----- Chunking -----
def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

# ----- Embedding -----
def embed_chunk(text):
    embedding = llm.create_embedding(text)["data"][0]["embedding"]
    return np.array(embedding, dtype=np.float32)  # force 1D array


# ----- RAG -----
def retrieve_context(question_embedding, chunk_embeddings, chunks, top_k=TOP_K):
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    selected_chunks = [chunks[i] for i in top_indices]
    return "\n\n".join(selected_chunks)

# ----- Generate Answer -----
def generate_answer(context, question):
    prompt = f"""[INST] You are a helpful teacher. Use the following context to answer the question clearly and step-by-step.

Context:
{context}

Question:
{question}
[/INST]"""
    output = llm(prompt, max_tokens=512)
    return output['choices'][0]['text'].strip()

# ----- Streamlit App -----
st.title("🤖 Robot Teacher App (Optimized)")
st.write("Upload a PDF and ask any question. The bot will only answer from the PDF!")

uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_pdf:
    with st.spinner("Reading PDF..."):
        pdf_text = read_pdf(uploaded_pdf)
        chunks = chunk_text(pdf_text)
        with st.spinner("Generating embeddings..."):
            chunk_embeddings = np.array([embed_chunk(c) for c in chunks])
        st.success("PDF processed! You can now ask questions.")

    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Thinking..."):
            question_embedding = embed_chunk(question)
            context = retrieve_context(question_embedding, chunk_embeddings, chunks)
            answer = generate_answer(context, question)
            st.write("### Answer:")
            st.write(answer)
