import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from llama_cpp import Llama

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load LLaMA model
llm = Llama(model_path="models\mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf")  # <-- Your .gguf model file

# Function to read PDF
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text, len(pdf_reader.pages)

# Split text into chunks
def split_text(text, chunk_size):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Embed a chunk
def embed_chunk(chunk):
    return embed_model.encode(chunk)

# Retrieve top context
def retrieve_context(question_embedding, chunk_embeddings, chunks, top_k=3):
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_chunks = [chunks[i] for i in top_indices if similarities[i] > 0.3]  # Only relevant ones
    return "\n\n".join(top_chunks)

# Generate answer
def generate_answer(context, question):
    prompt = f"""You are an expert teacher. Answer the following question **ONLY** using the provided context below. 
If the answer is not in the context, reply "I don't know."

Context:
{context}

Question: {question}

Answer:"""
    
    output = llm(prompt, max_tokens=512)
    return output["choices"][0]["text"].strip()

# Streamlit App
st.title("📚 Robot Teacher — Ask your PDF")

uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_pdf:
    pdf_text, num_pages = read_pdf(uploaded_pdf)
    num_tokens = len(pdf_text.split())
    
    # Dynamic chunk size
    target_chunks = 15
    chunk_size = max(100, int(num_tokens / target_chunks))
    
    st.write(f"PDF loaded: {num_pages} pages, {num_tokens} tokens.")
    st.write(f"Auto CHUNK_SIZE: {chunk_size} words per chunk.")
    
    chunks = split_text(pdf_text, chunk_size)
    chunk_embeddings = [embed_chunk(chunk) for chunk in chunks]
    
    question = st.text_input("Ask your question about this PDF:")
    
    if question:
        question_embedding = embed_chunk(question)
        context = retrieve_context(question_embedding, chunk_embeddings, chunks)
        
        if context.strip() == "":
            st.write("🤖 I don't know. This topic was not found in the provided PDF.")
        else:
            with st.spinner("Thinking..."):
                answer = generate_answer(context, question)
                st.write("### 🤖 Answer:")
                st.write(answer)
