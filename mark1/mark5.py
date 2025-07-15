import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import PyPDF2
import re

# Load LLM
llm = Llama(model_path="models/mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf", n_ctx=4096)  # your model path

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# PDF reading
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Text chunking
def chunk_text(text, chunk_size):
    words = re.findall(r'\w+', text)
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Embed a chunk
def embed_chunk(chunk):
    return embed_model.encode(chunk)

# Retrieve context
def retrieve_context(question_embedding, chunk_embeddings, chunks, top_k=3):
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    context = "\n\n".join([chunks[i] for i in top_indices])
    return context

# Generate answer
def generate_answer(context, question):
    messages = [
        {"role": "user", "content": f"Use the following context to answer: \n\n{context}\n\nQuestion: {question}"}
    ]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.7
    )
    return output["choices"][0]["message"]["content"]

# Streamlit UI
st.title("🤖 Robot Teacher")

uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_pdf is not None:
    pdf_text = read_pdf(uploaded_pdf)
    token_count = len(pdf_text.split())  # simple estimate
    auto_chunk_size = max(100, min(300, token_count // 10))
    st.write(f"PDF loaded! Auto CHUNK_SIZE = {auto_chunk_size} words per chunk.")

    chunks = chunk_text(pdf_text, chunk_size=auto_chunk_size)
    chunk_embeddings = [embed_chunk(chunk) for chunk in chunks]

    question = st.text_input("Ask your question about this PDF:")

    if question:
        question_embedding = embed_model.encode(question)
        context = retrieve_context(question_embedding, chunk_embeddings, chunks)
        answer = generate_answer(context, question)
        st.write("### 🤖 Answer:")
        st.write(answer)
