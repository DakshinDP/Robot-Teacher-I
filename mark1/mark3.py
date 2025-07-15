import streamlit as st
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ----- Config -----
CHUNK_SIZE = 50
TOP_K = 5

# ----- Load embedding model -----
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ----- Load LLaMA model -----
llm = Llama(model_path="models\mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf", chat_format="mistral-instruct")

# ----- Read PDF -----
def read_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# ----- Split PDF into chunks -----
def split_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# ----- Get embedding for chunk -----
def embed_chunk(text):
    embedding = embed_model.encode(text)
    return np.array(embedding, dtype=np.float32)

# ----- Retrieve context -----
def retrieve_context(question_embedding, chunk_embeddings, chunks, top_k=TOP_K):
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    selected_chunks = [chunks[i] for i in top_indices]
    return "\n\n".join(selected_chunks)

# ----- Generate answer -----
def generate_answer(context, question):
    prompt = f"""You are a helpful teacher. Answer the question using only the given context.

Context:
{context}

Question: {question}
Answer:"""

    output = llm(prompt, max_tokens=512)
    return output["choices"][0]["text"].strip()
#"You are an expert teacher. Answer the following question **ONLY** using the provided context below. If the answer is not in the context, reply 'I don't know.'\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

# ----- Streamlit UI -----
st.title("🤖 Robot Teacher 📚")

uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_pdf is not None:
    pdf_text = read_pdf(uploaded_pdf)
    chunks = split_text(pdf_text)
    st.success(f"PDF loaded! Total chunks: {len(chunks)}")

    chunk_embeddings = np.vstack([embed_chunk(c) for c in chunks])
    st.success("Embeddings generated ✅")

    question = st.text_input("Ask a question:")

    if question:
        question_embedding = embed_chunk(question)
        context = retrieve_context(question_embedding, chunk_embeddings, chunks)
        answer = generate_answer(context, question)

        st.markdown("### 📖 Answer:")
        st.write(answer)
