import fitz
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import cohere

co = cohere.Client("yNiOGAbn3zaiuAFjVj3pwW8e6eZ2mjIJmf2VcgKh")

def generate_answer(question, context):
    max_context_length = 1000  # or a smaller number based on testing
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    
    response = co.generate(
        model='command-light',
        prompt=prompt,
        max_tokens=200,
        temperature=0.5,
        k=0,
        stop_sequences=["\n"]
    )
    return response.generations[0].text.strip()

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:

        for page in doc:
            text += page.get_text()
    return text

def chunk_text(text, max_tokens = 500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
       if len(current_chunk + sentence) <= max_tokens:
           current_chunk += sentence + '.'
       else:
           chunks.append(current_chunk.strip())
           current_chunk = sentence + '.'

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()



           



st.title("PDF TO AI Chatbot")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    text= extract_text_from_pdf(uploaded_file)

    st.subheader("Extracted Text:")
    st.write(text[:9000])
    

    chunks = chunk_text(text)
    st.info(f" Total Chunks: {len(chunks)}")

    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    st.success("Embeddings created and indexed successfully!")


    st.subheader("💬 Ask your PDF a question")
    user_question = st.text_input("Type your question:")
    if user_question:
        question_embedding = model.encode([user_question])
        D, I = index.search(np.array(question_embedding), k=1)
        answer = chunks[I[0][0]]
        answer = generate_answer(user_question, answer)

        st.success("🧠 Best match from PDF:")
        st.write(answer)

