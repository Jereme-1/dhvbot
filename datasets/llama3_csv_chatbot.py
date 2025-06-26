import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Streamlit UI setup
st.set_page_config(page_title="DHVBOT", layout="centered")
st.title("🤖 DHVBOT AI")

# Load and prepare multiple CSVs from folder
@st.cache_resource
def load_vectorstore():
    folder_path = "datasets"
    all_texts = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip')  # Skip bad rows
                texts = df.apply(lambda row: f"[{filename}] " + " | ".join([str(x) for x in row]), axis=1).tolist()
                all_texts.extend(texts)
            except Exception as e:
                st.warning(f"⚠️ Skipping file `{filename}` due to error: {e}")
                continue

    if not all_texts:
        st.error("🚫 No valid CSV data found.")
        st.stop()

    docs = [Document(page_content=text) for text in all_texts]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Initialize retriever and LLM
retriever = load_vectorstore().as_retriever()
llm = Ollama(model="phi3:mini")  # 🧠 Ensure Ollama is running `phi3:mini`

# Custom prompt
custom_prompt = PromptTemplate.from_template("""
You are a friendly and helpful university assistant chatbot.
You respond in a casual, conversational tone — like a student helping another student.
Always try to give a helpful answer, even for greetings or unrelated questions.
Bullet form.
Straight forward, consice, clear and reliable information
Make it fast respose
                                             
                                        

Context: {context}

Question: {question}
Answer:
""")

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask a question about your data...")
if user_input:
    # Show user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and show AI response
    with st.chat_message("assistant"):
        try:
            response = qa_chain.run(user_input)
        except Exception as e:
            response = f"⚠️ Error: {str(e)}"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
