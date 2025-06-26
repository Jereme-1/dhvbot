import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import os

# Streamlit setup
st.set_page_config(page_title="DHVBOT", layout="centered")
st.title("🤖 DHVBOT - University Chat Assistant")

# Load and prepare vectorstore from all CSVs
@st.cache_resource
def load_vectorstore():
    folder_path = "datasets"
    all_texts = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip')
                texts = df.apply(lambda row: f"[{filename}] " + " | ".join([str(x) for x in row]), axis=1).tolist()
                all_texts.extend(texts)
            except Exception as e:
                st.warning(f"⚠️ Skipping `{filename}` due to error: {e}")
                continue

    if not all_texts:
        st.error("🚫 No valid CSV data found.")
        st.stop()

    # Convert to documents with optional metadata
    docs = [Document(page_content=text, metadata={"source": filename}) for text, filename in zip(all_texts, [filename]*len(all_texts))]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    # Better embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Load retriever and LLM
retriever = load_vectorstore().as_retriever(search_kwargs={"k": 4})
llm = OllamaLLM(model="phi3:mini")  # Replace with another model if needed

# Optimized prompt
custom_prompt = PromptTemplate.from_template("""
You are DHVBOT, a helpful, smart, and friendly university assistant chatbot.
Speak like a student helping another student. Be brief, clear, and reliable.
Use bullet points when needed. Always respond even to greetings or off-topic questions.
                                             
your local is dhvsu (don honorio ventura state university)

Context:
{context}

Question:
{question}

Answer:
""")

# Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
user_input = st.chat_input("Ask me anything about university services, enrollment, grades...")
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and show AI response
    with st.chat_message("assistant"):
        try:
            result = qa_chain(user_input)
            response = result['result']
            sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
        except Exception as e:
            response = f"⚠️ Error: {str(e)}"
            sources = []

        st.markdown(response)
        if sources:
            st.markdown("**Sources:**")
            for s in set(sources):
                st.write(f"📁 {s}")

        st.session_state.messages.append({"role": "assistant", "content": response})
