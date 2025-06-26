from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import os

app = FastAPI()

# Load vectorstore from all CSVs
def load_vectorstore():
    folder_path = "datasets"
    all_texts = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(folder_path, filename), on_bad_lines='skip')
                texts = df.apply(lambda row: f"[{filename}] " + " | ".join([str(x) for x in row]), axis=1).tolist()
                all_texts.extend([(text, filename) for text in texts])
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

    if not all_texts:
        raise ValueError("No valid CSV data found to load into vectorstore.")

    # Create documents with metadata
    docs = [Document(page_content=text, metadata={"source": filename}) for text, filename in all_texts]

    # Better chunking for retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # More accurate embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Load retriever and LLM
retriever = load_vectorstore()
llm = Ollama(model="phi3:mini")  # or try "phi3:medium", "mistral", "llama3:8b"

# Better custom prompt
custom_prompt = PromptTemplate.from_template("""
You are a helpful and friendly university chatbot. Speak like a student helping another student.
Give short, clear answers. Use bullet points when possible. use the dataset and be caution with the shortcut term dont mixed.

If you are not sure, say: "I'm not sure about that. Please try asking the admin office."

Context: {context}
Question: {question}
Answer:
""")

# QA chain setup
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Request model
class AskRequest(BaseModel):
    message: str

# Ask route
@app.post("/ask")
async def ask(req: AskRequest):
    try:
        result = qa_chain(req.message)
        return {
            "response": result["result"],
            "sources": [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
        }
    except Exception as e:
        return {"error": str(e)}

# Health check
@app.get("/")
def root():
    return {"status": "ok", "message": "University chatbot API is running."}
