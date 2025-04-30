from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI()

# Just a test route
@app.get("/")
def root():
    return {"message": "ParentWise API is running."}

# Input model
class AskInput(BaseModel):
    question: str

# Main Q&A endpoint
@app.post("/ask")
def ask_question(payload: AskInput):
    # 🔁 Lazy import (loads only when /ask is called)
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI
    from langchain.docstore.document import Document

    # Load FAISS index only when needed
    db = FAISS.load_local("app/core/faiss_index", OpenAIEmbeddings())

    # Search for similar chunks
    docs = db.similarity_search(payload.question, k=3)

    # Run a QA chain
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    result = chain.run(input_documents=docs, question=payload.question)

    return {"answer": result}



