# app/routes/qa.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.core.rag import search_answer

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_question(req: QuestionRequest):
    try:
        answer = search_answer(req.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

