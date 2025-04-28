
from fastapi import APIRouter, Form
from app.utils.translator import translate_text
from app.core.rag import search_question

router = APIRouter()

@router.post("/ask")
def ask_parent_question(question: str = Form(...), lang: str = Form(...)):
    q_en = translate_text(question, "en")
    response = search_question(q_en)
    return {"response": translate_text(response[0], lang)}
