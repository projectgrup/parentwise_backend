from fastapi import APIRouter, Form
from app.core.knowledge import search_answer
from app.utils.translator import translate_text

router = APIRouter()

@router.post("/ask")
def ask_question(question: str = Form(...), lang: str = Form(...)):
    q_en = translate_text(question, "en")
    a_en = search_answer(q_en)
    return {"response": translate_text(a_en, lang)}

