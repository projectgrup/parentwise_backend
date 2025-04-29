from fastapi import APIRouter, Form
from app.utils.translator import translate_text

# Optional import (only if Redis + RAG are working)
try:
    from app.core.rag import search_question
    use_rag = True
except:
    use_rag = False

router = APIRouter()

# Fallback answers if RAG is unavailable
fallback_answers = [
    "Toddlers should eat a variety of soft fruits, cereals, and vegetables.",
    "A 2-year-old usually needs 11–14 hours of sleep in 24 hours.",
    "Always supervise toddlers during play to ensure safety."
]

@router.post("/ask")
def ask_parent_question(question: str = Form(...), lang: str = Form("en")):
    try:
        q_en = translate_text(question, "en")

        # Use RAG if available, else fallback
        if use_rag:
            results = search_question(q_en)
            answer = results[0] if results else fallback_answers[0]
        else:
            answer = fallback_answers[0]

        translated = translate_text(answer, lang)
        return {"response": translated}
    except Exception as e:
        return {"response": f"❌ Error processing question. Please try again. (Debug: {str(e)})"}
