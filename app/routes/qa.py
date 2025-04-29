from fastapi import APIRouter, Form

# ✅ Import translation (must be implemented in app/utils/translator.py)
try:
    from app.utils.translator import translate_text
    use_translate = True
except:
    def translate_text(text, lang):
        return text
    use_translate = False

# ✅ Try importing RAG (optional)
try:
    from app.core.rag import search_question
    use_rag = True
except:
    use_rag = False

router = APIRouter()

# ✅ Fallback answers if RAG is unavailable
fallback_answers = [
    "Toddlers should eat a variety of soft fruits, cereals, and vegetables.",
    "A 2-year-old usually needs 11–14 hours of sleep in 24 hours.",
    "Always supervise toddlers during play to ensure safety."
]

@router.post("/ask")
def ask_parent_question(question: str = Form(...), lang: str = Form("en")):
    try:
        # Translate question to English if translator is available
        q_en = translate_text(question, "en") if use_translate else question

        # Use RAG if available, else fallback
        if use_rag:
            results = search_question(q_en)
            answer = results[0] if results else fallback_answers[0]
        else:
            answer = fallback_answers[0]

        # Translate response back to user language
        translated = translate_text(answer, lang) if use_translate else answer
        return {"response": translated}
    except Exception as e:
        return {"response": f"❌ Error processing question. Please try again. (Debug: {str(e)})"}
