from fastapi import APIRouter, Form

# Optional: Safe translator fallback
try:
    from app.utils.translator import translate_text
    use_translate = True
except:
    def translate_text(text, lang):
        return text
    use_translate = False

router = APIRouter()

# Dummy fallback response
fallback_response = "👶 For toddlers, pediatricians usually recommend soft fruits, cereals, boiled vegetables, and plenty of water. Always consult your child’s doctor for personal advice."

@router.post("/ask")
def ask_question(question: str = Form(...), lang: str = Form("en")):
    try:
        translated = translate_text(fallback_response, lang) if use_translate else fallback_response
        return {"response": translated}
    except Exception as e:
        return {"response": f"⚠️ Error answering your question. (Debug: {str(e)})"}
