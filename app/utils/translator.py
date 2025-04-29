from langdetect import detect
from deep_translator import GoogleTranslator

def translate_text(text, target_lang):
    try:
        if not text.strip():
            return text
        src_lang = detect(text)
        if src_lang == target_lang:
            return text
        return GoogleTranslator(source=src_lang, target=target_lang).translate(text)
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text


