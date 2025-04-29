from langdetect import detect
from deep_translator import GoogleTranslator

def translate_text(text, target_lang):
    try:
        if not text.strip():
            return text  # avoid translating empty strings

        # Detect source language
        src_lang = detect(text)

        # If already in target language, skip translation
        if src_lang == target_lang:
            return text

        # Translate using GoogleTranslator
        return GoogleTranslator(source=src_lang, target=target_lang).translate(text)

    except Exception as e:
        # If detection or translation fails, return original text
        print(f"[Translation Error] {e}")
        return text

