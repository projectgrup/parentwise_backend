from langdetect import detect
from deep_translator import GoogleTranslator

def translate_text(text: str, target_lang: str):
    src = detect(text)
    if src == target_lang:
        return text
    return GoogleTranslator(source=src, target=target_lang).translate(text)



