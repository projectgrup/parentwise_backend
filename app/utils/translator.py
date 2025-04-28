
from langdetect import detect
from deep_translator import GoogleTranslator

def translate_text(text, target_lang):
    src = detect(text)
    return GoogleTranslator(source=src, target=target_lang).translate(text) if src != target_lang else text
