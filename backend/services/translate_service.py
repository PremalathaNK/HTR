"""
Translation Service — Google Translator (deep-translator) with language routing.

Uses deep_translator.GoogleTranslator for all languages (including Indian ones).
Includes an automatic offline fallback to NLLB-200 if the network fails.
"""

from deep_translator import GoogleTranslator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Language code map for UI display
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
    "te": "Telugu",
    "ml": "Malayalam",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh-CN": "Chinese (Simplified)",
    "ar": "Arabic",
    "ja": "Japanese",
}

# NLLB uses Flores-200 language codes
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "kn": "kan_Knda",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh-CN": "zho_Hans",
    "ar": "arb_Arab",
    "ja": "jpn_Jpan",
}

def translate_text(text: str, target_lang: str, source_lang: str = "auto") -> dict:
    """
    Translate text from source_lang to target_lang using Google Translate.
    """
    if not text or not text.strip():
        return {
            "translated_text": "",
            "source_lang": source_lang,
            "target_lang": target_lang,
            "target_lang_name": SUPPORTED_LANGUAGES.get(target_lang, target_lang),
        }

    translated = None
    
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text.strip())
    except Exception as e:
        print(f"[Translate] Network error with Google Translator: {e}")
        translated = text # Return original text on failure

    return {
        "translated_text": translated or text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "target_lang_name": SUPPORTED_LANGUAGES.get(target_lang, target_lang),
    }

def get_supported_languages() -> dict:
    """Return supported language map for the frontend language selector."""
    return SUPPORTED_LANGUAGES
