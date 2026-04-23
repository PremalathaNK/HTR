"""
Translation Service — Google Translator (deep-translator) with language routing.

Uses deep_translator.GoogleTranslator for all languages (including Indian ones).
Easy to swap in NLLB or IndicTrans2 later when you want offline/fine-tuned translation.
"""

from deep_translator import GoogleTranslator

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


def translate_text(text: str, target_lang: str, source_lang: str = "auto") -> dict:
    """
    Translate text from source_lang to target_lang.

    Args:
        text: Input text to translate
        target_lang: BCP-47 language code (e.g. "hi", "ta", "fr")
        source_lang: Source language code or "auto" for auto-detection

    Returns:
        dict with keys:
            translated_text (str): Translated output
            source_lang (str): Detected or given source lang
            target_lang (str): Target language code
            target_lang_name (str): Human-readable target language name
    """
    if not text or not text.strip():
        return {
            "translated_text": "",
            "source_lang": source_lang,
            "target_lang": target_lang,
            "target_lang_name": SUPPORTED_LANGUAGES.get(target_lang, target_lang),
        }

    # deep-translator uses "auto" for auto-detection
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    translated = translator.translate(text.strip())

    return {
        "translated_text": translated or text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "target_lang_name": SUPPORTED_LANGUAGES.get(target_lang, target_lang),
    }


def get_supported_languages() -> dict:
    """Return supported language map for the frontend language selector."""
    return SUPPORTED_LANGUAGES
