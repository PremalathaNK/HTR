"""
TTS Service — Text-to-Speech using gTTS.

Returns:
- Base64-encoded MP3 audio for easy JSON transport
- Estimated word timestamps for frontend word-highlighting sync

Note: gTTS doesn't provide real word timestamps natively.
We generate estimated timestamps based on average speaking rate (~150 wpm)
so the frontend can still do word highlighting in near-sync.

When you upgrade to edge-tts in the future, you'll get precise timestamps.
"""

import io
import base64
import time
from gtts import gTTS

# Average speaking rates (words per second) per language
# Used to estimate word timestamps
LANG_WPS = {
    "en": 2.5,   # English ~150 wpm
    "hi": 2.2,   # Hindi slightly slower
    "ta": 2.0,   # Tamil
    "kn": 2.0,   # Kannada
    "te": 2.0,   # Telugu
    "ml": 2.0,   # Malayalam
    "fr": 2.8,   # French
    "de": 2.4,   # German
    "es": 3.0,   # Spanish (fast)
    "zh-CN": 3.5, # Chinese (characters = fast)
    "ar": 2.2,
    "ja": 3.5,
}

# Map our language codes to gTTS-compatible codes
GTTS_LANG_MAP = {
    "zh-CN": "zh",  # gTTS uses 'zh' for Chinese
}


def _get_gtts_lang(lang_code: str) -> str:
    return GTTS_LANG_MAP.get(lang_code, lang_code)


def _estimate_word_timestamps(words: list[str], lang: str) -> list[dict]:
    """
    Generate estimated word timestamps based on speaking rate.
    Returns list of {word, start_ms, end_ms} dicts.
    """
    wps = LANG_WPS.get(lang, 2.5)
    ms_per_word = 1000 / wps

    timestamps = []
    cursor_ms = 300  # Small intro pause

    for word in words:
        # Longer words take a bit more time
        duration = ms_per_word * (0.8 + 0.4 * min(len(word) / 6, 1.0))
        timestamps.append({
            "word": word,
            "start_ms": int(cursor_ms),
            "end_ms": int(cursor_ms + duration),
        })
        cursor_ms += duration + 80  # Small gap between words

    return timestamps


def synthesize_speech(text: str, lang: str = "en") -> dict:
    """
    Convert text to speech using gTTS and return base64 audio + word timestamps.

    Args:
        text: Input text to convert to speech
        lang: Language code (e.g. "en", "hi", "ta")

    Returns:
        dict with:
            audio_base64 (str): Base64-encoded MP3 data
            word_timestamps (list): [{word, start_ms, end_ms}, ...]
            duration_estimate_ms (int): Total estimated audio duration
            lang (str): Language used
    """
    if not text or not text.strip():
        return {
            "audio_base64": "",
            "word_timestamps": [],
            "duration_estimate_ms": 0,
            "lang": lang,
        }

    gtts_lang = _get_gtts_lang(lang)
    words = text.strip().split()

    # Generate TTS audio
    tts = gTTS(text=text.strip(), lang=gtts_lang, slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
    timestamps = _estimate_word_timestamps(words, lang)
    duration_ms = timestamps[-1]["end_ms"] + 300 if timestamps else 0

    return {
        "audio_base64": audio_base64,
        "word_timestamps": timestamps,
        "duration_estimate_ms": duration_ms,
        "lang": lang,
    }
