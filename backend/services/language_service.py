"""Language detection and multilingual support service for FactoryMind AI."""
from __future__ import annotations

import logging
from typing import Optional
from backend.config import settings

logger = logging.getLogger("factorymind")


class LanguageService:
    """
    Service for language detection and multilingual support.
    Supports English, Hindi, Telugu, Tamil, Kannada, Malayalam and major global languages.
    Retrieves in English, synthesizes in user's language.
    """
    
    LANGUAGE_NAMES = {
        "en": "English",
        "hi": "Hindi",
        "te": "Telugu",
        "ta": "Tamil",
        "kn": "Kannada",
        "ml": "Malayalam",
        "es": "Spanish",
        "fr": "French",
        "de": "German"
    }

    UNICODE_RANGES = {
        "hi": (0x0900, 0x097F),  # Devanagari
        "te": (0x0C00, 0x0C7F),  # Telugu
        "ta": (0x0B80, 0x0BFF),  # Tamil
        "kn": (0x0C80, 0x0CFF),  # Kannada
        "ml": (0x0D00, 0x0D7F),  # Malayalam
    }
    
    def __init__(self):
        self.default_language = "en"
        logger.info("LanguageService initialized with multilingual support (EN, HI, TE, TA, KN, ML).")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text using Unicode script detection and pattern matching.
        """
        if not text or len(text.strip()) < 2:
            return self.default_language
        
        # 1. Unicode Script Range Check (High accuracy for Indian scripts)
        for char in text:
            code = ord(char)
            for lang, (start, end) in self.UNICODE_RANGES.items():
                if start <= code <= end:
                    logger.info(f"Script-detected language: {self.LANGUAGE_NAMES[lang]} ({lang})")
                    return lang
        
        # 2. Text keyword detection for transliterated text
        text_lower = text.lower()
        if any(w in text_lower for w in ["kya", "kaise", "batao", "telugu", "kannada", "tamil"]):
            if "batao" in text_lower or "kya" in text_lower:
                return "hi"

        return self.default_language
    
    def get_system_prompt_language(self, detected_lang: str) -> str:
        """Get language-specific system prompt instructions for final LLM synthesis."""
        language_instructions = {
            "en": "Respond strictly in English.",
            "hi": "Respond in Hindi (हिंदी में उत्तर दें). Translate technical findings clearly into Hindi.",
            "te": "Respond in Telugu (తెలుగులో సమాధానం ఇవ్వండి). Translate technical findings clearly into Telugu.",
            "ta": "Respond in Tamil (தமிழில் பதிலளிக்கவும்). Translate technical findings clearly into Tamil.",
            "kn": "Respond in Kannada (ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ). Translate technical findings clearly into Kannada.",
            "ml": "Respond in Malayalam (മലയാളത്തിൽ മറുപടി നൽകുക). Translate technical findings clearly into Malayalam.",
            "es": "Responde en español.",
            "fr": "Répondez en français.",
            "de": "Antworten Sie auf Deutsch."
        }
        
        return language_instructions.get(detected_lang, language_instructions["en"])
    
    def get_response_language(self, query: str) -> str:
        detected = self.detect_language(query)
        logger.info(f"Query language detected: {self.LANGUAGE_NAMES.get(detected, detected)}")
        return detected


language_service = LanguageService()
