"""Language detection and multilingual support service."""
from __future__ import annotations

import logging
from typing import Optional
from backend.config import settings

logger = logging.getLogger("factorymind")


class LanguageService:
    """Service for language detection and multilingual support."""
    
    # Common language patterns
    LANGUAGE_PATTERNS = {
        "en": ["the", "is", "at", "which", "on", "and", "a", "an", "in", "to"],
        "es": ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se"],
        "fr": ["le", "la", "de", "et", "à", "un", "il", "être", "et", "en"],
        "de": ["der", "die", "das", "und", "in", "den", "von", "zu", "das", "mit"],
        "zh": ["的", "是", "在", "和", "有", "不", "这", "我", "他", "她"],
        "ja": ["の", "は", "を", "に", "が", "で", "と", "た", "です", "ます"],
        "ko": ["의", "는", "을", "에", "가", "에서", "과", "를", "입니다", "입니다"],
        "ru": ["и", "в", "не", "на", "я", "с", "что", "а", "по", "это"],
        "pt": ["o", "a", "de", "e", "do", "da", "em", "um", "para", "é"],
        "it": ["il", "la", "di", "e", "in", "un", "è", "per", "a", "da"]
    }
    
    LANGUAGE_NAMES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "pt": "Portuguese",
        "it": "Italian"
    }
    
    def __init__(self):
        self.default_language = "en"
        logger.info("LanguageService initialized with multilingual support")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text using pattern matching.
        Returns ISO 639-1 language code.
        """
        if not text or len(text.strip()) < 3:
            return self.default_language
        
        text_lower = text.lower()
        scores = {}
        
        # Score each language based on pattern matches
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                scores[lang] = score
        
        # Return language with highest score, or default
        if scores:
            detected_lang = max(scores, key=scores.get)
            logger.debug(f"Detected language: {detected_lang} (score: {scores[detected_lang]})")
            return detected_lang
        
        return self.default_language
    
    def get_system_prompt_language(self, detected_lang: str) -> str:
        """Get language-specific system prompt instructions."""
        language_instructions = {
            "en": "Respond in English.",
            "es": "Responde en español.",
            "fr": "Répondez en français.",
            "de": "Antworten Sie auf Deutsch.",
            "zh": "用中文回答。",
            "ja": "日本語で回答してください。",
            "ko": "한국어로 답변하십시오.",
            "ru": "Отвечайте на русском языке.",
            "pt": "Responda em português.",
            "it": "Rispondi in italiano."
        }
        
        return language_instructions.get(detected_lang, language_instructions["en"])
    
    def should_translate_manuals(self, detected_lang: str) -> bool:
        """
        Determine if manuals should be translated based on detected language.
        Returns False to avoid translating manuals as per requirements.
        """
        # Never translate manuals - always use original language
        return False
    
    def get_response_language(self, query: str) -> str:
        """
        Get the language to use for the response based on the query language.
        """
        detected = self.detect_language(query)
        logger.info(f"Query language detected: {self.LANGUAGE_NAMES.get(detected, detected)}")
        return detected


# Singleton instance
language_service = LanguageService()
