"""Vision service abstraction for image analysis and description generation."""
from __future__ import annotations

import logging
import base64
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any
from backend.config import settings

logger = logging.getLogger("factorymind")


class VisionProviderError(Exception):
    """Raised when a vision provider API call fails."""
    pass


class VisionService:
    """Vision service for image analysis and description generation."""
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        logger.info(f"VisionService initialized with provider: {self.provider}")
    
    def describe_image(self, image_path: str, prompt: str = "Describe this technical diagram in detail.") -> str:
        """
        Generate a description for an image using the configured vision provider.
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt for image description
            
        Returns:
            Image description text
        """
        try:
            if self.provider == "groq":
                return self._describe_with_groq(image_path, prompt)
            elif self.provider in ("openai", "openai_compatible"):
                return self._describe_with_openai(image_path, prompt)
            elif self.provider == "anthropic":
                return self._describe_with_anthropic(image_path, prompt)
            else:
                logger.warning(f"Vision not supported for provider: {self.provider}")
                return self._extractive_fallback(image_path)
        except VisionProviderError as exc:
            logger.error(f"Vision provider failed: {exc}")
            return self._extractive_fallback(image_path)
    
    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise VisionProviderError(f"Image encoding failed: {str(e)}")
    
    def _describe_with_groq(self, image_path: str, prompt: str) -> str:
        """Describe image using Groq's vision capabilities."""
        try:
            # Groq doesn't currently support vision, use fallback
            logger.warning("Groq doesn't support vision API yet, using fallback")
            return self._extractive_fallback(image_path)
        except Exception as e:
            logger.error(f"Groq vision failed: {e}")
            raise VisionProviderError(f"Groq vision error: {str(e)}")
    
    def _describe_with_openai(self, image_path: str, prompt: str) -> str:
        """Describe image using OpenAI's GPT-4 Vision API."""
        try:
            base64_image = self._encode_image_base64(image_path)
            api_key = settings.OPENAI_API_KEY
            
            if not api_key:
                raise VisionProviderError("OPENAI_API_KEY not set")
            
            url = "https://api.openai.com/v1/chat/completions"
            
            body = json.dumps({
                "model": "gpt-4o-mini",  # Vision-capable model
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }).encode("utf-8")
            
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            
            return payload["choices"][0]["message"]["content"].strip()
            
        except urllib.error.HTTPError as http_err:
            try:
                err_body = http_err.read().decode("utf-8")
                err_json = json.loads(err_body)
                error_msg = err_json.get("error", {}).get("message", str(http_err))
            except Exception:
                error_msg = f"HTTP {http_err.code}"
            logger.error(f"OpenAI vision HTTP error: {error_msg}")
            raise VisionProviderError(f"OpenAI vision error: {error_msg}")
        except Exception as e:
            logger.error(f"OpenAI vision failed: {e}")
            raise VisionProviderError(f"OpenAI vision error: {str(e)}")
    
    def _describe_with_anthropic(self, image_path: str, prompt: str) -> str:
        """Describe image using Anthropic's Claude Vision API."""
        try:
            base64_image = self._encode_image_base64(image_path)
            api_key = settings.ANTHROPIC_API_KEY
            
            if not api_key:
                raise VisionProviderError("ANTHROPIC_API_KEY not set")
            
            url = "https://api.anthropic.com/v1/messages"
            
            body = json.dumps({
                "model": "claude-3-5-sonnet-latest",
                "max_tokens": 300,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            }).encode("utf-8")
            
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            
            return payload["content"][0]["text"].strip()
            
        except urllib.error.HTTPError as http_err:
            try:
                err_body = http_err.read().decode("utf-8")
                err_json = json.loads(err_body)
                error_msg = err_json.get("error", {}).get("message", str(http_err))
            except Exception:
                error_msg = f"HTTP {http_err.code}"
            logger.error(f"Anthropic vision HTTP error: {error_msg}")
            raise VisionProviderError(f"Anthropic vision error: {error_msg}")
        except Exception as e:
            logger.error(f"Anthropic vision failed: {e}")
            raise VisionProviderError(f"Anthropic vision error: {str(e)}")
    
    def _extractive_fallback(self, image_path: str) -> str:
        """Fallback when vision is unavailable - return basic metadata."""
        import os
        filename = os.path.basename(image_path)
        return f"[Image: {filename} - Vision description unavailable. Using fallback metadata.]"


# Singleton instance
vision_service = VisionService()
