"""Conversation memory service for follow-up question handling."""
from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque
from backend.config import settings

logger = logging.getLogger("factorymind")


class ConversationMemory:
    """Service for managing conversation history and follow-up context."""
    
    def __init__(self, max_history: int = 10, max_age_hours: int = 24):
        self.max_history = max_history
        self.max_age_hours = max_age_hours
        # Store conversations per user: {user_id: deque of messages}
        self.conversations: Dict[str, deque] = {}
        logger.info(f"ConversationMemory initialized (max_history={max_history}, max_age_hours={max_age_hours})")
    
    def add_message(self, user_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation history."""
        if user_id not in self.conversations:
            self.conversations[user_id] = deque(maxlen=self.max_history)
        
        message = {
            "role": role,  # "user" or "assistant"
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        
        self.conversations[user_id].append(message)
        logger.debug(f"Added {role} message for user {user_id}: {content[:50]}...")
    
    def get_conversation_history(self, user_id: str, max_messages: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation history for a user."""
        if user_id not in self.conversations:
            return []
        
        # Filter by age and return recent messages
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
        recent_messages = [
            msg for msg in self.conversations[user_id]
            if msg["timestamp"] > cutoff_time
        ]
        
        # Return last N messages
        return recent_messages[-max_messages:] if recent_messages else []
    
    def clear_conversation(self, user_id: str):
        """Clear conversation history for a user."""
        if user_id in self.conversations:
            self.conversations[user_id].clear()
            logger.info(f"Cleared conversation history for user {user_id}")
    
    def format_conversation_context(self, user_id: str) -> str:
        """Format conversation history as context for LLM."""
        history = self.get_conversation_history(user_id, max_messages=5)
        
        if not history:
            return ""
        
        context_lines = ["Previous conversation:"]
        for msg in history:
            role_display = "User" if msg["role"] == "user" else "Assistant"
            context_lines.append(f"{role_display}: {msg['content']}")
        
        return "\n".join(context_lines)
    
    def detect_follow_up(self, current_query: str, user_id: str) -> bool:
        """
        Detect if the current query is a follow-up based on conversation history.
        Follow-up indicators: "it", "that", "this", "same", "again", "what about", etc.
        """
        follow_up_indicators = [
            "it", "that", "this", "same", "again", "what about", "how about",
            "also", "additionally", "furthermore", "more", "another", "other"
        ]
        
        query_lower = current_query.lower()
        
        # Check for follow-up indicators
        has_indicator = any(indicator in query_lower for indicator in follow_up_indicators)
        
        # Check if conversation exists
        has_history = len(self.get_conversation_history(user_id, max_messages=1)) > 0
        
        return has_indicator and has_history
    
    def resolve_references(self, current_query: str, user_id: str) -> str:
        """
        Resolve pronoun references in follow-up questions by looking at conversation history.
        """
        if not self.detect_follow_up(current_query, user_id):
            return current_query
        
        history = self.get_conversation_history(user_id, max_messages=3)
        if not history:
            return current_query
        
        # Get the last assistant response for context
        last_assistant_msg = None
        for msg in reversed(history):
            if msg["role"] == "assistant":
                last_assistant_msg = msg["content"]
                break
        
        if not last_assistant_msg:
            return current_query
        
        # Simple reference resolution (can be enhanced with NLP)
        resolved_query = current_query
        
        # Replace "it" with the main topic from last response
        if " it " in resolved_query.lower():
            # Extract key terms from last response (simple heuristic)
            words = last_assistant_msg.split()
            key_terms = [w for w in words if len(w) > 4 and w.isalpha()][:3]
            if key_terms:
                resolved_query = resolved_query.replace(" it ", f" {key_terms[0]} ", 1)
        
        logger.debug(f"Resolved follow-up query: '{current_query}' -> '{resolved_query}'")
        return resolved_query
    
    def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary statistics for a user's conversation."""
        if user_id not in self.conversations:
            return {
                "total_messages": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "last_activity": None
            }
        
        messages = list(self.conversations[user_id])
        user_count = sum(1 for msg in messages if msg["role"] == "user")
        assistant_count = sum(1 for msg in messages if msg["role"] == "assistant")
        last_activity = messages[-1]["timestamp"] if messages else None
        
        return {
            "total_messages": len(messages),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "last_activity": last_activity.isoformat() if last_activity else None
        }


# Singleton instance
conversation_memory = ConversationMemory(
    max_history=10,
    max_age_hours=24
)
