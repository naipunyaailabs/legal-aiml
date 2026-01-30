# agent_handlers/ask_agent.py - Ask Agent - Routes to existing chatbot

import os
import httpx
import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent, get_httpx_client

logger = logging.getLogger(__name__)

# Backend service URL (mapped via environment for Docker support)
ASK_DRAFT_URL = os.getenv("CHATBOT_API_URL", "http://127.0.0.1:8000")


class AskAgent(BaseAgent):
    """Ask Agent - Routes to existing ASK/DRAFT chatbot service"""
    
    def __init__(self):
        super().__init__("ask", "Ask Agent")
    
    async def process_message(
        self, 
        message: str, 
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process message by searching legal docs and statutes
        """
        try:
            from .tools import legal_doc_search, search_indian_statutes
            
            context = context or {}
            
            # Step 1: Use internal search tool (this maps to our backend)
            search_result = await legal_doc_search(message, context.get("chatbot_mode", "Hybrid (Smart)"))
            
            # Step 2: Use broader statute lookup for Indian law context
            statutes = await search_indian_statutes(message)
            
            # Step 3: Use LLM to synthesize
            llm = self._initialize_llm()
            
            prompt = f"""You are an advanced Legal AI Assistant.
            
INTERNAL SEARCH RESULTS:
{search_result.get('answer', 'No specific document matches found.')}

RELEVANT INDIAN STATUTES:
{statutes}

USER QUERY: {message}

Synthesize a professional legal answer. Always cite your sources (Document Search or Statute Lookup).
If the query is a general legal greeting, just be helpful and professional.

YOUR RESPONSE:"""
            
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "success": True,
                "response": answer,
                "sources": [
                    {"type": "document_search", "found": "answer" in search_result},
                    {"type": "statute_lookup", "count": len(statutes)}
                ]
            }
                    
        except Exception as e:
            logger.error(f"Ask agent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
