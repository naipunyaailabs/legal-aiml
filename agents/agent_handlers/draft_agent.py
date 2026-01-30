# agent_handlers/draft_agent.py - Draft Agent - Routes to existing drafting service

import httpx
import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent, get_httpx_client

logger = logging.getLogger(__name__)

# Backend service URL (using 127.0.0.1 for faster Windows resolution)
ASK_DRAFT_URL = "http://127.0.0.1:8000"


class DraftAgent(BaseAgent):
    """Draft Agent - Routes to existing ASK/DRAFT drafting service"""
    
    def __init__(self):
        super().__init__("draft", "Draft Agent")
    
    async def process_message(
        self, 
        message: str, 
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process drafting request using tools
        """
        try:
            from .tools import generate_legal_document
            
            context = context or {}
            
            # Extract parameters
            doc_type = context.get("doc_type", "Legal Document")
            style = context.get("style", "Formal Legal")
            
            # Step 1: Use drafting tool
            result = await generate_legal_document(doc_type, message, style)
            
            if "error" not in result:
                return {
                    "success": True,
                    "response": result.get("document", ""),
                    "doc_type": result.get("doc_type", doc_type),
                    "style": result.get("style", style),
                    "word_count": result.get("word_count", 0),
                    "metadata": result.get("metadata", {})
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error")
                }
                    
        except Exception as e:
            logger.error(f"Draft agent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
