# agent_handlers/research_agent.py - Research Agent - Legal research and case law

import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """Research Agent - Deep dive into case law and legal precedents"""
    
    def __init__(self):
        super().__init__("research", "Research Agent")
    
    async def process_message(
        self, 
        message: str, 
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process legal research request using tools and LLM
        """
        try:
            from .tools import search_indian_statutes, case_law_lookup
            
            # Step 1: Use tools to get real data
            statutes = await search_indian_statutes(message)
            cases = await case_law_lookup(message)
            
            # Step 2: Initialize LLM and construct enhanced prompt
            llm = self._initialize_llm()
            
            tool_context = f"RELEVANT STATUTES:\n{statutes}\n\nRELEVANT CASES:\n{cases}"
            
            prompt = f"""You are a legal research expert specializing in Indian law.
            
Use the following tool-provided data to ground your research:
{tool_context}

RESEARCH QUERY: {message}

Please provide a detailed research report including:
1. **Relevant Case Law**: Detail the cases provided above and add others you know that are relevant.
2. **Legal Precedents**: Key established principles.
3. **Statutory Provisions**: Elaborate on the sections of the Acts mentioned.
4. **Legal Analysis**: How these apply to the user's query.

Format your response clearly with sections and bullet points.
Cite cases in proper Indian legal format.

YOUR RESEARCH RESPONSE:"""
            
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "success": True,
                "response": answer,
                "sources": [
                    {"type": "statute_lookup", "count": len(statutes)},
                    {"type": "case_law_search", "count": len(cases)},
                    {"type": "ai_synthesis", "reference": "Research Agent"}
                ],
                "tokens_used": 0
            }
            
        except Exception as e:
            logger.error(f"Research agent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
