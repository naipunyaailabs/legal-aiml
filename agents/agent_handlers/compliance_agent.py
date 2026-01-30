# agent_handlers/compliance_agent.py - Compliance Agent - Regulatory compliance checking

import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ComplianceAgent(BaseAgent):
    """Compliance Agent - Check regulatory compliance and identify violations"""
    
    def __init__(self):
        super().__init__("compliance", "Compliance Agent")
    
    async def process_message(
        self, 
        message: str, 
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process compliance check using tools and LLM
        """
        try:
            from .tools import compliance_check, search_indian_statutes
            
            # Step 1: Call tools
            compliance_data = await compliance_check(message)
            regulations = await search_indian_statutes("compliance regulations")
            
            # Step 2: Initialize LLM and construct prompt
            llm = self._initialize_llm()
            
            prompt = f"""You are a Legal Compliance Expert specializing in Indian Regulatory frameworks.
            
TOOL CONTEXT:
Compliance Checklist: {compliance_data}
Relevant Regulations: {regulations}

USER QUERY: {message}

Please provide a comprehensive compliance analysis:
1. **Regulatory Framework**: Which Indian laws apply (Companies Act, GST, SEBI, Labour laws, etc.)?
2. **Current Compliance Status**: Based on the query, what is the status?
3. **Required Actions**: Specific steps to ensure 100% compliance.
4. **Potential Penalties**: Risks of non-compliance.
5. **Next Review Date**: When should this be re-evaluated?

Format with professional legal clarity. Use tables or lists where appropriate.

YOUR COMPLIANCE RESPONSE:"""
            
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "success": True,
                "response": answer,
                "sources": [
                    {"type": "compliance_tool", "status": "completed"},
                    {"type": "regulatory_database", "reference": "MCA/SEBI/Labour"}
                ]
            }
            
        except Exception as e:
            logger.error(f"Compliance agent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
