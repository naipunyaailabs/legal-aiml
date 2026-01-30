# agent_handlers/review_agent.py - Review Agent - Contract review and risk analysis

import logging
from typing import Dict, Any, Optional
from io import BytesIO

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReviewAgent(BaseAgent):
    """Review Agent - Review contracts and highlight key terms and risks"""
    
    def __init__(self):
        super().__init__("review", "Review Agent")
        self.document_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def process_document(
        self, 
        filename: str, 
        file_content: bytes,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Upload contract document for review
        """
        try:
            from PyPDF2 import PdfReader
            
            ext = filename.lower().split('.')[-1]
            text = ""
            
            if ext == 'pdf':
                reader = PdfReader(BytesIO(file_content))
                text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
            elif ext == 'txt':
                text = file_content.decode('utf-8', errors='ignore')
            elif ext in ['docx', 'doc']:
                try:
                    from docx import Document
                    doc = Document(BytesIO(file_content))
                    text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                except:
                    text = ""
            
            if not text.strip():
                return {
                    "success": False,
                    "error": "Could not extract text from document"
                }
            
            self.document_sessions[session_id] = {
                "filename": filename,
                "text": text
            }
            
            return {
                "success": True,
                "filename": filename,
                "char_count": len(text),
                "preview": text[:500] + "..." if len(text) > 500 else text
            }
            
        except Exception as e:
            logger.error(f"Review agent document error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_message(
        self, 
        message: str, 
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Review contract and analyze risks
        """
        try:
            # Check if document is uploaded
            doc_info = self.document_sessions.get(session_id)
            
            llm = self._initialize_llm()
            
            if doc_info:
                # Document-based review
                doc_text = doc_info.get("text", "")
                # Limit context
                if len(doc_text) > 15000:
                    doc_text = doc_text[:7500] + "\n\n[...document continues...]\n\n" + doc_text[-7500:]
                
                from .tools import risk_analyzer
                
                # Use tool for preliminary risk analysis
                preliminary_risks = await risk_analyzer(doc_text[:5000])
                
                prompt = f"""You are an expert contract reviewer and legal risk analyst.

CONTRACT DOCUMENT: "{doc_info.get('filename', 'contract')}"
---
{doc_text}
---

TOOL-EXTRACTED RISKS:
{preliminary_risks}

USER REQUEST: {message}

Provide a comprehensive contract review including:

1. **EXECUTIVE SUMMARY**:
   - Type of contract/agreement
   - Parties involved
   - Key purpose

2. **KEY TERMS IDENTIFIED**:
   - ⚖️ Important clauses and their implications
   - 💰 Financial terms (payment, penalties, deposits)
   - 📅 Key dates and deadlines
   - 🔄 Renewal/termination provisions

3. **RISK ANALYSIS**:
   🔴 **HIGH RISK** - Critical issues that need immediate attention
   🟡 **MEDIUM RISK** - Concerns that should be addressed
   🟢 **LOW RISK** - Minor issues or standard clauses
   (Incorporate the tool-extracted risks above into your analysis)

4. **MISSING CLAUSES**:
   - Important provisions that are absent
   - Recommended additions

5. **RECOMMENDATIONS**:
   - Suggested changes or negotiations
   - Points to clarify with the other party
   - Protective provisions to add

6. **LEGAL COMPLIANCE**:
   - Whether the contract complies with applicable laws
   - Any clauses that might be unenforceable

Be specific and cite clause numbers/sections from the document where applicable.

YOUR CONTRACT REVIEW:"""
            else:
                # General contract review advice
                prompt = f"""You are an expert contract reviewer and legal risk analyst.

USER QUESTION: {message}

Provide expert guidance on contract review, including:
- Key areas to focus on when reviewing contracts
- Common risks and red flags
- Best practices for contract negotiation
- Legal considerations

YOUR RESPONSE:"""
            
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            sources = []
            if doc_info:
                sources.append({"type": "contract_review", "filename": doc_info.get("filename")})
            sources.append({"type": "risk_analysis", "reference": "AI Contract Review"})
            
            return {
                "success": True,
                "response": answer,
                "document_name": doc_info.get("filename") if doc_info else None,
                "sources": sources,
                "tokens_used": 0
            }
            
        except Exception as e:
            logger.error(f"Review agent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
