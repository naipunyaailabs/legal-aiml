# agent_handlers/interact_agent.py - Interact Agent - Routes to document analysis service

import os
import httpx
import logging
from typing import Dict, Any, Optional
from io import BytesIO

from .base_agent import BaseAgent, get_httpx_client

logger = logging.getLogger(__name__)

# Backend service URL (mapped via environment for Docker support)
INTERACT_URL = os.getenv("INTERACT_API_URL", "http://127.0.0.1:8001")


class InteractAgent(BaseAgent):
    """Interact Agent - Routes to existing document analysis service"""
    
    def __init__(self):
        super().__init__("interact", "Interact Agent")
        self.document_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def process_document(
        self, 
        filename: str, 
        file_content: bytes,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Upload document to interact service
        """
        try:
            # Upload to document analysis service
            files = {"file": (filename, file_content)}
            
            # Call existing interact API using shared client
            response = await get_httpx_client().post(
                f"{INTERACT_URL}/upload",
                files=files
            )
                
            if response.status_code == 200:
                data = response.json()
                
                # Store document info for session
                self.document_sessions[session_id] = {
                    "filename": filename,
                    "text": data.get("text", ""),
                    "document_id": data.get("document_id")
                }
                
                return {
                    "success": True,
                    "document_id": data.get("document_id"),
                    "filename": filename,
                    "char_count": len(data.get("text", "")),
                    "preview": data.get("text", "")[:500] + "..." if len(data.get("text", "")) > 500 else data.get("text", "")
                }
            else:
                # Fallback: process locally
                return await self._process_document_locally(filename, file_content, session_id)
                    
        except Exception as e:
            logger.warning(f"Interact service unavailable, processing locally: {e}")
            return await self._process_document_locally(filename, file_content, session_id)
    
    async def _process_document_locally(
        self, 
        filename: str, 
        file_content: bytes,
        session_id: str
    ) -> Dict[str, Any]:
        """Fallback: process document locally if interact service unavailable"""
        try:
            from PyPDF2 import PdfReader
            
            ext = filename.lower().split('.')[-1]
            text = ""
            
            if ext == 'pdf':
                reader = PdfReader(BytesIO(file_content))
                text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
            elif ext == 'txt':
                text = file_content.decode('utf-8', errors='ignore')
            
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
            logger.error(f"Local document processing error: {e}")
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
        Process message about uploaded document using tools
        """
        try:
            from .tools import legal_doc_search
            
            # Check if document is uploaded for this session
            doc_info = self.document_sessions.get(session_id)
            
            # Step 1: Use document search tool for specific context
            search_result = await legal_doc_search(message, "Document Only")
            
            # Step 2: Initialize LLM
            llm = self._initialize_llm()
            
            # Build prompt with document context
            doc_text = doc_info.get("text", "") if doc_info else ""
            # Limit context to avoid token overflow
            if len(doc_text) > 15000:
                doc_text = doc_text[:7500] + "\n\n[...document continues...]\n\n" + doc_text[-7500:]
            
            prompt = f"""You are a legal document analysis assistant.
            
UPLOADED DOCUMENT: "{doc_info.get('filename', 'Unknown') if doc_info else 'None'}"
DOC CONTENT: {doc_text}

INTERNAL SEARCH RESULTS:
{search_result.get('answer', 'No matches found.')}

USER QUESTION: {message}

Provide a detailed, helpful answer based on the document and search results.

YOUR ANALYSIS:"""
            
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "success": True,
                "response": answer,
                "document_name": doc_info.get("filename") if doc_info else "Search Database",
                "sources": [
                    {"type": "uploaded_document", "active": doc_info is not None},
                    {"type": "vector_search", "found": "answer" in search_result}
                ]
            }
            
        except Exception as e:
            logger.error(f"Interact agent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
