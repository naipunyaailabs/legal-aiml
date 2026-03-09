# document_chat.py - Smart Document Chat with Full Doc + Map-Reduce + InLegalBERT
# Supports: Full Document mode (small PDFs), Map-Reduce (large PDFs), InLegalBERT reranking

import os
import re
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from io import BytesIO

# PDF and Document Processing
from PyPDF2 import PdfReader

# Image Processing (for scanned documents)
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# DOCX Processing
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# LangChain for text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Environment
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB (increased for large legal PDFs)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg'}

# Token estimation: ~1 token ≈ 4 characters for English text
FULL_DOC_TOKEN_LIMIT = 5000  # Groq Free tier has very low TPM (6k-12k). 5K ensures we stay under it.
CHARS_PER_TOKEN = 4


# ============================================================
# STRONG GROUNDING PROMPT — Anti-Hallucination Instructions
# ============================================================

GROUNDED_SYSTEM_PROMPT = """You are a legal analyst reviewing an uploaded document.

CRITICAL RULES — YOU MUST FOLLOW THESE:
1. ONLY answer using facts found in the provided document text below.
2. For EVERY claim you make, cite the source with [Page X] when page markers are available.
3. If the information needed to answer the question is NOT in the document, say:
   "This information is not available in the uploaded document."
4. NEVER add legal knowledge from outside this document.
5. NEVER invent or guess case names, section numbers, dates, or party names.
6. If asked to summarize, cover ALL parties, issues, arguments, and orders found in the document.
7. If you are uncertain about something, say "The document is unclear on this point."

RESPONSE FORMAT — YOU MUST USE THESE HEADERS:
- # Legal Issues: [Identify the core legal questions]
- # Legal Provisions: [List all Statutes, Sections, and Orders cited]
- # Court Findings: [Explain the court's detailed reasoning and analysis]
- # Final Holding: [State the ultimate order or decision found at the end]
"""

GROUNDED_DOC_PROMPT = """{system_prompt}

FULL DOCUMENT TEXT:
---
{document_text}
---

USER QUESTION: {query}

GROUNDED LEGAL ANALYSIS (cite [Page X] for every claim):"""

MAP_EXTRACT_PROMPT = """Read the following text excerpt from a legal document.
Extract and label the following components if they are mentioned in this excerpt. 
If a component is NOT mentioned, skip it.

FOR EACH COMPONENT, CITE THE SOURCE AS [Page X].

COMPONENTS TO EXTRACT:
1. Facts: [The underlying events or background]
2. Legal Issue: [The specific conflict or question of law]
3. Arguments: [Contentions made by the Appellant or Respondent]
4. Legal Provisions: [Sections, Acts, or Orders cited]
5. Court Reasoning: [The judge's analysis or observations]
6. Conclusion: [Any interim or final decision made on this page]

If the excerpt contains NO relevant legal information, respond with exactly: NONE

TEXT EXCERPT:
---
{chunk_text}
---

DETAILED EXTRACTION:"""

REDUCE_SYNTHESIZE_PROMPT = """{system_prompt}

Below is the structured extraction from a {page_count}-page legal document "{filename}".
These extractions were captured by scanning EVERY page of the document for Facts, Issues, Provisions, and Reasoning.

STRUCTURED EXTRACTIONS FROM DOCUMENT:
---
{extractions}
---

USER QUESTION: {query}

Using ONLY the structured extractions above, provide a PROFESSIONAL LEGAL ANALYSIS in the following format:
# Legal Issues: 
# Legal Provisions:
# Court Findings: 
# Final Holding: 

(Ensure you cite [Page X] for EVERY single point made.)

PROFESSIONAL LEGAL ANALYSIS:"""


class DocumentProcessor:
    """Process uploaded documents and extract text"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file with page markers"""
        try:
            pdf_file = BytesIO(file_content)
            reader = PdfReader(pdf_file)
            
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            
            full_text = "\n\n".join(text_parts)
            
            # If no text extracted (scanned PDF), try OCR
            if not full_text.strip() and OCR_AVAILABLE:
                logger.info("No text found in PDF, attempting OCR...")
                full_text = self._ocr_pdf(file_content)
            
            return full_text
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""
    
    def _ocr_pdf(self, file_content: bytes) -> str:
        """OCR scanned PDF using pytesseract"""
        try:
            import pypdfium2 as pdfium
            
            pdf = pdfium.PdfDocument(file_content)
            text_parts = []
            
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                # Render page at 300 DPI for better OCR
                bitmap = page.render(scale=300/72)
                pil_image = bitmap.to_pil()
                
                # OCR the image
                page_text = pytesseract.image_to_string(pil_image)
                if page_text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def extract_text_from_image(self, file_content: bytes) -> str:
        """Extract text from image using OCR"""
        if not OCR_AVAILABLE:
            logger.warning("OCR not available - pytesseract not installed")
            return ""
        
        try:
            image = Image.open(BytesIO(file_content))
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Image OCR error: {e}")
            return ""
    
    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            logger.warning("DOCX processing not available - python-docx not installed")
            return ""
        
        try:
            doc = DocxDocument(BytesIO(file_content))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            return ""
    
    def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extract text from TXT file"""
        try:
            # Try UTF-8 first, then fallback to other encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return file_content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return file_content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"TXT extraction error: {e}")
            return ""
    
    def process_document(self, filename: str, file_content: bytes) -> Dict[str, Any]:
        """Process document and return extracted text and metadata"""
        ext = filename.lower().split('.')[-1]
        
        if ext not in ALLOWED_EXTENSIONS:
            return {"error": f"Unsupported file type: {ext}", "text": "", "filename": filename}
        
        # Extract text based on file type
        if ext == 'pdf':
            text = self.extract_text_from_pdf(file_content)
        elif ext in ['png', 'jpg', 'jpeg']:
            text = self.extract_text_from_image(file_content)
        elif ext in ['docx', 'doc']:
            text = self.extract_text_from_docx(file_content)
        elif ext == 'txt':
            text = self.extract_text_from_txt(file_content)
        else:
            text = ""
        
        # Generate document hash for caching
        doc_hash = hashlib.md5(file_content).hexdigest()
        
        # Estimate token count
        estimated_tokens = len(text) // CHARS_PER_TOKEN
        
        # Count pages (from [Page X] markers)
        page_markers = re.findall(r'\[Page \d+\]', text)
        page_count = len(page_markers) if page_markers else max(1, len(text) // 3000)
        
        # Split into chunks for Map-Reduce mode
        chunks = self.text_splitter.split_text(text) if text else []
        
        return {
            "filename": filename,
            "text": text,
            "chunks": chunks,
            "hash": doc_hash,
            "char_count": len(text),
            "chunk_count": len(chunks),
            "file_type": ext,
            "estimated_tokens": estimated_tokens,
            "page_count": page_count
        }


class DocumentChatSession:
    """Manages a chat session with an uploaded document.
    
    Smart routing:
    - Full Document Mode: For PDFs under ~80 pages, sends entire text to LLM
    - Map-Reduce Mode: For large PDFs, reads every chunk, extracts relevant info, synthesizes
    """
    
    def __init__(self, document_data: Dict[str, Any], llm, legal_reranker=None):
        self.document_data = document_data
        self.llm = llm
        self.legal_reranker = legal_reranker  # InLegalBERT for reranking extractions
        self.chat_history = []
        
        # Determine the mode based on document size
        estimated_tokens = document_data.get("estimated_tokens", 0)
        self.use_full_doc = estimated_tokens <= FULL_DOC_TOKEN_LIMIT
        
        mode_name = "Full Document" if self.use_full_doc else "Map-Reduce"
        page_count = document_data.get("page_count", "?")
        logger.info(f"📄 Document Chat Mode: {mode_name} | Pages: ~{page_count} | Tokens: ~{estimated_tokens}")
    
    async def chat(self, user_query: str, legal_knowledge_response: Optional[str] = None) -> Dict[str, Any]:
        """Process a chat query about the document using the appropriate mode"""
        try:
            try:
                if self.use_full_doc:
                    answer = await self._full_document_chat(user_query)
                else:
                    answer = await self._map_reduce_chat(user_query)
            except Exception as e:
                # RUNTIME FALLBACK: If Groq fails (Rate limit), switch to Ollama
                is_groq = "groq" in str(type(self.llm)).lower()
                if is_groq:
                    logger.warning(f"Groq failed at runtime (likely rate limit): {e}. Trying Ollama...")
                    
                    from langchain_ollama import ChatOllama
                    # Try remote IP first, then fallback to localhost
                    base_urls = [os.getenv('OLLAMA_BASE_URL', 'http://192.168.0.29:11434'), 'http://localhost:11434', 'http://127.0.0.1:11434']
                    model = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b')
                    
                    self.llm = None
                    for url in base_urls:
                        try:
                            logger.info(f"Checking Ollama at {url}...")
                            # Create a test instance with short timeout to verify connection
                            test_llm = ChatOllama(model=model, base_url=url, timeout=5)
                            # If this doesn't error, we use it
                            self.llm = test_llm
                            logger.info(f"✅ Connected to Ollama at {url}")
                            break
                        except Exception as conn_err:
                            logger.debug(f"Could not connect to {url}: {conn_err}")
                            continue
                    
                    if not self.llm:
                        logger.error("❌ All Ollama connection attempts failed.")
                        raise e # Raise original Groq error if Ollama is also down
                    
                    # Retry once with Ollama
                    if self.use_full_doc:
                        answer = await self._full_document_chat(user_query)
                    else:
                        answer = await self._map_reduce_chat(user_query)
                else:
                    raise e
            
            # Store in history
            self.chat_history.append({"role": "user", "content": user_query})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            return {
                "answer": answer,
                "document_name": self.document_data.get("filename", "Unknown"),
                "sources": [{
                    "type": "uploaded_document",
                    "filename": self.document_data.get("filename"),
                    "mode": "full_document" if self.use_full_doc else "map_reduce",
                    "pages": self.document_data.get("page_count", 0)
                }]
            }
            
        except Exception as e:
            logger.error(f"Document chat error: {e}")
            return {
                "answer": f"I encountered an error processing your question: {str(e)}",
                "error": str(e)
            }
    
    async def _full_document_chat(self, query: str) -> str:
        """FULL DOCUMENT MODE: Send entire document text to LLM.
        
        Used for documents under ~80 pages.
        The LLM reads everything — nothing is missed.
        """
        logger.info("📖 Using Full Document Mode — sending entire text to LLM")
        
        prompt = GROUNDED_DOC_PROMPT.format(
            system_prompt=GROUNDED_SYSTEM_PROMPT,
            document_text=self.document_data.get("text", ""),
            query=query
        )
        
        response = await self.llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    async def _map_reduce_chat(self, query: str) -> str:
        """MAP-REDUCE MODE: Read every chunk, extract relevant info, synthesize.
        
        Used for documents over ~80 pages.
        Stage 1 (Map): LLM reads each chunk and extracts relevant info
        Stage 2 (Rerank): InLegalBERT ranks the extractions by relevance
        Stage 3 (Reduce): LLM synthesizes a final answer from best extractions
        """
        chunks = self.document_data.get("chunks", [])
        if not chunks:
            return "Could not process the document. No text chunks available."
        
        logger.info(f"🗺️ Map-Reduce Mode: Processing {len(chunks)} chunks...")
        
        # ─── STAGE 1: MAP — Read every chunk with LLM ───
        logger.info(f"📤 Stage 1 (Map): Sending {len(chunks)} chunks to LLM for extraction...")
        
        extraction_tasks = []
        for i, chunk in enumerate(chunks):
            prompt = MAP_EXTRACT_PROMPT.format(query=query, chunk_text=chunk)
            extraction_tasks.append(self._extract_from_chunk(prompt, i, len(chunks)))
        
        # Run extractions concurrently (batch of 5 to avoid rate limits)
        extractions = []
        batch_size = 5
        for i in range(0, len(extraction_tasks), batch_size):
            batch = extraction_tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, str) and result.strip() and result.strip().upper() != "NONE":
                    extractions.append(result)
        
        logger.info(f"📥 Stage 1 complete: {len(extractions)} relevant extractions from {len(chunks)} chunks")
        
        if not extractions:
            return "After reading the entire document, I could not find information relevant to your question. The document may not contain the answer you're looking for."
        
        # ─── STAGE 2: RERANK with InLegalBERT ───
        if self.legal_reranker and self.legal_reranker.available and len(extractions) > 10:
            logger.info(f"🧠 Stage 2 (Rerank): InLegalBERT scoring {len(extractions)} extractions...")
            scored = []
            for ext in extractions:
                score = self.legal_reranker.score_pair(query, ext)
                scored.append((ext, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            # Keep top 15 extractions
            extractions = [ext for ext, score in scored[:15]]
            logger.info(f"🧠 Stage 2 complete: Kept top {len(extractions)} extractions")
        
        # ─── STAGE 3: REDUCE — Synthesize final answer ───
        logger.info("📝 Stage 3 (Reduce): Synthesizing final answer...")
        
        combined_extractions = "\n\n---\n\n".join(extractions)
        
        prompt = REDUCE_SYNTHESIZE_PROMPT.format(
            system_prompt=GROUNDED_SYSTEM_PROMPT,
            page_count=self.document_data.get("page_count", "unknown"),
            filename=self.document_data.get("filename", "document"),
            extractions=combined_extractions,
            query=query
        )
        
        response = await self.llm.ainvoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        logger.info("✅ Map-Reduce complete — full document analyzed")
        return answer
    
    async def _extract_from_chunk(self, prompt: str, chunk_idx: int, total_chunks: int) -> str:
        """Extract relevant info from a single chunk (used in Map stage)"""
        try:
            response = await self.llm.ainvoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            if (chunk_idx + 1) % 10 == 0:
                logger.info(f"   📄 Processed chunk {chunk_idx + 1}/{total_chunks}")
            return result
        except Exception as e:
            logger.debug(f"Chunk {chunk_idx} extraction error: {e}")
            return "NONE"


class DocumentChatManager:
    """Manages document chat sessions with LLM integration"""
    
    def __init__(self, chatbot_manager=None):
        """
        Initialize with optional chatbot_manager for legal knowledge integration
        
        Args:
            chatbot_manager: The main ChatbotManager instance for accessing legal knowledge & reranker
        """
        self.processor = DocumentProcessor()
        self.chatbot_manager = chatbot_manager
        self.active_sessions: Dict[str, DocumentChatSession] = {}
        
        # Get InLegalBERT reranker from chatbot_manager if available
        self.legal_reranker = None
        if chatbot_manager and hasattr(chatbot_manager, 'legal_reranker'):
            self.legal_reranker = chatbot_manager.legal_reranker
        
        # Initialize LLM (same as main chatbot)
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM using Groq as primary and Ollama as fallback."""
        groq_api_key = os.getenv('GROQ_API_KEY')
        ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://192.168.0.29:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b')

        # 1. Try Groq (Primary - High Speed, 128K context)
        try:
            if groq_api_key:
                from langchain_groq import ChatGroq
                # llama-3.1-8b-instant has MUCH higher TPM limits than the 70B model
                llm = ChatGroq(
                    model_name="llama-3.1-8b-instant",
                    temperature=0.0,
                    groq_api_key=groq_api_key
                )
                logger.info("📖 Document Chat LLM: Groq (llama-3.1-8b-instant) 🚀")
                return llm
        except Exception as e:
            logger.warning(f"Groq not available: {e}")

        # 2. Try Ollama (Fallback - Local)
        try:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=ollama_model,
                temperature=0.0,
                base_url=ollama_base_url,
                timeout=300
            )
            logger.info(f"📖 Document Chat LLM: Ollama Fallback ({ollama_model})")
            return llm
        except Exception as e:
            logger.error(f"All LLMs failed: {e}")
        
        raise ValueError("No LLM available. Please check your Groq API key or Ollama server.")
    
    def upload_document(self, filename: str, file_content: bytes, session_id: str) -> Dict[str, Any]:
        """
        Upload and process a document for chat
        
        Args:
            filename: Name of the uploaded file
            file_content: Raw bytes of the file
            session_id: Unique session identifier
        
        Returns:
            Dict with processing result
        """
        # Validate file size
        if len(file_content) > MAX_FILE_SIZE:
            return {
                "success": False,
                "error": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            }
        
        # Process the document
        doc_data = self.processor.process_document(filename, file_content)
        
        if "error" in doc_data:
            return {
                "success": False,
                "error": doc_data["error"]
            }
        
        if not doc_data.get("text"):
            return {
                "success": False,
                "error": "Could not extract text from document. Please try a different file."
            }
        
        # Determine mode for logging
        mode = "Full Document" if doc_data["estimated_tokens"] <= FULL_DOC_TOKEN_LIMIT else "Map-Reduce"
        
        # Create chat session with smart routing
        chat_session = DocumentChatSession(
            doc_data, 
            self.llm,
            legal_reranker=self.legal_reranker
        )
        self.active_sessions[session_id] = chat_session
        
        return {
            "success": True,
            "session_id": session_id,
            "filename": filename,
            "char_count": doc_data["char_count"],
            "chunk_count": doc_data["chunk_count"],
            "file_type": doc_data["file_type"],
            "preview": doc_data["text"][:500] + "..." if len(doc_data["text"]) > 500 else doc_data["text"],
            "mode": mode,
            "estimated_tokens": doc_data["estimated_tokens"],
            "page_count": doc_data["page_count"]
        }
    
    async def chat_with_document(
        self, 
        session_id: str, 
        query: str,
        include_legal_knowledge: bool = True
    ) -> Dict[str, Any]:
        """
        Chat about an uploaded document
        
        Args:
            session_id: The document session ID
            query: User's question
            include_legal_knowledge: Whether to include responses from legal knowledge base
        
        Returns:
            Dict with answer and metadata
        """
        if session_id not in self.active_sessions:
            return {
                "success": False,
                "error": "No document uploaded for this session. Please upload a document first."
            }
        
        session = self.active_sessions[session_id]
        
        # Chat with the document using smart routing (Full Doc or Map-Reduce)
        result = await session.chat(query)
        result["success"] = True
        result["session_id"] = session_id
        
        return result
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a document chat session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            "filename": session.document_data.get("filename"),
            "char_count": session.document_data.get("char_count"),
            "page_count": session.document_data.get("page_count"),
            "mode": "Full Document" if session.use_full_doc else "Map-Reduce",
            "chat_history_length": len(session.chat_history)
        }
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a document chat session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False


# Singleton instance
_document_chat_manager: Optional[DocumentChatManager] = None

def get_document_chat_manager(chatbot_manager=None) -> DocumentChatManager:
    """Get or create the document chat manager singleton"""
    global _document_chat_manager
    if _document_chat_manager is None:
        _document_chat_manager = DocumentChatManager(chatbot_manager)
    return _document_chat_manager
