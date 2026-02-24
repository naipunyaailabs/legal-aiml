# chatbot.py - Advanced Chatbot Manager with Accurate Token Counting, Enhanced Security, and Custom LLM Support

import re
import os

# CRITICAL: Set NO_PROXY before importing langchain_ollama (which uses httpx)
os.environ['NO_PROXY'] = '192.168.0.56'

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import tiktoken  # For accurate token counting
from lightrag_manager import get_lightrag_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# FIXED: Realistic Security and Cost Controls
MAX_QUERY_LENGTH = 2000
MAX_RESPONSE_LENGTH = 15000
MAX_TOKENS_PER_SESSION = 75000  # Reduced to realistic limit
RATE_LIMIT_SECONDS = 0.5
BLOCKED_PATTERNS = [
    r'<script.*?>.*?</script>',
    r'javascript:',
    r'data:text/html',
    r'<iframe.*?>.*?</iframe>'
]

# Enhanced PII patterns
PII_PATTERNS = {
    'ssn': r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
    'credit_card': r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
    'phone': r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
}

# Content filtering
PROFANITY_LIST = [
    'badword1', 'badword2', 'inappropriate1', 'inappropriate2'
    # Add actual profanity list for production
]

SENSITIVE_TOPICS = [
    'password', 'api_key', 'secret', 'token', 'private_key',
    'social_security', 'bank_account', 'credit_card'
]

def rate_limit(min_interval=RATE_LIMIT_SECONDS):
    """Rate limiting decorator"""
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

class TokenCounter:
    """FIXED: Accurate token counting using tiktoken"""
    
    def __init__(self):
        try:
            # Try to get the correct tokenizer for the model
            self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 tokenizer
        except:
            # Fallback to approximate counting
            self.encoding = None
            logger.warning("tiktoken not available, using approximate token counting")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens accurately"""
        if not text:
            return 0
            
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except:
                pass
        
        # Fallback: approximate token counting
        # For English text, roughly 1 token = 0.75 words
        words = len(text.split())
        return int(words * 1.33)
    
    def estimate_tokens_from_messages(self, messages: List[Dict]) -> int:
        """Estimate tokens for a conversation"""
        total = 0
        for message in messages:
            total += self.count_tokens(message.get('content', ''))
            total += 4  # Overhead per message
        return total + 3  # Conversation overhead

class ContentFilter:
    """Enhanced content filtering with multiple security layers"""
    
    def __init__(self):
        self.pii_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in PII_PATTERNS.items()}
        self.blocked_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in BLOCKED_PATTERNS]
        self.profanity_pattern = re.compile('|'.join(PROFANITY_LIST), re.IGNORECASE)
        self.sensitive_pattern = re.compile('|'.join(SENSITIVE_TOPICS), re.IGNORECASE)
    
    def scan_for_pii(self, text: str) -> Tuple[bool, List[str], str]:
        """Scan text for PII and return cleaned version"""
        found_pii = []
        cleaned_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                found_pii.append(pii_type)
                cleaned_text = pattern.sub(f'[{pii_type.upper()}_REDACTED]', cleaned_text)
        
        return bool(found_pii), found_pii, cleaned_text
    
    def check_content_safety(self, text: str) -> Tuple[bool, List[str]]:
        """Check for various content safety issues"""
        issues = []
        
        # Check for blocked patterns (XSS, injection, etc.)
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                issues.append("malicious_content")
                break
        
        # Check for profanity
        if self.profanity_pattern.search(text):
            issues.append("profanity")
        
        # Check for sensitive topics
        if self.sensitive_pattern.search(text):
            issues.append("sensitive_content")
        
        # Check for excessive length
        if len(text) > MAX_QUERY_LENGTH:
            issues.append("excessive_length")
        
        return bool(issues), issues
    
    def filter_response(self, response: str) -> Tuple[str, bool, List[str]]:
        """Filter and clean response content"""
        issues = []
        
        # Check for PII in response
        has_pii, pii_types, cleaned_response = self.scan_for_pii(response)
        if has_pii:
            issues.extend(pii_types)
        
        # Check response safety
        has_safety_issues, safety_issues = self.check_content_safety(cleaned_response)
        if has_safety_issues:
            issues.extend(safety_issues)
        
        # Truncate if too long
        if len(cleaned_response) > MAX_RESPONSE_LENGTH:
            cleaned_response = cleaned_response[:MAX_RESPONSE_LENGTH] + "... [Response truncated for safety]"
            issues.append("response_truncated")
        
        return cleaned_response, bool(issues), issues

class AdvancedRetriever:
    """Advanced retrieval with hybrid search and reranking"""
    
    def __init__(self, vector_store: QdrantVectorStore, embeddings, k: int = 8, score_threshold: float = 0.4):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.k = k
        self.score_threshold = score_threshold
        self.token_counter = TokenCounter()
    
    def hybrid_search(self, query: str, k: int = None) -> List[Document]:
        """Perform hybrid search combining semantic and keyword search"""
        k = k or self.k
        
        try:
            # Semantic search
            semantic_results = self.vector_store.similarity_search_with_score(
                query, 
                k=k
            )
            
            # Filter by score threshold
            filtered_results = [
                doc for doc, score in semantic_results 
                if score >= self.score_threshold
            ]
            
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            # Fallback to basic search
            return self.vector_store.similarity_search(query, k=k)
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Enhanced reranking based on relevance and fact-density"""
        if not documents:
            return []
        
        try:
            query_terms = set(query.lower().split())
            
            def calculate_score(doc: Document) -> float:
                content = doc.page_content.lower()
                # 1. Semantic overlap
                doc_terms = set(content.split())
                overlap = len(query_terms.intersection(doc_terms))
                overlap_score = overlap / len(query_terms) if query_terms else 0
                
                # 2. Fact density (Penalize chunks that are mostly numbers/garbage)
                text_only = re.sub(r'[^a-zA-Z\s]', '', content)
                fact_score = len(text_only) / len(content) if len(content) > 0 else 0
                
                # 3. Legal weight (Reward key terms & Adversarial findings)
                legal_keywords = ["section", "act", "article", "court", "order", "judgment", "petitioner", "respondent"]
                adversarial_keywords = ["held", "overruled", "contended", "alleged", "finding", "decree", "rejected", "affirmed"]
                legal_weight = (sum(1 for kw in legal_keywords if kw in content) * 0.1) + \
                               (sum(1 for kw in adversarial_keywords if kw in content) * 0.15)
                
                # 4. Length reward (Favor substantial chunks over tiny ones)
                length_reward = min(len(content) / 2000, 0.2)
                
                return (overlap_score * 0.5) + (fact_score * 0.2) + legal_weight + length_reward
            
            # Sort by enhanced score
            scored_docs = [(doc, calculate_score(doc)) for doc in documents]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in scored_docs]
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return documents

class ChatbotManager:
    """Enhanced chatbot with accurate token counting and security"""
    
    # FIXED: Use persistent collection name matching vectors.py
    DEFAULT_COLLECTION_NAME = "Legal_documents"
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = None,
        llm_model: str = "llama-3.3-70b-versatile",
        llm_temperature: float = 0.3,
        max_tokens: int = 4000,
        qdrant_url: str = None,
        collection_name: str = None,
        retrieval_k: int = 8,
        score_threshold: float = 0.4,
        use_custom_llm: bool = False,  # NEW: Toggle for custom LLM
        custom_llm_url: str = None,  # NEW: Custom LLM endpoint  
        custom_llm_api_key: str = None,  # NEW: Custom LLM API key
        custom_llm_model_name: str = None,  # NEW: Custom model name
        qdrant_client: Optional[Any] = None # NEW: Pre-initialized client support
    ):
        """Initialize chatbot with comprehensive configuration + custom LLM support"""
        
        # Initialize LLM based on environment
        app_env = os.getenv('APP_ENV', 'local')

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs=encode_kwargs or {"normalize_embeddings": True}
        )
        
        if use_custom_llm and custom_llm_url:
            # Use custom/internal LLM
            logger.info(f"🔧 Initializing CUSTOM LLM: {custom_llm_url}")
            self.llm = self._initialize_custom_llm(
                custom_llm_url,
                custom_llm_api_key,
                custom_llm_model_name or llm_model,
                llm_temperature,
                max_tokens
            )
            self.llm_type = "custom"
            logger.info(f"✅ Custom LLM initialized: {custom_llm_model_name or llm_model}")
            
        elif app_env == 'production':
            # Use Groq Cloud LLM
            logger.info("🚀 Initializing Groq Cloud LLM (Production Mode)")
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                logger.error("❌ GROQ_API_KEY not found in environment variables!")
                raise ValueError("GROQ_API_KEY is required for production mode")
            
            # Using a powerful model by default for production
            production_model = "llama-3.3-70b-versatile"
            self.llm = ChatGroq(
                model_name=production_model,
                temperature=llm_temperature,
                groq_api_key=groq_api_key
            )
            self.llm_type = "groq"
            logger.info(f"✅ Groq LLM initialized: {production_model}")
            
        else:
            # Use Ollama (Local Mode)
            logger.info("🏠 Initializing Ollama LLM (Local Mode)")
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://192.168.0.56:11434')
            model = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b')
            logger.info(f"🔗 Connecting to Ollama at: {base_url}")
            
            self.llm = ChatOllama(
                model=model,
                temperature=llm_temperature,
                base_url=base_url,
                timeout=120
            )
            self.llm_type = "ollama"
            logger.info(f"✅ Ollama LLM initialized: {model}")
        
        # FIXED: Initialize Qdrant with persistent collection
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        # Use the persistent collection name from vectors.py
        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        
        if qdrant_client:
            # Use the pre-initialized singleton client
            self.qdrant_client = qdrant_client
            logger.info("Using pre-initialized Qdrant client singleton")
        elif not self.qdrant_url or self.qdrant_url.lower() == "local":
            # Use local persistent storage
            storage_path = os.path.join(os.getcwd(), "qdrant_storage")
            os.makedirs(storage_path, exist_ok=True)
            self.qdrant_client = QdrantClient(path=storage_path)
            logger.info(f"Using local Qdrant storage at {storage_path}")
        else:
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                prefer_grpc=False
            )
            logger.info(f"Connected to Qdrant at {self.qdrant_url}")
        
        # Check if collection exists before initializing vector store
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            if not collection_exists:
                logger.warning(f"Collection '{self.collection_name}' does not exist yet. It will be created when documents are processed.")
        except Exception as e:
            logger.warning(f"Could not check collection existence: {e}")
        
        # Initialize vector store - FIX for newer LangChain versions
        try:
            # Try the newer initialization method (works with langchain-qdrant >= 0.1.0)
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )
        except TypeError as e:
            # Fallback: Try older initialization method
            logger.warning(f"QdrantVectorStore() failed: {e}, trying from_existing_collection")
            self.vector_store = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=self.collection_name,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
        except Exception as e:
            # If collection doesn't exist, set vector_store to None temporarily
            logger.warning(f"Could not initialize vector store: {e}. Will initialize after documents are processed.")
            self.vector_store = None
        
        # Retrieval settings
        self.retrieval_k = retrieval_k
        self.score_threshold = score_threshold
        
        # Basic retriever - only initialize if vector_store exists
        if self.vector_store is not None:
            self.basic_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.retrieval_k,
                    "score_threshold": self.score_threshold
                }
            )
            
            # Advanced retriever with hybrid search
            self.advanced_retriever = AdvancedRetriever(
                self.vector_store,
                self.embeddings,
                k=self.retrieval_k,
                score_threshold=self.score_threshold
            )
        else:
            self.basic_retriever = None
            self.advanced_retriever = None
            logger.warning("Retrievers not initialized - waiting for documents to be processed")
        
        # ENHANCED: RAG Prompt Template for Elite Forensic Analysis
        self.prompt_template = """ROLE: You are an Elite Litigation Strategist and Senior Counsel.
Your mission is to perform a HIGH-STAKES FORENSIC ANALYSIS using the provided context.

RESPONSE STRUCTURE (Strictly follow this formatting):
📌 CASE AT A GLANCE: [A simple, high-impact summary of the situation]

⚖️ THE CORE CONFLICT:
- **Parties**: [Who is the Petitioner vs Respondent]
- **The Issue**: [What is the central legal question or 'The Trap'?]

📜 LEGAL BACKGROUND & JOURNEY:
[Summarize how the case reached this point - e.g. NCLT -> NCLAT -> Supreme Court]

🛡️ STRATEGIC SYLLOGISM (IRAC):
- 🔍 **ISSUE**: [Specific legal question]
- 📜 **RULE**: [Cite Section/Act/Precedent from context]
- 📝 **APPLICATION**: [Directly connect facts to the law]
- ✅ **CONCLUSION**: [The tactical result]

💡 TACTICAL ACTION PLAN (The Win/Lag Strategy):
- [Immediate Action Item 1]
- [Immediate Action Item 2]

🧠 IN ONE LINE:
[A powerful one-sentence summary for the CEO/Partner]

Context:
{context}

User Question: {question}

SENIOR COUNSEL FORENSIC BRIEF:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        # ENHANCED: General Knowledge Prompt (The Strategic Legal Encyclopedia)
        self.general_prompt_template = """ROLE: You are an Elite Litigation Strategist and Senior Legal Partner.
Your mission is to provide high-stakes legal analysis and tactical advice.

ANALYSIS STRUCTURE:
📌 LEGAL LANDSCAPE: [Identify Acts/Sections - e.g. BNS, BNSS, Companies Act]
🛡️ STRATEGIC POSITIONING: [Offensive and Defensive moves]
💡 TACTICAL LOOPHOLES: [Procedural gaps to exploit]
⏳ CRITICAL ACTION ITEMS: [Bullet points for immediate execution]

User Question: {question}

SENIOR PARTNER ANALYSIS:"""
        
        # ENHANCED: Forensic Layman Mode (CEO Briefing Style)
        self.layman_prompt_template = """ROLE: You are the Lead Defense Counsel explaining a case to a CEO.
Your goal is to make the situation crystal clear and highly strategic.

RESPONSE STRUCTURE:
🧾 WHAT THIS IS ABOUT (Simple Explanation):
[Jargon-free summary]

🔍 THE CORE ISSUE:
[What is the main problem we are solving?]

🛡️ THE TRAP & THE SHIELD:
- **The Trap**: [What the opponent is trying to do to us]
- **The Shield**: [How we block them using the law]

🧠 IN ONE LINE:
[The ultimate bottom line for the CEO]

USER QUESTION:
{question}

CEO BRIEFING:"""
        
        # Chain configuration
        self.chain_type_kwargs = {"prompt": self.prompt}
        
        # Initialize QA chain only if vector_store and retriever exist
        if self.vector_store is not None and self.basic_retriever is not None:
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            self.qa_chain = (
                {
                    "context": self.basic_retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            self.qa_chain = None
            logger.warning("QA chain not initialized - waiting for documents to be processed")
        
        # Security and content filtering
        self.content_filter = ContentFilter()
        self.token_counter = TokenCounter()
        
        # Session statistics and persistence
        self.session_stats = {
            'total_queries': 0,
            'flagged_queries': 0,
            'total_tokens_used': 0,
            'input_tokens_used': 0,
            'output_tokens_used': 0,
            'start_time': time.time()
        }
        self.last_retrieved_context = None # Persistence for follow-ups
        self.last_query = "" # Track previous query for expansion
        
        logger.info(f"ChatbotManager initialized with persistent collection: {self.collection_name}")
    
    def _initialize_custom_llm(
        self,
        custom_url: str,
        api_key: str,
        model_name: str,
        temperature: float,
        max_tokens: int
    ):
        """
        Initialize custom/internal LLM endpoint
        
        Supports OpenAI-compatible APIs including:
        - vLLM deployments (internal model serving)
        - FastChat servers
        - Ollama local models
        - LM Studio
        - Internal company LLMs (trained on intranet data)
        - Any OpenAI-compatible endpoint
        
        Args:
            custom_url: Base URL of the custom LLM API
            api_key: API key (optional for internal endpoints)
            model_name: Model name to use
            temperature: Temperature setting
            max_tokens: Max tokens in response
            
        Returns:
            Initialized LLM instance
        """
        try:
            from langchain_openai import ChatOpenAI
            
            logger.info(f"Connecting to custom LLM endpoint: {custom_url}")
            logger.info(f"Model: {model_name}")
            
            # Create custom LLM with OpenAI-compatible interface
            custom_llm = ChatOpenAI(
                base_url=custom_url,  # Your internal endpoint
                api_key=api_key or "not-needed",  # Some internal endpoints don't need API keys
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60.0,
                max_retries=2,
                request_timeout=60.0
            )
            
            # Test connection with a simple query
            try:
                test_response = custom_llm.invoke("Test")
                logger.info("✅ Custom LLM connection test successful")
            except Exception as test_error:
                logger.warning(f"⚠️ Custom LLM test query failed: {test_error}")
                logger.warning("Proceeding anyway - errors may occur during actual use")
            
            return custom_llm
            
        except ImportError as e:
            error_msg = "langchain-openai not installed. Install with: pip install langchain-openai"
            logger.error(error_msg)
            raise ImportError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to initialize custom LLM: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _reinitialize_vector_store(self):
        """Reinitialize vector store after documents have been processed"""
        try:
            logger.info(f"Attempting to reinitialize vector store for collection: {self.collection_name}")
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                logger.warning(f"Collection '{self.collection_name}' still does not exist")
                return
            
            # Try newer initialization method
            try:
                self.vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings
                )
            except TypeError:
                # Fallback to older method
                self.vector_store = QdrantVectorStore.from_existing_collection(
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key
                )
            
            # Reinitialize retrievers
            self.basic_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.retrieval_k,
                    "score_threshold": self.score_threshold
                }
            )
            
            self.advanced_retriever = AdvancedRetriever(
                self.vector_store,
                self.embeddings,
                k=self.retrieval_k,
                score_threshold=self.score_threshold
            )
            
            # Reinitialize QA chain
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            self.qa_chain = (
                {
                    "context": self.basic_retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("✅ Vector store, retrievers, and QA chain successfully reinitialized")
            
        except Exception as e:
            logger.error(f"Failed to reinitialize vector store: {e}")
            raise
    
    def _calculate_real_tokens(self, query: str, answer: str, context: str) -> Dict[str, int]:
        """FIXED: Calculate actual token usage accurately"""
        # Input tokens: query + context + prompt overhead
        prompt_overhead = self.token_counter.count_tokens(self.prompt_template)
        query_tokens = self.token_counter.count_tokens(query)
        context_tokens = self.token_counter.count_tokens(context)
        
        input_tokens = query_tokens + context_tokens + prompt_overhead
        
        # Output tokens: the generated answer
        output_tokens = self.token_counter.count_tokens(answer)
        
        # Total tokens
        total_tokens = input_tokens + output_tokens
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'query_tokens': query_tokens,
            'context_tokens': context_tokens
        }
    
    async def get_response(
        self,
        query: str,
        enable_content_filter: bool = True,
        enable_pii_detection: bool = True,
        use_rag: bool = True,
        layman_mode: bool = False,
        chatbot_mode: str = "Hybrid (Smart)"
    ) -> Dict[str, Any]:
        """Generate response with comprehensive security and token tracking - HYBRID MODE"""
        try:
            if not query or not query.strip(): 
                return self._create_error_response("Empty query provided", "invalid_input")
            
            query = query.strip()[:MAX_QUERY_LENGTH]
            self.session_stats['total_queries'] += 1
            
            # Pre-process query security
            if enable_content_filter or enable_pii_detection:
                if enable_pii_detection:
                    has_pii, pii_types, cleaned_query = self.content_filter.scan_for_pii(query)
                    if has_pii:
                        self.session_stats['flagged_queries'] += 1
                        return self._create_error_response(f"PII detected: {', '.join(pii_types)}", "pii_detected")
                
                if enable_content_filter:
                    has_safety_issues, safety_issues = self.content_filter.check_content_safety(query)
                    if has_safety_issues:
                        self.session_stats['flagged_queries'] += 1
                        return self._create_error_response(f"Blocked: {', '.join(safety_issues)}", "content_filtered")
            
            if use_rag and (self.vector_store is None or self.basic_retriever is None):
                try: 
                    self._reinitialize_vector_store()
                except: 
                    use_rag = False
            
            if use_rag and self.vector_store is not None and self.basic_retriever is not None:
                # ⚖️ SPECIALIST FIRM LOGIC: Route to domain
                domain = self.route_query(query)
                
                # Determine RAG sub-mode
                rag_mode = "standard"
                if chatbot_mode == "Document Only":
                    rag_mode = "standard"
                elif chatbot_mode == "Hybrid (Smart)":
                    rag_mode = "standard"
                    
                return await self._get_rag_response(query, enable_content_filter, enable_pii_detection, mode=rag_mode, domain=domain)
            else:
                return await self._get_direct_llm_response(query, enable_content_filter, enable_pii_detection, layman_mode)

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._create_error_response(f"System error: {str(e)}", "system_error")
    
    async def _get_rag_response(
        self,
        query: str,
        enable_content_filter: bool,
        enable_pii_detection: bool,
        mode: str = "standard",
        domain: str = "general_legal"
    ) -> Dict[str, Any]:
        """Get response using RAG with Specialist Routing and Fallback"""
        try:
            # OPTION 1: Standard Vector Search (Enhanced with Specialist Logic)
            if mode == "standard":
                logger.info(f"Using Advanced Hybrid Search for Domain: {domain}")
                
                docs = []
                # ⚖️ Specialist search for specific domains
                if domain and domain != "general_legal":
                    logger.info(f"⚖️ Routing query to {domain} specialty...")
                    # Use Advanced Hybrid Search (Keyword + Semantic)
                    all_found = self.advanced_retriever.hybrid_search(query, k=self.retrieval_k * 2)
                    
                    # 💡 FIX: Include docs matching domain OR documents with NO specialty tag (User Uploads)
                    docs = [d for d in all_found if d.metadata.get('specialty') == domain or not d.metadata.get('specialty')]
                    
                    # Ensure we don't have too many, yet keep the best ones
                    docs = docs[:self.retrieval_k]
                    
                    # FALLBACK: If we still have nothing relevant, check everything
                    if not docs:
                        logger.info(f"⚠️ No specific {domain} or untagged docs found. Checking fallback...")
                        docs = self.advanced_retriever.hybrid_search(query, k=self.retrieval_k)
                # 💡 ENHANCEMENT: Sticky Context persistence (Stay on the case document)
                # Expand keywords to capture more legal follow-up intents
                is_follow_up = any(kw in query.lower() for kw in [
                    "this", "it", "about", "case", "detail", "more", "explain", "elaborate", 
                    "flaw", "gap", "issue", "gather", "next", "why", "how", "who", "where", "what"
                ])
                is_briefing_request = any(kw in query.lower() for kw in ["brief", "who", "summary", "analyze", "overview"])
                
                # If search is dry but we have history, LATCH ON to the history
                if not docs and self.last_retrieved_context and (is_follow_up or len(query.split()) < 10):
                    logger.info("🔄 Sticky Context: search was dry, but latching on to last known case context...")
                    context = self.last_retrieved_context
                elif docs:
                    # Build context with specific instructions for the AI
                    context = "\n\n".join(f"[DOCUMENT: {d.metadata.get('file_name', 'Case File')}]\n{d.page_content}" for d in docs)
                    self.last_retrieved_context = context # Save for next follow-up
                else:
                    # Fallback to history if it exists, even if not a clear follow-up
                    context = self.last_retrieved_context if self.last_retrieved_context else "No specific documents found."
                
                # ⚡ STAGE A: Forensic Case Mapping (Mental Draft)
                # If it's a new briefing or search triggered new docs, extract the 'Who is Who'
                case_map_metadata = ""
                if (is_briefing_request or not self.last_query) and docs:
                    try:
                        mapping_prompt = f"""Identify the legal parties and the core conflict from this context:
                        {context[:5000]} # Increase lookup for parties
                        
                        OUTPUT FORMAT:
                        PARTIES: [e.g. Tata Sons (Petitioner) vs Cyrus Mistry (Respondent)]
                        CONFLICT: [1 sentence summary of dispute]
                        """
                        map_response = await self.llm.ainvoke(mapping_prompt)
                        case_map_metadata = map_response.content if hasattr(map_response, 'content') else str(map_response)
                        logger.info(f"Forensic Case Map Generated: {case_map_metadata}")
                    except Exception as e:
                        logger.warning(f"Forensic mapping failed: {e}")

                # Format prompt with the Forensic Map if generated
                final_context = f"{case_map_metadata}\n\n{context}" if case_map_metadata else context
                
                # STRIKE THROUGH: Never let the AI say "blank slate" if context is present
                if "No specific documents found" not in final_context:
                    disclaimer = f"STRICT MANDATE: You are currently counsel for the case described in the context. Do NOT claim you have no information. Analyze the existing text for: {query}"
                    chain_prompt = self.prompt.format(context=final_context, question=f"{disclaimer}\n\nUser Question: {query}")
                else:
                    chain_prompt = self.prompt.format(context=final_context, question=query)
                
                # ENHANCED: Deep Analysis Instruction for the LLM
                if any(kw in query.lower() for kw in ["analysis", "case", "detail", "tactical", "about", "flaw", "issue"]):
                    chain_prompt = f"INSTRUCTION: Perform a deep, forensic analysis using ONLY the provided context. Connect the parties identified to the legal rules. Identify the TRAP and the LOOPHOLE.\n\n{chain_prompt}"
                
                self.last_query = query # Save query
                response = await self.llm.ainvoke(chain_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                sources = self._process_sources(docs)
                response_type = "standard_rag"
                process_type = "Vector Similarity"

            # OPTION 2: LightRAG Knowledge Graph Search (Deep / Smart)
            else:
                logger.info(f"Using LightRAG (Knowledge Graph) for: {query}")
                lrag = get_lightrag_manager()
                answer = await lrag.query(query, mode="hybrid")
                
                if not answer or "Error" in answer:
                    logger.warning("LightRAG not ready, falling back to standard search")
                    return await self._get_rag_response(query, enable_content_filter, enable_pii_detection, mode="standard")
                
                context = "LightRAG Knowledge Graph"
                sources = [{"type": "lightrag", "context": "Knowledge Graph"}]
                response_type = "lightrag"
                process_type = "Hybrid (Graph + Vector)"

            # Post-process response security
            if enable_content_filter or enable_pii_detection:
                filtered_answer, has_filter_issues, filter_issues = self.content_filter.filter_response(answer)
                answer = filtered_answer

            # Record processing time
            processing_time = time.time()
            
            # Calculate tokens
            token_breakdown = self._calculate_real_tokens(query, answer, context)
            
            # Update session stats
            self.session_stats['total_tokens_used'] += token_breakdown['total_tokens']
            
            return {
                "answer": answer.strip(),
                "sources": sources,
                "is_flagged": False,
                "flag_reason": None,
                "tokens_used": token_breakdown['total_tokens'],
                "input_tokens": token_breakdown['input_tokens'],
                "output_tokens": token_breakdown['output_tokens'],
                "processing_time": processing_time,
                "response_type": response_type,
                "process_info": process_type
            }
            
        except Exception as e:
            logger.error(f"RAG response error: {e}")
            return self._create_error_response(
                "Unable to process documents to answer your question. Please try rephrasing your question.",
                "rag_processing_error"
            )
    
    async def _get_direct_llm_response(
        self,
        query: str,
        enable_content_filter: bool,
        enable_pii_detection: bool,
        layman_mode: bool = False
    ) -> Dict[str, Any]:
        """Get response directly from LLM without RAG (general knowledge) with optional layman mode"""
        try:
            # Choose appropriate prompt based on mode
            if layman_mode:
                general_prompt = self.layman_prompt_template.format(question=query)
                response_type = "layman"
            else:
                general_prompt = self.general_prompt_template.format(question=query)
                response_type = "general_knowledge"
            
            # Get response from LLM
            response = await self.llm.ainvoke(general_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Calculate token usage (no document context)
            token_breakdown = self._calculate_real_tokens(query, answer, general_prompt)
            
            # Post-process response security
            if enable_content_filter or enable_pii_detection:
                filtered_answer, has_filter_issues, filter_issues = self.content_filter.filter_response(answer)
                if has_filter_issues:
                    self.session_stats['flagged_queries'] += 1
                    return {
                        "answer": filtered_answer,
                        "sources": [],
                        "is_flagged": True,
                        "flag_reason": f"Response filtered: {', '.join(filter_issues)}",
                        "tokens_used": token_breakdown['total_tokens'],
                        "input_tokens": token_breakdown['input_tokens'],
                        "output_tokens": token_breakdown['output_tokens'],
                        "response_type": response_type
                    }
                answer = filtered_answer
            
            # Update session stats
            self.session_stats['total_tokens_used'] += token_breakdown['total_tokens']
            self.session_stats['input_tokens_used'] += token_breakdown['input_tokens']
            self.session_stats['output_tokens_used'] += token_breakdown['output_tokens']
            
            return {
                "answer": answer.strip() or "I'm not sure how to answer that question.",
                "sources": [],  # No document sources for general knowledge
                "is_flagged": False,
                "flag_reason": None,
                "tokens_used": token_breakdown['total_tokens'],
                "input_tokens": token_breakdown['input_tokens'],
                "output_tokens": token_breakdown['output_tokens'],
                "processing_time": time.time(),
                "response_type": response_type
            }
            
        except Exception as e:
            logger.error(f"General knowledge response error: {e}")
            return self._create_error_response(f"System error: {str(e)}", "system_error")

    def _create_error_response(self, message: str, reason: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "answer": f"⚠️ {message}",
            "sources": [],
            "is_flagged": True,
            "flag_reason": reason,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "processing_time": time.time(),
            "response_type": "error"
        }

    def route_query(self, query: str) -> str:
        """
        ⚖️ THE LAWYER ROUTER: Categorize query into legal specialty
        This is used for specialist firm filtering.
        """
        query_lower = query.lower()
        
        # 🏢 Corporate / Company Law
        if any(kw in query_lower for kw in ["company", "directors", "board", "shareholder", "mca", "roc", "audit", "compliance", "corporate"]):
            return "corporate_law"
            
        # ⚖️ Criminal Law / BNSS / BNS
        if any(kw in query_lower for kw in ["criminal", "police", "arrest", "bail", "jail", "ipc", "bns", "bnss", "fir", "crpc", "assault", "crime"]):
            return "criminal_law"
            
        # 📄 Contract Law
        if any(kw in query_lower for kw in ["contract", "agreement", "lease", "rent", "partnership", "vendor", "deed", "clause", "signing"]):
            return "contract_law"
            
        # 💳 Banking & Finance
        if any(kw in query_lower for kw in ["bank", "loan", "foreclosure", "debt", "recovery", "mortgage", "finance", "cheque", "dishonour", "sarfaesi"]):
            return "banking_finance_law"

        # 🖥️ Cyber Law / IT Act
        if any(kw in query_lower for kw in ["cyber", "online", "fraud", "hacked", "data", "privacy", "it act", "technology", "internet"]):
            return "cyber_law"
            
        # 💰 Taxation Law
        if any(kw in query_lower for kw in ["tax", "gst", "income tax", "vat", "customs", "excise", "assessment", "itat", "revenue", "tds"]):
            return "taxation_law"
            
        # 👷 Labour & Employment Law
        if any(kw in query_lower for kw in ["labour", "employee", "employer", "salary", "pf", "esi", "gratuity", "dismissal", "workplace", "posh", "workers", "trade union"]):
            return "labour_employment_law"
            
        # ⚖️ Litigation / Civil Procedure
        if any(kw in query_lower for kw in ["litigation", "court", "judge", "petition", "suit", "appeal", "high court", "supreme court", "summons", "civil"]):
            return "litigation_cases"
            
        # Default to general legal
        return "general_legal"

    def _process_sources(self, source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Process and secure source documents - simplified for UI"""
        if not source_documents:
            return []
        
        sources = []
        seen_sources = set()
        
        for i, doc in enumerate(source_documents[:8]):  # Limit to 8 sources
            try:
                metadata = doc.metadata
                
                # Extract metadata safely
                file_name = str(metadata.get('file_name', metadata.get('source', 'Unknown Document')))
                if '/' in file_name or '\\' in file_name:
                    file_name = os.path.basename(file_name)
                file_name = file_name[:50]  # Limit length
                
                page = metadata.get('page', 'N/A')
                
                # Deduplication based on file and page
                source_key = f"{file_name}_{page}"
                if source_key in seen_sources:
                    continue
                
                source_info = {
                    "file_name": file_name,
                    "page": page
                }
                
                sources.append(source_info)
                seen_sources.add(source_key)
                
            except Exception as e:
                logger.warning(f"Error processing source {i}: {e}")
                continue
        
        return sources

    def cleanup_collection(self) -> None:
        """
        FIXED: This method should NOT delete the persistent collection
        The collection "Legal_documents" is shared across all sessions
        Only call this if you want to delete ALL documents permanently
        """
        try:
            logger.warning(f"⚠️ cleanup_collection() called - this will delete the entire '{self.collection_name}' collection!")
            logger.warning("This is a shared persistent collection. Deletion will affect all users/sessions.")
            # Commenting out the actual deletion to prevent accidental data loss
            # If you really want to clear the collection, uncomment the line below:
            # self.qdrant_client.delete_collection(self.collection_name)
            logger.info(f"Collection cleanup skipped to preserve persistent data: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Collection cleanup operation: {e}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics with accurate token data"""
        current_time = time.time()
        session_duration = current_time - self.session_stats['start_time']
        
        return {
            **self.session_stats,
            "session_duration_seconds": round(session_duration, 2),
            "average_tokens_per_query": (
                self.session_stats['total_tokens_used'] / max(1, self.session_stats['total_queries'])
            ),
            "flagged_percentage": (
                (self.session_stats['flagged_queries'] / max(1, self.session_stats['total_queries'])) * 100
            ),
            "tokens_remaining": MAX_TOKENS_PER_SESSION - self.session_stats['total_tokens_used'],
            "input_output_ratio": (
                self.session_stats['input_tokens_used'] / max(1, self.session_stats['output_tokens_used'])
            )
        }

    def update_retrieval_settings(self, k: int = None, threshold: float = None) -> None:
        """Update retrieval parameters with validation"""
        try:
            if k is not None:
                self.retrieval_k = max(3, min(k, 15))
                self.advanced_retriever.k = self.retrieval_k
            
            if threshold is not None:
                self.score_threshold = max(0.0, min(threshold, 1.0))
                self.advanced_retriever.score_threshold = self.score_threshold
            
            # Update basic retriever
            self.basic_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.retrieval_k,
                    "score_threshold": self.score_threshold
                }
            )
            
            # Update chain using modern LangChain 1.0+ approach
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            self.qa_chain = (
                {
                    "context": self.basic_retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info(f"Updated retrieval: k={self.retrieval_k}, threshold={self.score_threshold}")
            
        except Exception as e:
            logger.error(f"Failed to update retrieval settings: {e}")

    def reset_session(self) -> None:
        """Reset session statistics"""
        self.session_stats = {
            'total_queries': 0,
            'flagged_queries': 0,
            'total_tokens_used': 0,
            'input_tokens_used': 0,
            'output_tokens_used': 0,
            'start_time': time.time()
        }
        logger.info("Session statistics reset")