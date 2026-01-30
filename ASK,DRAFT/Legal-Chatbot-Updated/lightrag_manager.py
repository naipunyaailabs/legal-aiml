# lightrag_manager.py - LightRAG Integration for Legal AI
import os
import asyncio
import logging
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: Bypass proxy for Ollama
os.environ['NO_PROXY'] = '192.168.0.56'
os.environ['no_proxy'] = '192.168.0.56'

load_dotenv()

class LightRAGManager:
    """Manager for LightRAG (Knowledge Graph + Vector RAG)"""
    
    def __init__(self, working_dir: str = "./lightrag_storage"):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        
        # Determine environment and models
        self.app_env = os.getenv('APP_ENV', 'local')
        # Use the confirmed host
        self.ollama_host = os.getenv('OLLAMA_BASE_URL', 'http://192.168.0.56:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b')
        self.embed_model = os.getenv('OLLAMA_EMBED_MODEL', 'qwen3-embedding:0.6b')
        
        # Initialize LightRAG
        self.rag = self._setup_rag()
        logger.info(f"LightRAG initialized in {working_dir} using {self.app_env} mode")

    def _setup_rag(self) -> LightRAG:
        """Setup LightRAG with appropriate LLM and Embedding functions"""
        
        if self.app_env == 'production':
            # Production - Use Groq (via OpenAI-compatible interface)
            groq_api_key = os.getenv("GROQ_API_KEY")
            
            async def groq_complete(prompt, system_prompt=None, history=None, **kwargs):
                # Use 8B model for higher rate limits during indexing/querying
                h_messages = history or kwargs.pop("history_messages", [])
                kwargs.pop("model", None)
                kwargs.pop("prompt", None)
                
                return await openai_complete_if_cache(
                    model="llama-3.1-8b-instant",
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=h_messages,
                    base_url="https://api.groq.com/openai/v1",
                    api_key=groq_api_key,
                    **kwargs
                )
            
            # Use local HuggingFace embeddings in production to avoid Ollama dependency
            from langchain_huggingface import HuggingFaceEmbeddings
            hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            async def hf_embed_func(texts):
                embeddings = hf_embeddings.embed_documents(texts)
                return np.array(embeddings)

            return LightRAG(
                working_dir=self.working_dir,
                llm_model_name="llama-3.1-8b-instant", # Explicitly set model name
                llm_model_func=groq_complete,
                llm_model_max_async=1, # Correct param name to throttle Groq (limit 1 for stability)
                chunk_token_size=600, # Smaller chunks to fit in Groq's 6k TPM limit
                embedding_func=EmbeddingFunc(
                    embedding_dim=384,
                    max_token_size=8192,
                    func=hf_embed_func
                )
            )
        else:
            # Local - Use Ollama
            async def llm_model_func(prompt, system_prompt=None, history=None, **kwargs):
                # LightRAG passes the model name via hashing_kv, but we can also override here
                return await ollama_model_complete(
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history or [],
                    host=self.ollama_host,
                    **kwargs
                )

            async def embedding_func(texts):
                return await ollama_embed(
                    texts,
                    embed_model=self.embed_model,
                    host=self.ollama_host
                )

            return LightRAG(
                working_dir=self.working_dir,
                llm_model_func=llm_model_func,
                llm_model_name=self.ollama_model,
                embedding_func=EmbeddingFunc(
                    embedding_dim=1024,
                    max_token_size=8192,
                    func=embedding_func
                ),
                # Legal optimized chunking
                chunk_overlap_token_size=100  # Overlap for context preservation
            )

    async def _ensure_initialized(self):
        """Ensure LightRAG storages are initialized (required for insert/query)"""
        if not hasattr(self, '_initialized') or not self._initialized:
            logger.info("Initializing LightRAG storages...")
            # If we are already in an async context, just await it
            await self.rag.initialize_storages()
            self._initialized = True
            logger.info("LightRAG storages initialized")

    async def insert_text(self, text: str):
        """Insert raw text into LightRAG"""
        if not text or not text.strip():
            return
        logger.info(f"Inserting text into LightRAG ({len(text)} chars)...")
        try:
            await self._ensure_initialized()
            await self.rag.ainsert(text)
            logger.info("Successfully inserted text into LightRAG")
        except Exception as e:
            logger.error(f"LightRAG insertion error: {e}")

    async def query(self, query: str, mode: str = "hybrid") -> str:
        """Query LightRAG
        
        Modes: naive, local, global, hybrid
        """
        try:
            await self._ensure_initialized()
            # Use aquery for non-blocking execution in FastAPI
            return await self.rag.aquery(query, param=QueryParam(mode=mode))
        except Exception as e:
            logger.error(f"LightRAG query error: {e}")
            return f"Error querying LightRAG: {str(e)}"

    def clear_storage(self):
        """Clear all LightRAG storage (knowledge graph + embeddings)"""
        try:
            logger.warning("Clearing LightRAG storage...")
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
                os.makedirs(self.working_dir, exist_ok=True)
            # Re-initialize the RAG instance
            self.rag = self._setup_rag()
            logger.info("LightRAG storage cleared and re-initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to clear LightRAG storage: {e}")
            return False

# Singleton pattern
_lightrag_instance = None

def get_lightrag_manager() -> LightRAGManager:
    global _lightrag_instance
    if _lightrag_instance is None:
        _lightrag_instance = LightRAGManager()
    return _lightrag_instance
