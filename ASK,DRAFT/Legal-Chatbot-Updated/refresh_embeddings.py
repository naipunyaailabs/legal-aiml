import os
import asyncio
import logging
from pathlib import Path
from vectors import EmbeddingsManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PERSISTENT_COLLECTION_NAME = "Legal_documents"
DOCUMENTS_FOLDER = Path(__file__).parent / "documents"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def refresh_embeddings():
    """Clear existing embeddings and re-embed all documents from the documents folder"""
    try:
        logger.info(f"Starting embeddings refresh for collection: {PERSISTENT_COLLECTION_NAME}")
        
        # Initialize EmbeddingsManager
        embeddings_mgr = EmbeddingsManager(
            model_name="BAAI/bge-large-en-v1.5",
            device="cpu",
            encode_kwargs={"normalize_embeddings": True},
            qdrant_url=os.getenv('QDRANT_URL'),
            collection_name=PERSISTENT_COLLECTION_NAME,
            chunk_size=1200,
            chunk_overlap=300
        )
        
        # Clear the collection
        # Skip clearing — ADD new documents to existing collection
        # To do a full refresh, uncomment the next 2 lines:
        # embeddings_mgr.clear_collection()
        # logger.info("Collection cleared successfully.")
        logger.info("Step 1: Keeping existing embeddings, adding new documents...")
        
        # Get documents to process
        logger.info(f"Step 2: Scanning {DOCUMENTS_FOLDER} for documents...")
        allowed_extensions = {'.pdf', '.txt', '.docx', '.pptx', '.png', '.jpg', '.jpeg'}
        document_files = []
        
        if not DOCUMENTS_FOLDER.exists():
            logger.error(f"Documents folder {DOCUMENTS_FOLDER} not found!")
            return
            
        for file in DOCUMENTS_FOLDER.rglob('*'):
            if file.is_file() and file.suffix.lower() in allowed_extensions:
                document_files.append(file)
        
        if not document_files:
            logger.warning("No documents found to re-embed.")
            return
            
        logger.info(f"Found {len(document_files)} documents to process.")
        
        # Process each document (FIXED: await the async create_embeddings)
        processed_count = 0
        for doc_path in document_files:
            logger.info(f"Processing: {doc_path.name}...")
            try:
                msg = await embeddings_mgr.create_embeddings(str(doc_path))
                logger.info(f"✅ Success: {doc_path.name} - {msg}")
                processed_count += 1
            except Exception as e:
                logger.error(f"❌ Error processing {doc_path.name}: {e}")
        
        logger.info("=" * 50)
        logger.info(f"REFRESH COMPLETE")
        logger.info(f"Total documents processed: {processed_count}/{len(document_files)}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"An error occurred during refresh: {e}")

if __name__ == "__main__":
    asyncio.run(refresh_embeddings())
