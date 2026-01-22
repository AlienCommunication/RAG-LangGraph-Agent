import os
import sys
import shutil
import logging
import argparse
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Ingest")

def validate_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Input file is not readable: {file_path}")

def ingest_data(file_path: str) -> None:
    try:
        # 1. Validation
        validate_file(file_path)
        logger.info(f"Starting ingestion for: {file_path}")

        # 2. Load Data
        loader = TextLoader(file_path, encoding='utf-8')
        raw_docs = loader.load()
        logger.info(f"Loaded {len(raw_docs)} document(s).")

        # 3. Context-Aware Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(raw_docs)

        if not chunks:
            logger.warning("No chunks generated. Check file content.")
            return

        # 4. Enrich Metadata (Critical for Citation)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx + 1
            chunk.metadata["source"] = os.path.basename(file_path)
        
        logger.info(f"Created {len(chunks)} searchable chunks.")

        # 5. Idempotency: Reset DB
        if os.path.exists(VECTOR_DB_DIR):
            shutil.rmtree(VECTOR_DB_DIR)
            logger.info("Cleared existing vector store.")

        # 6. Create Vector Store
        logger.info("Generating embeddings (this may take a moment)...")
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=VECTOR_DB_DIR
        )
        logger.info(f"Ingestion complete. DB saved to {VECTOR_DB_DIR}")

    except Exception as e:
        logger.critical(f"Ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")
    parser.add_argument("file", help="Path to text file")
    args = parser.parse_args()
    ingest_data(args.file)