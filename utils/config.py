"""Configuration settings for the Clinical RAG system."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"  # PDFs are in data/raw/
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"  # Processed files
MODELS_DIR = PROJECT_ROOT / "models"

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/clinical_rag")

# Processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"