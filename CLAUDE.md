# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Clinical RAG (Retrieval-Augmented Generation) system for Orthopedic Physical Therapy built with ZenML. The system processes clinical PDF documents, extracts and chunks text, generates embeddings, stores them in PostgreSQL with pgvector, and provides semantic search capabilities.

## Key Development Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize ZenML (if needed)
zenml init
```

### Database Operations
```bash
# Setup PostgreSQL database with schema
psql -d clinical_rag -f schema.sql

# Check database status
python utils/check_db_status.py
```

### Running Pipelines
```bash
# Set required environment variable for macOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Process a PDF document
python run.py ingest data/raw/Achilles_Pain.pdf

# Search for clinical information
python run.py search "What exercises help with pain?"

# Check database status
python run.py status

# Run tests
python tests/test_pipeline.py
python tests/test_search.py
python tests/test_database.py
```

### ZenML Server Management
```bash
# Start ZenML server (if needed)
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES zenml up

# Stop ZenML server
zenml down
```

## Architecture Overview

### Core Pipeline Flow
The system follows a linear ZenML pipeline: **PDF → Extract → Preprocess → Chunk → Embed → Store**

1. **PDF Extraction** (`extract_text_from_pdf`): Uses pdfplumber to extract text with metadata
2. **Preprocessing** (`preprocess_text`): Minimal cleaning while preserving clinical content
3. **Chunking** (`chunk_text`): Sentence-based chunking with overlap and quality filtering
4. **Embedding** (`generate_embeddings`): Local sentence transformers (all-MiniLM-L6-v2)
5. **Storage** (`store_embeddings_in_database`): PostgreSQL with pgvector for similarity search

### Key Components

- **DatabaseManager** (`utils/database.py`): Handles all PostgreSQL operations with connection pooling
- **ClinicalRAGSearcher** (`utils/search.py`): Provides semantic search with cosine similarity using pgvector `<=>` operator
- **Pipeline Definition** (`pipelines/clinical_rag_pipeline.py`): ZenML pipeline orchestrating all steps
- **Individual Steps** (`steps/`): Modular ZenML pipeline steps (extract, preprocess, chunk, embed, store)
- **Main CLI** (`run.py`): Command-line interface for ingesting PDFs and searching

### Database Schema
- **documents**: PDF metadata and processing status
- **chunks**: Text chunks with section classification (treatment, diagnosis, methods, etc.)
- **embeddings**: 384-dimensional vectors with pgvector indices

### Search Capabilities
- Natural language queries with semantic similarity
- Section filtering (treatment, diagnosis, methods)
- Document filtering and similarity thresholds
- Query enhancement for clinical context

## Critical Environment Variables

- `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`: Required for ZenML on macOS
- `DATABASE_URL`: PostgreSQL connection (defaults to `postgresql://postgres:password@localhost:5432/clinical_rag`)
- `TOKENIZERS_PARALLELISM`: Set to `false` to avoid tokenizer warnings

## Data Flow

1. **Input**: Clinical PDFs in `data/raw/` (Achilles_Pain.pdf, Concussion.pdf)
2. **Processing**: Temporary outputs stored in `data/processed/` directory
3. **Storage**: Final embeddings and chunks in PostgreSQL
4. **Search**: ClinicalRAGSearcher queries the vector database via CLI (`python run.py search`)

## Project Structure

```
clinical-rag/
├── pipelines/           # ZenML pipeline definitions
├── steps/              # Individual ZenML pipeline steps
├── utils/              # Database, search, and utility functions
├── tests/              # Test files
├── configs/            # Configuration files (dev.yaml, production.yaml)
├── data/
│   ├── raw/           # Original PDF files
│   └── processed/     # Extracted and processed text files
├── run.py             # Main CLI entry point
├── setup.py           # Package configuration
├── schema.sql         # Database schema
└── requirements.txt   # Python dependencies
```

## Testing Structure

- `tests/test_search.py`: Comprehensive search functionality testing with clinical queries
- `tests/test_database.py`: Database connectivity and operations testing
- `tests/test_pipeline.py`: End-to-end pipeline execution testing
- `utils/check_db_status.py`: Database content inspection utility

## ZenML Integration

- Uses ZenML's step decorators and pipeline orchestration
- Artifact caching enabled for efficiency
- Steps return structured dictionaries with success/error handling
- Pipeline runs tracked in ZenML dashboard (localhost:8237)

## Current Status

✅ **Completed Features:**
- Proper Python package structure with modular design
- Individual ZenML pipeline steps in separate files
- Configuration management system (`configs/`, `utils/config.py`)
- Organized test structure (`tests/` directory)
- Command-line interface (`run.py`) for all operations
- High-quality semantic search (0.7+ similarity scores)

## Phase 1 Foundation Complete

This system successfully processes clinical documents and provides high-quality semantic search for clinical decision support queries. The foundation is ready for Phase 2 adaptive intelligence features.