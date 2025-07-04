# Clinical RAG System

> **Adaptive Clinical Decision Support System for Orthopedic Physical Therapy**  
> A comprehensive ZenML example project demonstrating MLOps best practices for healthcare AI

[![ZenML](https://img.shields.io/badge/Built%20with-ZenML-blue)](https://zenml.io)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL%20%2B%20pgvector-blue)](https://postgresql.org)

## 🎯 Project Overview

This project demonstrates how to build a production-ready **Retrieval-Augmented Generation (RAG) system** for clinical decision support using **ZenML's MLOps framework**. It serves as both a practical healthcare AI tool and a comprehensive example of implementing MLOps best practices for LLM-based applications.

### 🏥 Clinical Use Case
**Problem**: Physical therapists need quick access to evidence-based guidance for complex orthopedic cases, but manually searching clinical literature during patient care is impractical.

**Solution**: An adaptive clinical decision support system that:
- Provides instant semantic search over clinical practice guidelines
- Adapts information sources based on case complexity (Phase 2)
- Maintains clinical context and section awareness
- Ensures high-quality, relevant recommendations (0.7+ similarity scores)

### 🛠️ ZenML MLOps Showcase
This project exemplifies modern MLOps practices using ZenML:
- **Modular Pipeline Architecture**: Separate steps for extraction, preprocessing, chunking, embedding, and storage
- **Artifact Management**: Proper tracking and versioning of processed data and embeddings
- **Configuration Management**: Environment-specific configs for dev/production
- **Local Model Deployment**: Sentence transformers + Ollama LLM without API costs
- **Database Integration**: PostgreSQL with pgvector for production-ready vector storage
- **Dual CLI Interface**: Separate commands for retrieval (`search`) and RAG (`ask`)
- **Complete RAG Pipeline**: End-to-end retrieval-augmented generation workflow

## 🏗️ Architecture

### Ingestion Pipeline (ZenML)
```
┌─────────────────┐    ┌──────────────────────────────────────┐    ┌─────────────────┐
│   Clinical PDFs │───▶│           ZenML Pipeline             │───▶│   PostgreSQL    │
│   (Guidelines)  │    │  Extract→Preprocess→Chunk→Embed     │    │   + pgvector    │
└─────────────────┘    └──────────────────────────────────────┘    └─────────────────┘
```

### Query Time Architecture
```
                        ┌─────────────────┐
                        │   PostgreSQL    │
                        │   + pgvector    │
                        └─────────┬───────┘
                                  │
                      ┌───────────▼───────────┐
                      │   Semantic Search     │
                      │   (Clinical Context)  │
                      └───────────┬───────────┘
                                  │
                          ┌───────▼───────┐
                          │ CLI Interface │
                          │ (search/ask)  │
                          └───┬───────┬───┘
                              │       │
                    ┌─────────▼───┐   │
                    │   search    │   │
                    │ (Raw Chunks)│   │
                    └─────────────┘   │
                                      │
                              ┌───────▼───────┐
                              │      ask      │
                              │ (RAG: Chunks  │
                              │ + Ollama LLM) │
                              └───────────────┘
```

### Pipeline Flow
**Ingestion**: `PDF → Extract → Preprocess → Chunk → Embed → Store`
**Query**: `search` returns raw chunks | `ask` adds LLM generation

1. **PDF Extraction**: pdfplumber handles complex medical document layouts
2. **Preprocessing**: Minimal cleaning while preserving clinical content
3. **Chunking**: Sentence-based chunking respecting clinical section boundaries
4. **Embedding**: Local sentence transformers (all-MiniLM-L6-v2)
5. **Storage**: PostgreSQL with pgvector for similarity search
6. **Retrieval**: Semantic search with clinical context awareness
7. **Generation**: Ollama Llama 3.2 3B creates clinical responses with retrieved context

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL with pgvector extension
- Ollama (for local LLM inference)
- 4GB+ RAM (for local embeddings and LLM)

### Installation

1. **Clone and setup environment**:
```bash
git clone https://github.com/your-org/clinical-rag
cd clinical-rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup database**:
```bash
# Create PostgreSQL database with pgvector
createdb clinical_rag
psql -d clinical_rag -c "CREATE EXTENSION vector;"
psql -d clinical_rag -f schema.sql
```

4. **Install and setup Ollama**:
```bash
# Install Ollama (visit https://ollama.ai for platform-specific instructions)
# Pull the Llama 3.2 3B model
ollama pull llama3.2:3b
```

5. **Initialize ZenML**:
```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # macOS only
zenml init
```

### Usage

**Process clinical documents**:
```bash
python run.py ingest data/raw/Achilles_Pain.pdf
```

**Search for raw chunks (retrieval only)**:
```bash
python run.py search "What exercises help with Achilles tendon pain?"
```

**Ask clinical questions with AI response (full RAG)**:
```bash
python run.py ask "What exercises help with Achilles tendon pain?"
```

**Check system status**:
```bash
python run.py status
```

**Run tests**:
```bash
python tests/test_pipeline.py
python tests/test_search.py
python tests/test_database.py
```

## 📊 Current Status

### ✅ Phase 1: Complete RAG System (COMPLETED)
- [x] **Data Ingestion Pipeline**: PDF processing with pdfplumber
- [x] **Vector Database**: PostgreSQL + pgvector setup
- [x] **Embedding Generation**: Local sentence transformers
- [x] **Semantic Search**: Clinical context-aware retrieval (0.7+ similarity)
- [x] **LLM Integration**: Local Ollama with Llama 3.2 3B model
- [x] **Augmented Generation**: Clinical response generation with context
- [x] **Dual CLI Interface**: Separate `search` (retrieval) and `ask` (RAG) commands
- [x] **Modular Architecture**: Professional Python package structure
- [x] **Configuration Management**: Dev/production configs
- [x] **Test Suite**: Comprehensive testing framework

**Current Performance**:
- 📄 **2 clinical documents** processed (Achilles Pain, Concussion)
- 🧩 **869 text chunks** with section awareness
- 🔍 **Sub-5 second** retrieval response time
- 🤖 **Sub-15 second** end-to-end RAG response time
- 🎯 **0.7+ similarity scores** for clinical queries
- 🏥 **Section filtering** (treatment, diagnosis, methods)
- 💬 **Professional clinical responses** with evidence citations

### 🔄 Phase 1: Evaluation Framework (IN PROGRESS)
- [ ] **Clinical Test Suite**: 20-30 expert-validated scenarios
- [ ] **Evaluation Metrics**: Precision@k, recall@k, clinical accuracy
- [ ] **Expert Validation**: Clinical expert review system
- [ ] **Automated Evaluation**: ZenML evaluation pipeline

### 🎯 Phase 2: Adaptive Intelligence (PLANNED)
- [ ] **Context Classification**: Complexity-based routing
- [ ] **Multi-Source Retrieval**: CPGs → Research literature → Case studies
- [ ] **Adaptive Routing**: Automatic escalation for complex cases
- [ ] **Enhanced Evaluation**: Clinical expert validation at scale

## 🧪 ZenML Pipeline Details

### Pipeline Steps
```python
@pipeline
def pdf_processing_pipeline(
    pdf_path: str, 
    chunk_size: int = 1000, 
    overlap: int = 200, 
    model_name: str = "all-MiniLM-L6-v2"
):
    extraction_result = extract_text_from_pdf(pdf_path)
    processed_result = preprocess_text(extraction_result)
    chunking_result = chunk_text(processed_result, chunk_size, overlap)
    embedding_result = generate_embeddings(chunking_result, model_name)
    storage_result = store_embeddings_in_database(embedding_result)
    return storage_result
```

### Key ZenML Features Demonstrated
- **Step Decorators**: Individual pipeline steps with clear interfaces
- **Artifact Caching**: Efficient pipeline re-runs
- **Parameter Management**: Configurable pipeline parameters
- **Error Handling**: Robust error propagation and logging
- **Metadata Tracking**: Comprehensive artifact and run metadata

## 🔧 Configuration

### Development vs Production
```yaml
# configs/dev.yaml
enable_cache: false
chunk_size: 1000
database:
  max_connections: 5
processing:
  save_temp_files: true

# configs/production.yaml  
enable_cache: true
chunk_size: 800
database:
  max_connections: 20
processing:
  save_temp_files: false
```

### Environment Variables
```bash
# Required for ZenML on macOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Database configuration
export DATABASE_URL="postgresql://user:pass@localhost:5432/clinical_rag"

# Optional: Disable tokenizer warnings
export TOKENIZERS_PARALLELISM=false
```

## 📁 Project Structure

```
clinical-rag/
├── pipelines/           # ZenML pipeline definitions
│   ├── document_ingestion_pipeline.py  # PDF ingestion pipeline
│   └── response_generation_pipeline.py # RAG generation pipeline
├── steps/              # Individual modular pipeline steps
│   ├── extract_text_from_pdf.py        # PDF text extraction
│   ├── preprocess_text.py              # Text preprocessing
│   ├── chunk_text.py                   # Text chunking
│   ├── generate_embeddings.py          # Embedding generation
│   ├── store_embeddings.py             # Database storage
│   ├── format_retrieval_context.py     # Context formatting for LLM
│   ├── build_clinical_prompt.py        # Clinical prompt construction
│   ├── generate_with_ollama.py         # LLM generation with Ollama
│   ├── parse_and_validate_response.py  # Response validation
│   └── extract_metadata_and_citations.py # Citation extraction
├── utils/              # Core utilities
│   ├── database.py     # PostgreSQL + pgvector manager
│   ├── search.py       # Clinical semantic search
│   └── config.py       # Configuration management
├── tests/              # Comprehensive test suite
├── configs/            # Environment configurations
├── data/
│   ├── raw/           # Original clinical PDFs
│   └── processed/     # Pipeline outputs
├── run.py             # Main CLI interface
└── requirements.txt   # Dependencies
```

## 🧑‍⚕️ Clinical Data

### Sample Documents
- **Achilles_Pain.pdf**: Orthopedic guidelines for Achilles tendinopathy
- **Concussion.pdf**: Clinical protocols for concussion management

### Section Classification
The system automatically identifies and classifies content:
- `treatment`: Therapeutic interventions and protocols
- `diagnosis`: Assessment and diagnostic criteria  
- `methods`: Clinical procedures and techniques
- `introduction`: Background and overview content
- `conclusion`: Recommendations and summaries

## 🎯 Success Metrics

### Technical Performance
- ✅ **Sub-5 second** retrieval response time (`search` command)
- ✅ **Sub-15 second** end-to-end RAG response time (`ask` command)  
- ✅ **0.7+ similarity scores** for clinical relevance
- ✅ **869 embeddings** successfully processed and stored
- ✅ **Professional clinical responses** with evidence citations
- ✅ **Local LLM inference** (no API costs, data privacy)

### Clinical Validation (Phase 1 Target)
- 🔄 **20-30 test scenarios** with ground truth
- 🔄 **Expert validation** from orthopedic PTs
- 🔄 **Clinical workflow** integration assessment

## 🤝 Contributing

This project serves as a ZenML community example. Contributions welcome for:

- Additional clinical document types
- Enhanced evaluation metrics
- Advanced chunking strategies
- Multi-modal capabilities (images, tables)
- Integration with clinical workflows

## 🏥 Healthcare Compliance

**Important**: This system provides **decision support**, not diagnosis. It is designed for:
- Educational purposes
- Clinical research
- Healthcare professional assistance

Always consult qualified healthcare professionals for medical decisions.

## 📚 ZenML Resources

- [ZenML Documentation](https://docs.zenml.io)
- [ZenML Examples](https://github.com/zenml-io/zenml-projects)
- [MLOps Best Practices](https://docs.zenml.io/how-to)

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **ZenML Team** for the excellent MLOps framework
- **Clinical Guidelines** from professional medical societies
- **Open Source Community** for sentence transformers and pgvector

---

*Built with ❤️ using [ZenML](https://zenml.io) for robust MLOps in healthcare AI*