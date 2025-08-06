# 🏥 Clinical RAG with ZenML

A production-ready RAG (Retrieval-Augmented Generation) system for clinical decision support, built with ZenML to demonstrate MLOps best practices for LLM applications in healthcare.

[![ZenML](https://img.shields.io/badge/Built%20with-ZenML-blue)](https://zenml.io)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL%20%2B%20pgvector-blue)](https://postgresql.org)

## 📖 What does this project do?

This project demonstrates a complete RAG system for **clinical decision support in physical therapy**, built with ZenML to showcase MLOps best practices. Physical therapists can search clinical guidelines and get AI-powered recommendations for orthopedic conditions like Achilles tendinopathy and concussion management.

The system processes clinical PDF documents through a ZenML pipeline (extract → chunk → embed → store) and provides semantic search with multi-provider LLM generation via both CLI and web interfaces.

**Key ZenML features demonstrated:**
- **Pipeline Steps**: Modular PDF processing, text chunking, and embedding generation
- **Artifact Management**: Automated tracking of processed documents and embeddings  
- **Multi-Provider LLM**: Support for OpenAI, Anthropic, Ollama, and Azure via LiteLLM
- **Configuration Management**: Environment-specific settings for dev/production
- **Database Integration**: PostgreSQL with pgvector for similarity search

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────────────────────────┐    ┌─────────────────┐
│   Clinical PDFs │───▶│           ZenML Pipeline             │───▶│   PostgreSQL    │
│   (Guidelines)  │    │  Extract→Preprocess→Chunk→Embed     │    │   + pgvector    │
└─────────────────┘    └──────────────────────────────────────┘    └─────────────────┘
                                                                            │
                                                                            ▼
                       ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
                       │    User Query   │───▶│  Semantic Search │───▶│   LLM Response  │
                       │  (CLI/Web UI)   │    │   + Context      │    │  (Multi-provider)│
                       └─────────────────┘    └──────────────────┘    └─────────────────┘
```

**Pipeline Flow**: PDF documents are processed through ZenML steps to extract, chunk, and embed text into PostgreSQL with pgvector. Users can then search (`search` command) or get AI responses (`ask` command) through CLI or web interface.

## 🚀 How to run

### Prerequisites
- Python 3.8+
- PostgreSQL with pgvector extension
- At least one LLM provider: OpenAI, Anthropic, Azure, or local Ollama

### Installation

```bash
# Clone and setup
git clone https://github.com/your-org/clinical-rag
cd clinical-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup database
createdb clinical_rag
psql -d clinical_rag -c "CREATE EXTENSION vector;"
psql -d clinical_rag -f schema.sql

# Initialize ZenML
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # macOS only
zenml init

# Setup LLM provider (choose one):
export OPENAI_API_KEY="your-key"        # OpenAI
export ANTHROPIC_API_KEY="your-key"     # Anthropic
# OR install Ollama for local models
ollama serve && ollama pull llama3.2:3b
```

### Usage

```bash
# Process clinical documents
python run.py ingest data/raw/Achilles_Pain.pdf

# Search for information (retrieval only)
python run.py search "What exercises help with Achilles tendon pain?"

# Ask questions with AI responses (full RAG)
python run.py ask "What exercises help with Achilles tendon pain?" --model openai/gpt-4o

# Launch web interface
streamlit run app.py  # Then open http://localhost:8501

# Run tests
python tests/test_pipeline.py
```


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

The project supports environment-specific configurations in `configs/` directory and key environment variables:

```bash
# Required for ZenML on macOS  
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Optional: Custom database URL
export DATABASE_URL="postgresql://user:pass@localhost:5432/clinical_rag"
```

## 📁 Project Structure

```
clinical-rag/
├── pipelines/           # ZenML pipeline definitions
├── steps/              # Individual pipeline steps 
├── utils/              # Database, search, and config utilities
├── tests/              # Test suite
├── configs/            # Environment configurations
├── data/raw/           # Original clinical PDFs
├── app.py             # Streamlit web interface
├── run.py             # Main CLI interface
└── requirements.txt   # Dependencies
```

## 📋 Sample Data

The project includes clinical guidelines for Achilles tendinopathy and concussion management. The system automatically classifies content into sections (treatment, diagnosis, methods) for better retrieval accuracy.

## ✨ Key Features

- **Fast retrieval**: Sub-5 second semantic search with clinical context
- **Multi-provider LLM**: Support for OpenAI, Anthropic, Ollama, and Azure
- **Dual interface**: Both CLI commands and Streamlit web UI
- **Clinical accuracy**: Section-aware chunking and 0.7+ similarity scoring
- **Production ready**: PostgreSQL + pgvector with comprehensive testing
- **ZenML showcase**: Demonstrates MLOps best practices for LLM applications

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