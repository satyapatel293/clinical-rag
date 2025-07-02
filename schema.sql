-- Clinical RAG Database Schema
-- PostgreSQL with pgvector extension

-- Ensure pgvector extension is available
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table - stores PDF metadata and processing status
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    title TEXT,
    author TEXT,
    subject TEXT,
    num_pages INTEGER,
    file_size_bytes INTEGER,
    original_text_length INTEGER,
    processed_text_length INTEGER,
    total_chunks INTEGER,
    processing_status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, error
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunks table - stores text chunks with metadata
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL, -- Sequential ID within document
    text TEXT NOT NULL,
    section_type VARCHAR(50), -- introduction, methods, diagnosis, treatment, etc.
    sentence_start INTEGER,
    sentence_end INTEGER,
    char_length INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique chunk_id per document
    UNIQUE(document_id, chunk_id)
);

-- Embeddings table - stores vector embeddings for chunks
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    embedding vector(384), -- Default sentence transformer dimension
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure one embedding per chunk per model
    UNIQUE(chunk_id, model_name)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section_type ON chunks(section_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name);

-- Vector similarity search index (HNSW for fast approximate search)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_cosine 
    ON embeddings USING hnsw (embedding vector_cosine_ops);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Helper views for common queries
CREATE OR REPLACE VIEW chunk_details AS
SELECT 
    c.id as chunk_id,
    c.chunk_id as chunk_number,
    c.text,
    c.section_type,
    c.char_length,
    d.filename,
    d.title as document_title,
    d.processing_status,
    e.model_name as embedding_model,
    (e.embedding IS NOT NULL) as has_embedding
FROM chunks c
JOIN documents d ON c.document_id = d.id
LEFT JOIN embeddings e ON c.id = e.chunk_id;

-- Sample query functions
CREATE OR REPLACE FUNCTION search_similar_chunks(
    query_embedding vector(384),
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 10
) RETURNS TABLE (
    chunk_id INTEGER,
    filename VARCHAR(255),
    text TEXT,
    section_type VARCHAR(50),
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        d.filename,
        c.text,
        c.section_type,
        1 - (e.embedding <=> query_embedding) AS similarity
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    JOIN embeddings e ON c.id = e.chunk_id
    WHERE 1 - (e.embedding <=> query_embedding) > similarity_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;