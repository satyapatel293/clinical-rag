"""
Vector similarity search functionality for Clinical RAG system.
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from src.database import DatabaseManager


class ClinicalRAGSearcher:
    """
    Search engine for clinical documents using vector similarity.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", database_url: str = None):
        """
        Initialize the search engine.
        
        Args:
            model_name: Sentence transformer model to use
            database_url: Database connection URL
        """
        self.model_name = model_name
        self.model = None  # Lazy load
        if database_url is None:
            database_url = "postgresql://postgres:password@localhost:5432/clinical_rag"
        self.db = DatabaseManager(database_url)
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            print(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Natural language search query
            
        Returns:
            Query embedding as numpy array
        """
        self._load_model()
        return self.model.encode([query], convert_to_numpy=True)[0]
    
    def search_similar_chunks(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        section_filter: Optional[str] = None,
        document_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.
        
        Args:
            query: Natural language search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1)
            section_filter: Filter by section type (e.g., 'treatment', 'diagnosis')
            document_filter: Filter by document filename
            
        Returns:
            List of similar chunks with metadata and similarity scores
        """
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Build SQL query with optional filters
        sql_parts = [
            """
            SELECT 
                c.id as chunk_id,
                c.text,
                c.section_type,
                c.chunk_id as original_chunk_id,
                d.filename,
                d.title as document_title,
                e.embedding <=> %s::vector as cosine_distance,
                (1 - (e.embedding <=> %s::vector)) as similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            JOIN embeddings e ON c.id = e.chunk_id
            """
        ]
        
        conditions = []
        params = [query_embedding.tolist(), query_embedding.tolist()]
        
        # Add filters
        if section_filter:
            conditions.append("c.section_type = %s")
            params.append(section_filter)
        
        if document_filter:
            conditions.append("d.filename = %s")
            params.append(document_filter)
        
        if similarity_threshold > 0:
            conditions.append("(1 - (e.embedding <=> %s::vector)) >= %s")
            params.extend([query_embedding.tolist(), similarity_threshold])
        
        # Add WHERE clause if there are conditions
        if conditions:
            sql_parts.append("WHERE " + " AND ".join(conditions))
        
        # Add ordering and limit
        sql_parts.extend([
            "ORDER BY e.embedding <=> %s::vector",
            f"LIMIT {top_k}"
        ])
        
        params.append(query_embedding.tolist())
        
        final_sql = " ".join(sql_parts)
        
        # Execute search
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(final_sql, params)
                results = cur.fetchall()
                
                # Format results
                formatted_results = []
                for row in results:
                    formatted_results.append({
                        'chunk_id': row[0],
                        'text': row[1],
                        'section_type': row[2],
                        'original_chunk_id': row[3],
                        'filename': row[4],
                        'document_title': row[5],
                        'cosine_distance': float(row[6]),
                        'similarity': float(row[7])
                    })
                
                return formatted_results
    
    def search_by_section(
        self, 
        query: str, 
        section_type: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific section type.
        
        Args:
            query: Search query
            section_type: Section to search in (treatment, diagnosis, etc.)
            top_k: Number of results
            
        Returns:
            Search results from the specified section
        """
        return self.search_similar_chunks(
            query=query,
            top_k=top_k,
            section_filter=section_type
        )
    
    def get_treatment_recommendations(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search specifically for treatment recommendations.
        
        Args:
            query: Treatment-related query
            top_k: Number of recommendations
            
        Returns:
            Treatment-related chunks
        """
        return self.search_by_section(query, "treatment", top_k)
    
    def get_diagnostic_info(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for diagnostic information.
        
        Args:
            query: Diagnosis-related query
            top_k: Number of results
            
        Returns:
            Diagnosis-related chunks
        """
        return self.search_by_section(query, "diagnosis", top_k)
    
    def print_search_results(self, results: List[Dict[str, Any]], show_full_text: bool = False):
        """
        Pretty print search results.
        
        Args:
            results: Search results from search methods
            show_full_text: Whether to show full text or preview
        """
        if not results:
            print("No results found.")
            return
        
        print(f"\n=== Found {len(results)} results ===\n")
        
        for i, result in enumerate(results, 1):
            print(f"Result {i} (Similarity: {result['similarity']:.3f})")
            print(f"Document: {result['filename']}")
            print(f"Section: {result['section_type']}")
            
            text = result['text']
            if not show_full_text and len(text) > 200:
                text = text[:200] + "..."
            
            print(f"Content: {text}")
            print("-" * 80)