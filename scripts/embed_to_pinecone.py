"""
Generate embeddings for AI-filtered regulations and upload to Pinecone.
Optimized with batch processing and comprehensive error handling.
"""

import os
import logging
import sys
from typing import List, Dict, Any, Tuple
from pathlib import Path
import duckdb
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "regusense-index")
EMBEDDING_BATCH_SIZE = 15  # Optimal batch size for OpenAI API (10-20)
UPSERT_BATCH_SIZE = 100    # Batch size for Pinecone upsert
DB_PATH = Path(__file__).parent.parent / "data" / "regu_sense.duckdb"

# Cost tracking
EMBEDDING_COST_PER_1K_TOKENS = 0.00002  # text-embedding-3-small pricing


class AIRegulationEmbedder:
    """Generate embeddings for AI-filtered regulations and sync to Pinecone."""
    
    def __init__(self):
        """Initialize the embedder service."""
        self.openai_client = None
        self.pinecone_client = None
        self.index = None
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Validate API keys
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is required")
    
    def connect_services(self):
        """Connect to OpenAI and Pinecone services."""
        logger.info("Connecting to OpenAI and Pinecone services...")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize Pinecone client
        self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        
        # Setup Pinecone index
        self._setup_index()
        
        logger.info("✓ Connected to services successfully")
    
    def _setup_index(self):
        """Create Pinecone serverless index if it doesn't exist."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            
            if INDEX_NAME not in existing_indexes:
                logger.info(f"Creating new Pinecone serverless index: {INDEX_NAME}")
                self.pinecone_client.create_index(
                    name=INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=PINECONE_ENVIRONMENT
                    )
                )
                logger.info(f"✓ Created index: {INDEX_NAME}")
                
                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                time.sleep(10)
            else:
                logger.info(f"✓ Index already exists: {INDEX_NAME}")
            
            # Connect to the index
            self.index = self.pinecone_client.Index(INDEX_NAME)
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {e}")
            raise
    
    def fetch_ai_regulations(self) -> List[Dict[str, Any]]:
        """
        Fetch AI-filtered regulations from DuckDB.
        
        Returns:
            List of regulation dictionaries.
        """
        logger.info(f"Fetching AI-filtered regulations from {DB_PATH}...")
        
        try:
            conn = duckdb.connect(str(DB_PATH))
            
            # Query for AI-filtered regulations only
            query = """
            SELECT 
                document_number,
                title,
                abstract,
                html_url as url,
                publication_date,
                type,
                agency_names,
                ai_reasoning,
                ai_confidence
            FROM raw_regulations
            WHERE is_ai = TRUE
            ORDER BY publication_date DESC
            """
            
            results = conn.execute(query).fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            regulations = []
            for row in results:
                regulations.append({
                    "document_number": row[0],
                    "title": row[1],
                    "abstract": row[2] or "",
                    "url": row[3],
                    "publication_date": str(row[4]),
                    "type": row[5] or "Rule",
                    "agency_names": row[6] or "Unknown",
                    "ai_reasoning": row[7] or "",
                    "ai_confidence": float(row[8]) if row[8] else 0.0
                })
            
            logger.info(f"✓ Fetched {len(regulations)} AI-filtered regulations")
            return regulations
            
        except Exception as e:
            logger.error(f"Error fetching regulations from DuckDB: {e}")
            raise
    
    def create_searchable_text(self, regulation: Dict[str, Any]) -> str:
        """
        Create searchable string by combining title and abstract.
        
        Args:
            regulation: Regulation dictionary.
        
        Returns:
            Combined searchable text.
        """
        title = regulation.get("title", "")
        abstract = regulation.get("abstract", "")
        
        # Combine with clear separator
        searchable_text = f"{title}\n\n{abstract}"
        
        return searchable_text.strip()
    
    def generate_embeddings_batch(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Generate embeddings for a batch of texts using OpenAI.
        
        Args:
            texts: List of text strings to embed.
        
        Returns:
            Tuple of (embeddings list, tokens used).
        """
        try:
            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            
            # Extract embeddings and token count
            embeddings = [item.embedding for item in response.data]
            tokens_used = response.usage.total_tokens
            
            return embeddings, tokens_used
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def prepare_vectors(self, regulations: List[Dict[str, Any]]) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        """
        Generate embeddings for all regulations with batching optimization.
        
        Args:
            regulations: List of regulation dictionaries.
        
        Returns:
            List of tuples (id, embedding, metadata).
        """
        logger.info(f"Generating embeddings for {len(regulations)} regulations...")
        logger.info(f"Using batch size: {EMBEDDING_BATCH_SIZE}")
        
        vectors = []
        
        # Process in batches
        for i in tqdm(range(0, len(regulations), EMBEDDING_BATCH_SIZE), desc="Generating embeddings"):
            batch = regulations[i:i + EMBEDDING_BATCH_SIZE]
            
            # Create searchable texts for batch
            batch_texts = [self.create_searchable_text(reg) for reg in batch]
            
            # Generate embeddings for batch
            try:
                embeddings, tokens_used = self.generate_embeddings_batch(batch_texts)
                self.total_tokens += tokens_used
                
                # Create vector tuples with metadata
                for reg, embedding in zip(batch, embeddings):
                    vector_id = reg["document_number"]
                    
                    metadata = {
                        "title": reg["title"][:500],  # Limit metadata size
                        "url": reg["url"],
                        "agency": reg["agency_names"][:200],
                        "publication_date": reg["publication_date"],
                        "type": reg["type"],
                        "abstract_preview": reg["abstract"][:300] if reg["abstract"] else "",
                        "ai_reasoning": reg["ai_reasoning"][:500] if reg["ai_reasoning"] else "",
                        "ai_confidence": reg["ai_confidence"]
                    }
                    
                    vectors.append((vector_id, embedding, metadata))
                
            except Exception as e:
                logger.warning(f"Failed to process batch {i//EMBEDDING_BATCH_SIZE + 1}: {e}")
                continue
        
        # Calculate cost
        self.total_cost = (self.total_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
        
        logger.info(f"✓ Generated {len(vectors)} embeddings")
        logger.info(f"  Total tokens: {self.total_tokens:,}")
        logger.info(f"  Total cost: ${self.total_cost:.4f}")
        
        return vectors
    
    def upsert_to_pinecone(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]):
        """
        Upsert vectors to Pinecone with batch processing and error handling.
        
        Args:
            vectors: List of tuples (id, embedding, metadata).
        """
        if not vectors:
            logger.warning("No vectors to upsert")
            return
        
        logger.info(f"Upserting {len(vectors)} vectors to Pinecone...")
        logger.info(f"Using batch size: {UPSERT_BATCH_SIZE}")
        
        successful_upserts = 0
        failed_upserts = 0
        
        # Process in batches
        for i in tqdm(range(0, len(vectors), UPSERT_BATCH_SIZE), desc="Upserting vectors"):
            batch = vectors[i:i + UPSERT_BATCH_SIZE]
            
            try:
                # Upsert batch to Pinecone
                self.index.upsert(vectors=batch)
                successful_upserts += len(batch)
                
            except Exception as e:
                logger.error(f"Failed to upsert batch {i//UPSERT_BATCH_SIZE + 1}: {e}")
                failed_upserts += len(batch)
                
                # Try individual upserts as fallback
                for vector in batch:
                    try:
                        self.index.upsert(vectors=[vector])
                        successful_upserts += 1
                        failed_upserts -= 1
                    except Exception as e2:
                        logger.error(f"Failed to upsert vector {vector[0]}: {e2}")
        
        logger.info(f"✓ Successfully upserted {successful_upserts} vectors")
        if failed_upserts > 0:
            logger.warning(f"⚠ Failed to upsert {failed_upserts} vectors")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
    
    def run(self):
        """Execute the complete embedding and upload pipeline."""
        logger.info("=" * 80)
        logger.info("AI Regulation Embedder - Starting Pipeline")
        logger.info("=" * 80)
        
        try:
            # Step 1: Connect to services
            self.connect_services()
            
            # Step 2: Fetch AI-filtered regulations
            regulations = self.fetch_ai_regulations()
            
            if not regulations:
                logger.warning("No regulations found to process")
                return
            
            # Step 3: Generate embeddings with batching
            vectors = self.prepare_vectors(regulations)
            
            # Step 4: Upsert to Pinecone
            self.upsert_to_pinecone(vectors)
            
            # Step 5: Display final stats
            stats = self.get_index_stats()
            logger.info("")
            logger.info("✓ Pinecone Index Stats:")
            logger.info(f"  - Total vectors: {stats.get('total_vectors', 'Unknown')}")
            logger.info(f"  - Dimension: {stats.get('dimension', 'Unknown')}")
            
            logger.info("")
            logger.info("💰 Cost Summary:")
            logger.info(f"  - Total tokens used: {self.total_tokens:,}")
            logger.info(f"  - Total cost: ${self.total_cost:.4f}")
            
            logger.info("")
            logger.info("=" * 80)
            logger.info("✓ Pipeline completed successfully!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main entry point."""
    try:
        embedder = AIRegulationEmbedder()
        embedder.run()
    except KeyboardInterrupt:
        logger.info("\\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
