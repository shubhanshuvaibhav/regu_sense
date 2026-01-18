"""Standalone script to test the ingestion pipeline."""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import fetch_federal_register_documents
from src.database import RegulatoryDatabase
from src.models import RegulationDocument


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete ingestion pipeline locally."""
    
    try:
        logger.info("=" * 80)
        logger.info("ReguSense AI - Federal Register Data Ingestion Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Fetch documents from API
        logger.info("\n[STEP 1] Fetching documents from Federal Register API...")
        documents_with_scores = fetch_federal_register_documents(
            keyword="Artificial Intelligence",
            days_back=30,
            timeout=15
        )
        logger.info(f"✓ Fetched {len(documents_with_scores)} AI-centric documents")
        
        if not documents_with_scores:
            logger.warning("No documents fetched. Exiting.")
            return
        
        # Step 2: Show sample documents
        logger.info("\n[STEP 2] Sample AI-centric documents:")
        for doc, score, ai_metadata in documents_with_scores[:3]:
            logger.info(f"  - [{score}] {doc.document_number}: {doc.title[:60]}...")
            logger.info(f"    Publication Date: {doc.publication_date}")
            logger.info(f"    AI Confidence: {ai_metadata['confidence']:.2f}")
        
        # Step 3: Store in DuckDB
        logger.info("\n[STEP 3] Storing documents in DuckDB...")
        db_path = Path(__file__).parent.parent / "data" / "regu_sense.duckdb"
        db = RegulatoryDatabase(str(db_path))
        
        try:
            db.connect()
            db.create_table()
            inserted_count = db.insert_documents(documents_with_scores)
            logger.info(f"✓ Inserted {inserted_count}/{len(documents_with_scores)} documents")
            
            # Step 4: Query and display recent documents
            logger.info("\n[STEP 4] Recent documents in database:")
            recent = db.query_recent_documents(limit=5)
            for row in recent:
                logger.info(f"  - {row['document_number']}: {row['title'][:50]}...")
            
            total_count = db.get_document_count()
            logger.info(f"\n✓ Total documents in database: {total_count}")
        
        finally:
            db.close()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Pipeline execution completed successfully!")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"✗ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
