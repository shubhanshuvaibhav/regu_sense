"""DuckDB database operations for storing regulation documents."""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import duckdb

from src.models import RegulationDocument


logger = logging.getLogger(__name__)


class RegulatoryDatabase:
    """Handler for DuckDB operations on regulatory documents."""
    
    def __init__(self, db_path: str = "regu_sense.duckdb"):
        """
        Initialize DuckDB connection.
        
        Args:
            db_path: Path to the DuckDB database file.
        """
        self.db_path = db_path
        self.conn = None
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        Connect to the DuckDB database.
        
        Returns:
            DuckDB connection object.
        """
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"Connected to DuckDB at {self.db_path}")
            return self.conn
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise
    
    def close(self):
        """Close the DuckDB connection."""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from DuckDB")
    
    def create_table(self):
        """Create the raw_regulations table if it doesn't exist."""
        try:
            if not self.conn:
                self.connect()
            
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS raw_regulations (
                document_number VARCHAR PRIMARY KEY,
                title VARCHAR NOT NULL,
                type VARCHAR,
                abstract VARCHAR,
                html_url VARCHAR,
                publication_date TIMESTAMP,
                agency_names VARCHAR,
                relevance_score FLOAT DEFAULT 0.5,
                is_ai BOOLEAN DEFAULT FALSE,
                ai_reasoning TEXT,
                ai_confidence FLOAT,
                ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            self.conn.execute(create_table_sql)
            logger.info("Created or verified 'raw_regulations' table")
        
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise
    
    def insert_documents(self, documents: List[Tuple[RegulationDocument, float, Dict[str, Any]]]) -> int:
        """
        Insert validated documents with relevance scores and AI metadata into the database.
        
        Args:
            documents: List of tuples (RegulationDocument, relevance_score, ai_metadata).
        
        Returns:
            Number of documents successfully inserted.
        """
        try:
            if not self.conn:
                self.connect()
            
            inserted_count = 0
            
            for doc, relevance_score, ai_metadata in documents:
                try:
                    insert_sql = """
                    INSERT INTO raw_regulations 
                    (document_number, title, type, abstract, html_url, publication_date, agency_names, relevance_score, is_ai, ai_reasoning, ai_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (document_number) DO UPDATE SET
                        title = EXCLUDED.title,
                        type = EXCLUDED.type,
                        abstract = EXCLUDED.abstract,
                        html_url = EXCLUDED.html_url,
                        publication_date = EXCLUDED.publication_date,
                        agency_names = EXCLUDED.agency_names,
                        relevance_score = EXCLUDED.relevance_score,
                        is_ai = EXCLUDED.is_ai,
                        ai_reasoning = EXCLUDED.ai_reasoning,
                        ai_confidence = EXCLUDED.ai_confidence;
                    """
                    
                    self.conn.execute(
                        insert_sql,
                        (
                            doc.document_number,
                            doc.title,
                            doc.type,
                            doc.abstract,
                            doc.html_url,
                            doc.publication_date,
                            doc.agency_names,
                            relevance_score,
                            ai_metadata.get('is_ai', False),
                            ai_metadata.get('reasoning', ''),
                            ai_metadata.get('confidence', 0.0)
                        )
                    )
                    inserted_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to insert document {doc.document_number}: {e}")
                    continue
            
            logger.info(f"Inserted {inserted_count}/{len(documents)} documents into raw_regulations")
            return inserted_count
        
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise
    
    def query_recent_documents(self, limit: int = 10) -> List[dict]:
        """
        Query recent documents from the database.
        
        Args:
            limit: Number of documents to retrieve.
        
        Returns:
            List of document dictionaries.
        """
        try:
            if not self.conn:
                self.connect()
            
            result = self.conn.execute(
                f"""
                SELECT * FROM raw_regulations 
                ORDER BY publication_date DESC 
                LIMIT {limit}
                """
            ).fetchall()
            
            columns = [desc[0] for desc in self.conn.description]
            return [dict(zip(columns, row)) for row in result]
        
        except Exception as e:
            logger.error(f"Failed to query documents: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get total count of documents in the database."""
        try:
            if not self.conn:
                self.connect()
            
            result = self.conn.execute("SELECT COUNT(*) FROM raw_regulations").fetchone()
            return result[0] if result else 0
        
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0


def ingest_and_store(documents: List[Tuple[RegulationDocument, float]], db_path: str = "regu_sense.duckdb") -> int:
    """
    High-level function to store documents with relevance scores in the database.
    
    Args:
        documents: List of tuples (RegulationDocument, relevance_score).
        db_path: Path to DuckDB database.
    
    Returns:
        Number of documents inserted.
    """
    db = RegulatoryDatabase(db_path)
    try:
        db.connect()
        db.create_table()
        count = db.insert_documents(documents)
        return count
    finally:
        db.close()
