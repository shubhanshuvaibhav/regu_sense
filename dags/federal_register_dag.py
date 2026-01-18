from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion import fetch_federal_register_documents
from database import ingest_and_store
from models import RegulationDocument


logger = logging.getLogger(__name__)


# Default DAG arguments
default_args = {
    "owner": "data_engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}


@dag(
    dag_id="federal_register_ingestion",
    default_args=default_args,
    description="Daily ingestion of Federal Register documents with AI keyword filter",
    schedule_interval="0 8 * * *",  # Run daily at 8 AM UTC
    start_date=days_ago(1),
    catchup=False,
    tags=["data-engineering", "federal-register", "ai"],
)
def federal_register_ingestion_dag():

    
    @task(
        task_id="fetch_documents",
        retries=3,
        pool="api_requests",  # Limit concurrent API requests
    )
    def fetch_documents_task() -> str:
       
        try:
            logger.info("Starting document fetch from Federal Register API")
            documents = fetch_federal_register_documents(
                keyword="Artificial Intelligence",
                days_back=7,
                timeout=15
            )
            logger.info(f"Successfully fetched {len(documents)} documents")
            
            # Serialize documents to JSON for Airflow task communication
            import json
            doc_dicts = [doc.model_dump(mode='json') for doc in documents]
            return json.dumps(doc_dicts)
        
        except Exception as e:
            logger.error(f"Failed to fetch documents: {e}")
            raise
    
    @task(
        task_id="validate_and_store",
        retries=2,
    )
    def validate_and_store_task(documents_json: str) -> dict:
        try:
            import json
            
            logger.info("Starting validation and storage task")
            
            # Deserialize documents
            doc_dicts = json.loads(documents_json)
            documents = [
                RegulationDocument(**doc) for doc in doc_dicts
            ]
            
            # Store in DuckDB
            db_path = Path(__file__).parent.parent / "regu_sense.duckdb"
            inserted_count = ingest_and_store(documents, str(db_path))
            
            stats = {
                "total_fetched": len(documents),
                "total_stored": inserted_count,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"Ingestion complete: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Failed to validate and store documents: {e}")
            raise
    
    @task(
        task_id="log_ingestion_summary",
    )
    def log_summary_task(stats: dict) -> None:
        """
        Log ingestion summary.
        
        Args:
            stats: Ingestion statistics dictionary.
        """
        logger.info(
            f"Federal Register ingestion completed. "
            f"Fetched: {stats['total_fetched']}, "
            f"Stored: {stats['total_stored']}, "
            f"Timestamp: {stats['timestamp']}"
        )
    
    # Define task dependencies
    docs_json = fetch_documents_task()
    ingestion_stats = validate_and_store_task(docs_json)
    log_summary_task(ingestion_stats)


# Instantiate the DAG
federal_register_ingestion = federal_register_ingestion_dag()
