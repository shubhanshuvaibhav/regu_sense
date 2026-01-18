"""Test script to verify document URLs and data integrity"""
import duckdb
from pathlib import Path

DB_PATH = Path(__file__).parent / "regu_sense.duckdb"

def test_document_data():
    """Test document numbers and URLs"""
    conn = duckdb.connect(str(DB_PATH))
    
    print("=" * 80)
    print("Testing Document Data Integrity")
    print("=" * 80)
    
    # Check raw data first
    print("\n1. Checking RAW data (raw_regulations table):")
    raw_query = """
    SELECT document_number, title, html_url 
    FROM raw.raw_regulations 
    WHERE document_number IS NOT NULL
    LIMIT 3
    """
    
    try:
        raw_results = conn.execute(raw_query).fetchall()
        for doc_num, title, url in raw_results:
            print(f"\nDoc Number: {doc_num}")
            print(f"Title: {title[:80]}")
            print(f"URL from API: {url}")
    except Exception as e:
        print(f"Error reading raw data: {e}")
    
    # Check gold layer data
    print("\n\n2. Checking GOLD layer data (fct_regulatory_updates):")
    gold_query = """
    SELECT document_number, title, risk_level
    FROM analytics_analytics.fct_regulatory_updates 
    WHERE risk_level = 'High'
    LIMIT 3
    """
    
    try:
        gold_results = conn.execute(gold_query).fetchall()
        for doc_num, title, risk in gold_results:
            constructed_url = f"https://www.federalregister.gov/documents/{doc_num}"
            print(f"\nDoc Number: {doc_num}")
            print(f"Title: {title[:80]}")
            print(f"Risk: {risk}")
            print(f"Constructed URL: {constructed_url}")
    except Exception as e:
        print(f"Error reading gold data: {e}")
    
    # Check Pinecone metadata
    print("\n\n3. Testing Pinecone metadata:")
    from pinecone import Pinecone
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "regulatory-documents"))
    
    # Fetch a specific vector by ID
    try:
        fetch_results = index.fetch(ids=['2024-21635', '2024-21979', '2024-18596'])
        print("\nFetching specific vectors by ID:")
        for vec_id, vec_data in fetch_results.vectors.items():
            print(f"\nVector ID: {vec_id}")
            if vec_data.metadata:
                print(f"  URL in metadata: {vec_data.metadata.get('url', 'MISSING!')}")
                print(f"  Title: {vec_data.metadata.get('title', 'N/A')[:80]}")
                print(f"  Risk: {vec_data.metadata.get('risk_level', 'N/A')}")
            else:
                print("  NO METADATA FOUND!")
    except Exception as e:
        print(f"Error fetching vectors: {e}")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_document_data()
