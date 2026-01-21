"""Data ingestion script for Federal Register API."""

import logging
import os
import math
import re
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import json
from pydantic import ValidationError
from openai import OpenAI
from dotenv import load_dotenv

from src.models import RegulationDocument

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Initialize OpenAI client
openai_client = None

def get_openai_client():
    """Get or initialize OpenAI client."""
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        openai_client = OpenAI(api_key=api_key)
    return openai_client


def check_ai_centricity(abstract: str) -> Optional[Dict[str, any]]:
    """
    Use GPT-4o-mini to determine if a document is primarily about AI regulation.
    Deterministic binary classifier with structured reasoning and confidence scoring.
    
    Args:
        abstract: Document abstract text.
    
    Returns:
        Dict with:
        - 'is_ai' (bool): True if primarily AI regulation
        - 'confidence' (float 0.0-1.0): Model confidence based on logprobs
        - 'reasoning' (str): Full structured reasoning
        - 'version_fingerprint' (str): Model version fingerprint for reproducibility
        - 'status' (str): 'Approved' or 'Low Confidence' (for manual review)
        
        Returns None if abstract is too short or processing fails.
    """
    if not abstract or len(abstract.strip()) < 50:
        logger.warning("Abstract too short, filtering out")
        return None
    
    try:
        client = get_openai_client()
        
        prompt = f"""You are a legal analyst evaluating whether regulatory documents are PRIMARILY about Artificial Intelligence.

Analyze this regulatory abstract using the following structured format:

**Topic Analysis**: [Briefly summarize the core regulatory subject in 1-2 sentences]

**AI Relevance**: [Explain whether this document specifically regulates the development, deployment, risk management, or governance of AI systems. Be specific about how AI is addressed.]

**Decision**: [[TRUE]] if this is primarily an AI regulation, or [[FALSE]] if AI is secondary, tangential, or not mentioned.

IMPORTANT:
- Use [[TRUE]] ONLY if AI is the main regulatory focus
- Use [[FALSE]] if AI is mentioned as a tool, example, or minor component
- The Decision MUST be exactly [[TRUE]] or [[FALSE]] in double brackets

Abstract:
{abstract[:1200]}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a regulatory compliance analyst. Follow the structured format exactly. End with [[TRUE]] or [[FALSE]]."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            seed=42,
            max_tokens=300,
            logprobs=True,
            top_logprobs=1
        )
        
        reasoning_text = response.choices[0].message.content.strip()
        logprobs_data = response.choices[0].logprobs
        version_fingerprint = response.system_fingerprint or "unknown"
        
        # Extract decision from double brackets
        decision_match = re.search(r'\[\[(TRUE|FALSE)\]\]', reasoning_text, re.IGNORECASE)
        
        if not decision_match:
            logger.error(f"No bracketed decision found in response: {reasoning_text[:100]}...")
            return {
                "is_ai": False,
                "confidence": 0.0,
                "reasoning": "Error: Invalid format",
                "version_fingerprint": version_fingerprint,
                "status": "Error"
            }
        
        decision_text = decision_match.group(1).upper()
        is_ai = (decision_text == "TRUE")
        
        # Calculate confidence score from logprobs using math.exp(logprob)
        # Find the token containing TRUE or FALSE
        confidence = 0.5  # Default
        
        if logprobs_data and logprobs_data.content:
            # Search for the decision token in the logprobs
            for token_data in logprobs_data.content:
                token = token_data.token.upper().strip()
                # Check if this token is part of our decision
                if 'TRUE' in token or 'FALSE' in token:
                    # Extract logprob and convert to linear probability
                    logprob = token_data.logprob
                    confidence = math.exp(logprob)
                    break
        
        # Determine status: flag low confidence TRUE decisions for manual review
        if is_ai and confidence < 0.8:
            status = "Low Confidence"
            logger.warning(f"Low confidence AI classification: confidence={confidence:.3f}")
        else:
            status = "Approved"
        
        logger.debug(f"AI centricity: is_ai={is_ai}, confidence={confidence:.3f}, status={status}")
        
        return {
            "is_ai": is_ai,
            "confidence": confidence,
            "reasoning": reasoning_text,
            "version_fingerprint": version_fingerprint,
            "status": status
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in AI centricity check: {e}")
        return {
            "is_ai": False,
            "confidence": 0.0,
            "reasoning": "Error: JSON parsing failed",
            "version_fingerprint": "unknown",
            "status": "Error"
        }
    except Exception as e:
        logger.error(f"Error checking AI centricity: {e}")
        return {
            "is_ai": False,
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}",
            "version_fingerprint": "unknown",
            "status": "Error"
        }


def create_session_with_retries(retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
    """
    Create a requests Session with retry logic for transient failures.
    
    Args:
        retries: Number of retries for failed requests.
        backoff_factor: Backoff factor for retries.
    
    Returns:
        Configured requests.Session object.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def calculate_relevance_score(title: str, keyword: str = "Artificial Intelligence") -> float:
    """
    Calculate relevance score based on keyword presence in title.
    
    Args:
        title: Document title.
        keyword: Keyword to search for (default: "Artificial Intelligence").
    
    Returns:
        Relevance score (1.0 if keyword in title, 0.5 otherwise).
    """
    if keyword.lower() in title.lower() or "AI" in title:
        return 1.0
    return 0.5




def fetch_federal_register_documents(
    keyword: str = "Artificial Intelligence",
    days_back: int = 7,
    timeout: int = 10,
    significant_only: bool = True
) -> List[Tuple[RegulationDocument, float]]:
    """
    Fetch documents from Federal Register API for the past N days with filters.
    
    Filters:
    - Significant documents (Executive Order 12866)
    - Specific agencies (NIST, NTIA, OMB, EOP)
    - Relevance scoring based on title
    
    Args:
        keyword: Search keyword (default: "Artificial Intelligence").
        days_back: Number of days to look back (default: 7).
        timeout: Request timeout in seconds (default: 10).
        significant_only: Filter for significant documents (default: True).
    
    Returns:
        List of tuples (RegulationDocument, relevance_score).
    
    Raises:
        requests.exceptions.RequestException: If API request fails.
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    date_filter = f"[{start_date.strftime('%Y-%m-%d')} TO {end_date.strftime('%Y-%m-%d')}]"
    
    # Build API URL with significance filter
    base_url = "https://www.federalregister.gov/api/v1/documents.json"
    params = {
        "conditions[publication_date]": date_filter,
        "conditions[term]": keyword,
        "per_page": 1000,
        "page": 1
    }
    
    # Add significance filter for Executive Order 12866
    if significant_only:
        params["conditions[significant]"] = "1"
    
    logger.info(f"Fetching significant Federal Register documents for keyword '{keyword}' from last {days_back} days...")
    
    documents = []
    session = create_session_with_retries()
    all_results = []
    
    try:
        # First request to get total count
        response = session.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        
        # Parse JSON with error handling
        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"Invalid JSON response from API: {e}")
            raise ValueError(f"Failed to parse API response as JSON: {e}")
        
        # Get pagination info
        total_count = data.get("count", 0)
        results = data.get("results", [])
        all_results.extend(results)
        
        logger.info(f"Total documents available: {total_count}")
        logger.info(f"Retrieved page 1: {len(results)} documents")
        
        # Fetch remaining pages if needed
        total_pages = math.ceil(total_count / params["per_page"])
        if total_pages > 1:
            logger.info(f"Fetching {total_pages - 1} more pages...")
            for page in range(2, total_pages + 1):
                params["page"] = page
                response = session.get(base_url, params=params, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                page_results = data.get("results", [])
                all_results.extend(page_results)
                logger.info(f"Retrieved page {page}: {len(page_results)} documents")
        
        logger.info(f"Total retrieved: {len(all_results)} documents from API")
        results = all_results
        
        # Filter and validate each document
        filtered_by_ai_check = 0
        
        for doc in results:
            try:
                # Map API fields to schema
                regulation_doc = RegulationDocument(
                    document_number=doc.get("document_number", ""),
                    title=doc.get("title", ""),
                    type=doc.get("type", ""),
                    abstract=doc.get("abstract"),
                    html_url=doc.get("html_url", ""),
                    publication_date=doc.get("publication_date", ""),
                    agency_names=", ".join([agency.get("name", "") for agency in doc.get("agencies", [])]) if doc.get("agencies") else None
                )
                
                # Check AI centricity using deterministic GPT-4o-mini classifier
                logger.info(f"Checking AI centricity for: {regulation_doc.title[:60]}...")
                ai_check = check_ai_centricity(regulation_doc.abstract)
                
                if not ai_check or not ai_check["is_ai"]:
                    filtered_by_ai_check += 1
                    if ai_check:
                        logger.info(f"  -> Filtered out (is_ai={ai_check['is_ai']}, confidence={ai_check['confidence']:.2f}, status={ai_check['status']})")
                        logger.debug(f"     Reasoning: {ai_check['reasoning'][:100]}...")
                    else:
                        logger.info(f"  -> Filtered out (check failed)")
                    continue
                
                # Log status and confidence
                status_emoji = "⚠" if ai_check['status'] == "Low Confidence" else "✓"
                logger.info(f"  -> {status_emoji} AI-centric (confidence={ai_check['confidence']:.2f}, status={ai_check['status']})")
                logger.debug(f"     Reasoning: {ai_check['reasoning'][:150]}...")
                logger.debug(f"     Version: {ai_check['version_fingerprint']}")
                
                # Use confidence score as relevance score (0.0-1.0)
                relevance_score = ai_check["confidence"]
                
                # Store AI metadata for database
                ai_metadata = {
                    'is_ai': ai_check['is_ai'],
                    'reasoning': ai_check['reasoning'],
                    'confidence': ai_check['confidence']
                }
                
                documents.append((regulation_doc, relevance_score, ai_metadata))
                
            except ValidationError as e:
                logger.warning(f"Skipping invalid document {doc.get('document_number', 'UNKNOWN')}: {e}")
                continue
        
        logger.info(f"\nFiltering Summary:")
        logger.info(f"  - Total documents from API: {len(results)}")
        logger.info(f"  - Filtered by AI centricity check: {filtered_by_ai_check}")
        logger.info(f"  - Final documents passing all filters: {len(documents)}")
        
        # Log relevance score distribution
        high_relevance = sum(1 for _, score, _ in documents if score == 1.0)
        low_relevance = sum(1 for _, score, _ in documents if score == 0.5)
        logger.info(f"\nRelevance scores - High (1.0): {high_relevance}, Medium (0.5): {low_relevance}")
        
        return documents
    
    except requests.exceptions.Timeout:
        logger.error(f"API request timed out after {timeout} seconds")
        raise
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error when contacting API: {e}")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error from API: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching documents: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    try:
        docs_with_scores = fetch_federal_register_documents()
        logger.info(f"\nTop 5 documents by relevance:")
        # Sort by relevance score
        sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        for doc, score in sorted_docs[:5]:
            print(f"\n[Score: {score}] {doc.document_number}: {doc.title[:80]}...")
    except Exception as e:
        logger.error(f"Failed to fetch documents: {e}")
