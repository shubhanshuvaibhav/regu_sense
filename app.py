
import os
import sys
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import duckdb
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from security_utils import (
    validate_user_input,
    sanitize_input,
    check_api_keys,
    log_security_event,
    CostTracker,
    RateLimiter,
    format_cost,
    get_budget_status
)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "regusense-index")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "regu_sense.duckdb")
MAX_COST = float(os.getenv("MAX_OPENAI_COST", "0.50"))

# Validate API keys on startup
is_valid, missing = check_api_keys()
if not is_valid:
    st.error(f"Missing required environment variables: {', '.join(missing)}")
    st.stop()

# Initialize clients (with caching)
@st.cache_resource
def get_openai_client():
    """Initialize and cache OpenAI client."""
    return OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def get_pinecone_index():
    """Initialize and cache Pinecone index."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

# Initialize session state for cost tracking and rate limiting
if 'cost_tracker' not in st.session_state:
    st.session_state.cost_tracker = CostTracker()

if 'rate_limiter' not in st.session_state:
    st.session_state.rate_limiter = RateLimiter(max_requests=10, time_window=3600)

def embed_query(query: str) -> List[float]:
    """Generate embedding for user query with cost tracking."""
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    
    # Track cost
    cost = st.session_state.cost_tracker.track_embedding(response.usage.total_tokens)
    log_security_event("API_CALL", {
        "type": "embedding",
        "tokens": response.usage.total_tokens,
        "cost": cost
    }, level="INFO")
    
    return response.data[0].embedding

def search_pinecone(query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
    """Search Pinecone for relevant documents."""
    index = get_pinecone_index()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches

def generate_answer(query: str, contexts: List[Dict[str, Any]]) -> str:
    """Generate answer using GPT-4o with retrieved context."""
    client = get_openai_client()
    
    # Format context from retrieved documents
    context_text = "\n\n".join([
        f"Document {i+1}:\nTitle: {doc['metadata']['title']}\n"
        f"Agency: {doc['metadata'].get('agency', 'Unknown')}\n"
        f"Publication Date: {doc['metadata']['publication_date']}\n"
        f"Content: {doc['metadata']['abstract_preview']}\n"
        f"URL: {doc['metadata']['url']}"
        for i, doc in enumerate(contexts)
    ])
    
    # Create prompt
    system_prompt = """You are a regulatory compliance assistant. Your role is to provide accurate, 
    clear, and actionable answers based on the provided regulatory documents. 
    
    Guidelines:
    - Only use information from the provided documents
    - Be specific and cite which documents support your answer
    - If the documents don't contain relevant information, say so
    - Highlight any high-risk or compliance-critical information
    - Be concise but thorough"""
    
    user_prompt = f"""Based on these regulatory documents:

{context_text}

Question: {query}

Provide a comprehensive answer with specific references to the documents."""
    
    # Get response from GPT-4o with timeout and cost tracking
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1000,
        timeout=30.0  # 30 second timeout
    )
    
    # Track cost
    cost = st.session_state.cost_tracker.track_gpt4o(
        response.usage.prompt_tokens,
        response.usage.completion_tokens
    )
    log_security_event("API_CALL", {
        "type": "gpt4o",
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "cost": cost
    }, level="INFO")
    
    return response.choices[0].message.content

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_latest_high_risk_updates() -> List[Dict[str, Any]]:
    """Fetch latest AI-filtered updates from DuckDB."""
    conn = duckdb.connect(DB_PATH)
    try:
        query = """
        SELECT 
            title,
            publication_date,
            html_url,
            agency_names
        FROM raw_regulations
        WHERE is_ai = TRUE
        ORDER BY publication_date DESC
        LIMIT 5
        """
        results = conn.execute(query).fetchall()
        return [
            {
                "title": r[0],
                "date": r[1],
                "url": r[2],
                "agency": r[3] or "Unknown"
            }
            for r in results
        ]
    finally:
        conn.close()

def format_date(date_str: str) -> str:
    """Format date string for display."""
    try:
        date_obj = datetime.fromisoformat(str(date_str).replace('Z', '+00:00'))
        return date_obj.strftime("%B %d, %Y")
    except:
        return str(date_str)

# Page configuration
st.set_page_config(
    page_title="ReguSense AI | Regulatory Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "ReguSense AI - Intelligent Regulatory Compliance Assistant"
    }
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Header with Gradient */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Source Cards with Modern Design */
    .source-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Badge Styles */
    .high-risk-badge {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.3);
    }
    
    .medium-risk-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.3);
    }
    
    /* Sidebar Updates - Dark Theme */
    .sidebar-update {
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 3px solid #ef4444;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        transition: all 0.2s;
    }
    
    .sidebar-update:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        transform: translateX(2px);
        background: linear-gradient(135deg, #3b4f66 0%, #253245 100%);
    }
    
    /* Question Section */
    .question-section {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Answer Box */
    .answer-box {
        background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
        line-height: 1.7;
        color: #1e293b;
        font-size: 1rem;
    }
    
    /* Sidebar Styling - Dark Theme */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e2e8f0 !important;
    }
    
    section[data-testid="stSidebar"] p {
        color: #94a3b8 !important;
    }
    
    /* Metric Cards - Dark Theme */
    .metric-card {
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        text-align: center;
        border: 1px solid #475569;
    }
    
    /* Links */
    a {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
    }
    
    a:hover {
        color: #764ba2;
        text-decoration: underline;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: border-color 0.2s;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">ReguSense AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Regulatory Compliance Assistant • Powered by GPT-4o & Federal Register</div>', unsafe_allow_html=True)

# Sidebar - Latest High-Risk Updates
with st.sidebar:
    st.markdown("")
    st.markdown("<h2 style='text-align: center; color: #667eea; font-size: 1.5rem; margin-bottom: 0.5rem;'>High-Risk Alerts</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem; margin-bottom: 1.5rem;'>Latest critical regulatory updates</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    try:
        high_risk_updates = get_latest_high_risk_updates()
        
        if high_risk_updates:
            for update in high_risk_updates:
                with st.container():
                    st.markdown(f"""
                    <div class="sidebar-update">
                        <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem; font-weight: 500;">
                            {format_date(update['date'])}
                        </div>
                        <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 0.75rem; color: #e2e8f0; line-height: 1.4;">
                            {update['title'][:100]}{'...' if len(update['title']) > 100 else ''}
                        </div>
                        <a href="{update['url']}" 
                           target="_blank" style="font-size: 0.85rem; color: #818cf8; text-decoration: none; font-weight: 500;">
                            View Document →
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No high-risk updates available")
            
    except Exception as e:
        st.error(f"Error loading updates: {e}")
    
    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #667eea; font-size: 1.2rem; margin: 1rem 0;'>Database Insights</h3>", unsafe_allow_html=True)
    try:
        conn = duckdb.connect(DB_PATH)
        total_docs = conn.execute("SELECT COUNT(*) FROM raw_regulations").fetchone()[0]
        ai_filtered = conn.execute("SELECT COUNT(*) FROM raw_regulations WHERE is_ai = TRUE").fetchone()[0]
        recent_docs = conn.execute("SELECT COUNT(*) FROM raw_regulations WHERE publication_date > DATE '2024-01-01'").fetchone()[0]
        conn.close()
        
        # Display metrics in a nice format
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; font-weight: 700; color: #818cf8;">{total_docs:,}</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem;">Total Documents</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; font-weight: 700; color: #10b981;">{ai_filtered}</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem;">AI-Filtered</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #334155 0%, #1e293b 100%); padding: 0.75rem; border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.3); border: 1px solid #475569;">
            <span style="color: #fbbf24; font-weight: 600; font-size: 1.5rem;">{recent_docs}</span>
            <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 0.5rem;">Recent (2024+)</span>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load database insights: {e}")

# Main content area with better styling
st.markdown("")
st.markdown("### Ask Your Regulatory Question")
st.markdown("<p style='color: #64748b; margin-bottom: 1.5rem;'>Get instant insights from AI-filtered regulatory documents</p>", unsafe_allow_html=True)

# Query input with better design
user_query = st.text_input(
    "Enter your question:",
    placeholder="e.g., What are the latest AI regulations and enforcement penalties?",
    help="Ask about regulations, compliance requirements, penalties, enforcement actions, or specific industries",
    label_visibility="collapsed",
    max_chars=500  # Enforce max length at UI level
)

# Search button with better layout
col1, col2, col3 = st.columns([2, 2, 8])
with col1:
    search_button = st.button("Search & Analyze", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.rerun()

# Process query when button clicked
if search_button and user_query:
    # Step 1: Validate input
    is_valid, validation_message = validate_user_input(user_query)
    if not is_valid:
        st.error(f"❌ {validation_message}")
        log_security_event("INPUT_VALIDATION_FAILED", {
            "reason": validation_message,
            "query_length": len(user_query)
        }, level="WARNING")
    else:
        # Step 2: Check rate limit
        allowed, seconds_until_reset = st.session_state.rate_limiter.check_rate_limit()
        if not allowed:
            st.error(f"❌ Rate limit exceeded. Please try again in {seconds_until_reset} seconds.")
            log_security_event("RATE_LIMIT_EXCEEDED", {
                "max_requests": st.session_state.rate_limiter.max_requests,
                "time_window": st.session_state.rate_limiter.time_window
            }, level="WARNING")
        else:
            # Step 3: Check budget
            budget_status = get_budget_status(
                st.session_state.cost_tracker.total_cost,
                MAX_COST
            )
            if budget_status['status'] == 'exceeded':
                st.error(f"❌ Budget exceeded: ${budget_status['current']:.4f} / ${budget_status['max']:.2f}")
                st.info("Please contact your administrator to increase the budget.")
            else:
                # Record request
                st.session_state.rate_limiter.record_request()
                
                # Sanitize input
                sanitized_query = sanitize_input(user_query)
                
                try:
                    with st.spinner("Searching regulatory database..."):
                        # Step 4: Embed query
                        query_embedding = embed_query(sanitized_query)
                        
                        # Step 5: Search Pinecone
                        search_results = search_pinecone(query_embedding, top_k=3)
                        
                        if not search_results:
                            st.warning("No relevant documents found. Try rephrasing your question.")
                        else:
                            # Step 6: Generate answer
                            with st.spinner("Generating answer..."):
                                answer = generate_answer(sanitized_query, search_results)
                            
                            # Display answer with better styling
                            st.markdown("")
                            st.markdown("### AI Analysis & Answer")
                            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                            
                            # Display sources with better styling
                            st.markdown("")
                            st.markdown("### Source Documents")
                            st.markdown(f"<p style='color: #64748b; font-size: 0.95rem; margin-bottom: 1rem;'>Analysis based on <strong>{len(search_results)}</strong> most relevant regulatory documents</p>", unsafe_allow_html=True)
                            
                            for i, result in enumerate(search_results, 1):
                                metadata = result['metadata']
                                score = result['score']
                                
                                # Agency badge
                                agency_display = metadata.get('agency', 'Unknown Agency')
                                
                                st.markdown(f"""
                                <div class="source-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                        <strong style="color: #1e293b; font-size: 1rem;">Source {i}</strong>
                                        <span class="high-risk-badge">{agency_display}</span>
                                    </div>
                                    <div style="margin-bottom: 0.75rem; color: #1e293b; font-size: 1.05rem; font-weight: 600; line-height: 1.5;">
                                        {metadata['title']}
                                    </div>
                                    <div style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.75rem; font-weight: 500;">
                                        Published: {format_date(metadata['publication_date'])} | Relevance: {score:.1%}
                                    </div>
                                    <div style="font-size: 0.95rem; margin-bottom: 0.75rem; color: #475569; line-height: 1.6;">
                                        {metadata['abstract_preview'][:250]}...
                                    </div>
                                    <a href="{metadata['url']}" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 500; font-size: 0.9rem;">
                                        View Full Document →
                                    </a>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display cost and rate limit information after search
                            st.markdown("")
                            summary = st.session_state.cost_tracker.get_summary()
                            budget_status = get_budget_status(summary['total_cost'], MAX_COST)
                            allowed, wait_seconds = st.session_state.rate_limiter.check_rate_limit()
                            rate_message = f"{len(st.session_state.rate_limiter.request_times)}/{st.session_state.rate_limiter.max_requests} queries used"
                            
                            st.markdown(f"""
                            <div style="background: #1e293b; padding: 1rem; border-radius: 8px; border: 1px solid #334155; margin-top: 1rem;">
                                <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem;">💰 Session Usage</div>
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="color: #cbd5e1;">Total Cost: {format_cost(summary['total_cost'])}</span>
                                    <span style="color: #64748b; font-size: 0.85rem;">{budget_status['percentage']:.1f}% of ${MAX_COST:.2f} budget</span>
                                </div>
                                <div style="margin-top: 0.5rem; color: #64748b; font-size: 0.8rem;">
                                    Tokens: {summary['total_tokens']:,} | Queries: {summary['request_count']} | {rate_message}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    log_security_event("QUERY_ERROR", {
                        "error": str(e),
                        "query_length": len(sanitized_query) if 'sanitized_query' in locals() else 0
                    }, level="ERROR")

elif search_button:
    st.warning("Please enter a question first")

# Example questions with better design
with st.expander("Example Questions to Get Started", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **AI Development & Deployment**
        - What are the new AI regulations and penalties?
        - Are there enforcement actions for AI systems?
        - What are reporting requirements for advanced AI models?
        
        **AI Safety & Ethics**
        - What regulations cover AI safety testing?
        - Are there requirements for AI transparency?
        - What are the rules for AI in critical infrastructure?
        """)
    with col2:
        st.markdown("""
        **AI in Specific Sectors**
        - What are AI regulations for financial services?
        - How is AI regulated in healthcare?
        - What are the rules for AI in government systems?
        
        **AI Data & Privacy**
        - What are data requirements for training AI models?
        - How do privacy laws affect AI systems?
        - What are the rules for automated decision-making?
        
        **General Compliance**
        - What enforcement actions have been taken recently?
        - What are high-risk regulatory changes?
        """)

# Footer with modern design
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0; color: #94a3b8;'>
    <div style='margin-bottom: 0.75rem;'>
        <strong style='color: #667eea;'>ReguSense AI</strong> | Intelligent Regulatory Compliance
    </div>
    <div style='font-size: 0.85rem;'>
        Powered by <strong>OpenAI GPT-4o</strong> • <strong>Pinecone Vector DB</strong> • <strong>Federal Register API</strong>
    </div>
    <div style='font-size: 0.8rem; margin-top: 0.5rem;'>
        Data updated daily via Apache Airflow • AI-filtered regulatory documents indexed
    </div>
</div>
""", unsafe_allow_html=True)
