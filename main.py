import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional
from functools import lru_cache
import google.generativeai as genai
from dotenv import load_dotenv
import os
import shutil
# from duckduckgo_search import ddg


# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_chroma_db():
    """Clear ChromaDB cache if needed"""
    try:
        cache_dir = "./chroma_db"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info("ChromaDB cache cleared successfully")
    except Exception as e:
        logger.error(f"Failed to clear ChromaDB cache: {e}")

def setup_llm():
    """Initialize LLM client (Gemini)"""
    try:
        gemini_api_key = os.getenv('GEMINI_API_KEY') or st.secrets.get('GOOGLE_API_KEY')
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')
            return model
        else:
            logger.warning("No Gemini API key found")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {str(e)}")
        return None

@st.cache_resource
def init_clients():
    """Initialize search components"""
    try:
        # Initialize ChromaDB with new client configuration
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )
        
        # Create or get collection with proper error handling
        try:
            collection = chroma_client.get_or_create_collection(
                name="search_results",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None  # We'll use sentence_transformers directly
            )
        except Exception as collection_error:
            logger.warning(f"Error with collection, attempting to create new: {collection_error}")
            try:
                chroma_client.delete_collection("search_results")
                collection = chroma_client.create_collection(
                    name="search_results",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None
                )
            except Exception as recreate_error:
                logger.error(f"Failed to recreate collection: {recreate_error}")
                raise
        
        # Initialize Sentence Transformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize LLM
        llm = setup_llm()
        
        return collection, encoder, llm
    except Exception as e:
        logger.error(f"Failed to initialize clients: {str(e)}")
        raise

@lru_cache(maxsize=100)
def search_web(query: str, num_results: int = 5) -> List[Dict]:
    """Search the web using DuckDuckGo with programming context"""
    is_valid, cleaned_query = validate_and_clean_query(query)
    if not is_valid:
        raise ValueError(cleaned_query)
    
    # Add programming context for code-related queries
    if any(term in cleaned_query.lower() for term in ['code', 'program', 'algorithm', 'c++', 'python', 'java']):
        cleaned_query = f"{cleaned_query} programming example code"
    
    results = []
    try:
        with DDGS() as ddgs:
            # Try text search first for programming queries
            text_results = list(ddgs.text(
                cleaned_query,
                region='wt-wt',
                safesearch='off',
                timelimit='y',
                max_results=num_results * 2  # Get more results to filter
            ))
            
            # Combine and deduplicate results
            all_results = []
            seen_urls = set()
            
            for r in text_results:
                if isinstance(r, dict):
                    url = r.get('link', r.get('url', ''))
                    if url and url not in seen_urls:
                        seen_urls.add(url)  
                        # Prioritize programming websites
                        domain_score = 2 if any(site in url.lower() for site in [
                            'github.com', 'stackoverflow.com', 'geeksforgeeks.org',
                            'cplusplus.com', 'programiz.com', 'tutorialspoint.com'
                        ]) else 1
                        
                        title = r.get('title', '').lower()
                        snippet = r.get('body', r.get('snippet', '')).lower()
                        query_terms = cleaned_query.lower().split()
                        
                        relevance_score = sum(
                            2 if term in title else 1 if term in snippet else 0 
                            for term in query_terms
                        ) * domain_score
                        
                        all_results.append({
                            'title': r.get('title', 'No Title'),
                            'url': url,
                            'snippet': r.get('body', r.get('snippet', 'No Snippet')),
                            'relevance_score': relevance_score
                        })
            
            # Sort by relevance score
            results = sorted(all_results, key=lambda x: x['relevance_score'], reverse=True)
            
        if not results:
            logger.warning(f"No relevant results found for query: {cleaned_query}")
            return []
            
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise Exception(f"Failed to perform web search: {str(e)}")
    
    return results[:num_results]

def fetch_content(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch and parse content from a webpage"""
    if not url:
        return None
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'form']):
            element.decompose()
            
        # Extract text with better formatting
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        text = ' '.join(p.get_text().strip() for p in paragraphs)
        return ' '.join(text.split())
        
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch content from {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {str(e)}")
        return None

def process_and_store_content(
    search_results: List[Dict],
    collection,
    encoder,
    chunk_size: int = 1000
) -> List[Dict]:
    """Process search results and store in vector database"""
    processed_results = []
    total = len(search_results)
    
    for i, result in enumerate(search_results):
        try:
            content = fetch_content(result['url'])
            if not content:
                continue
                
            # Store the full content for context
            processed_results.append({
                'title': result['title'],
                'url': result['url'],
                'content': content[:2000],  # Limit content for LLM context
                'snippet': result['snippet']
            })
            
            # Update progress
            progress = (i + 1) / total
            st.session_state['progress'] = progress
            
        except Exception as e:
            logger.error(f"Error processing result {result['url']}: {str(e)}")
            continue
    
    return processed_results

def generate_answer_gemini(
    query: str,
    context: List[Dict],
    model
) -> str:
    """Generate answer using Gemini"""
    if not model or not context:
        return None
    
    # Check if it's a programming query
    is_programming = any(term in query.lower() for term in ['code', 'program', 'algorithm', 'c++', 'python', 'java'])
    
    context_text = "\n\n".join([
        f"Source: {c['title']}\nURL: {c['url']}\nContent: {c['content'][:500]}..."
        for c in context
    ])
    
    if is_programming:
        prompt = f"""Based on the provided sources, provide a detailed answer about the programming question: {query}

Context:
{context_text}

Please provide:
1. A clear explanation of the code/concept
2. Example code if available
3. Key points about implementation
4. Common pitfalls or best practices
5. References to official documentation when available

Format your response as:
EXPLANATION: [Concept explanation]
CODE EXAMPLE: [If available, properly formatted code]
KEY POINTS: [Important implementation details]
SOURCES: [List relevant sources]"""
    else:
        # Use the regular prompt for non-programming queries
        prompt = f"""Based on the provided sources, answer the following question: {query}

Context:
{context_text}

Requirements:
1. Answer ONLY based on the provided context
2. If the context doesn't contain relevant information, say so
3. Stay strictly focused on the query
4. Use specific quotes and citations from sources
5. If sources conflict, mention the discrepancy
6. Maintain factual accuracy
7. Avoid speculation or inference

Format your response as:
ANSWER: [Direct answer to the query]
SOURCES: [List relevant sources]
CONFIDENCE: [High/Medium/Low based on source quality and relevance]"""

    try:
        generation_config = {
            "temperature": 0.3,  # Lower temperature for more factual responses
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        if response.prompt_feedback.block_reason:
            logger.warning(f"Response blocked: {response.prompt_feedback.block_reason}")
            return "I apologize, but I cannot provide a summary for this query due to content safety restrictions. Please try rephrasing your query in a more neutral way."

        return response.text if response.text else "Unable to generate a response. Please try rephrasing your query."

    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return "An error occurred while generating the summary. Please try again with a different query or rephrase your question."

def backup_search(query: str, num_results: int = 5) -> List[Dict]:
    """Backup search method using ddg"""
    try:
        results = ddgs(query, max_results=num_results)
        if results:
            return [
                {
                    'title': r.get('title', 'No Title'),
                    'url': r.get('link', ''),
                    'snippet': r.get('snippet', 'No Snippet')
                }
                for r in results
                if isinstance(r, dict)
            ]
    except Exception as e:
        logger.error(f"Backup search error: {str(e)}")
    return []

# Modify the main search function to use backup if primary fails
def search_with_fallback(query: str, num_results: int = 5) -> List[Dict]:
    """Search with fallback options"""
    results = search_web(query, num_results)
    if not results:
        logger.info("Primary search failed, trying backup method...")
        results = backup_search(query, num_results)
    return results

def validate_and_clean_query(query: str) -> tuple[bool, str]:
    """Validate and clean the search query"""
    # Remove extra whitespace and special characters
    cleaned_query = ' '.join(query.strip().split())
    
    if len(cleaned_query) < 2:
        return False, "Query too short. Please enter a more specific search term."
    
    if len(cleaned_query) > 200:
        return False, "Query too long. Please enter a shorter search term."
    
    # Check for meaningful content (not just special characters)
    if not any(c.isalnum() for c in cleaned_query):
        return False, "Please enter a valid search query containing letters or numbers."
        
    return True, cleaned_query

def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        # Clear ChromaDB cache if needed (uncomment if you want to clear on every run)
        # clear_chroma_db()
        
        # Initialize components
        collection, encoder, llm = init_clients()
        
        # App title and description
        st.title("🔍 AI Research Assistant")
        st.markdown("""
        This search engine uses AI to provide detailed answers to your questions
        by analyzing multiple sources and synthesizing the information.
        """)
        
        if not llm:
            st.warning("""
            No LLM API key found. The search will still work, but without AI-generated summaries.
            To enable AI summaries, add your Google API key to .env file:
            GOOGLE_API_KEY=your_key_here
            
            Get a free API key from: https://makersuite.google.com/app/apikey
            """)
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Search Settings")
            num_results = st.slider(
                "Number of sources:",
                min_value=1,
                max_value=10,
                value=3,
                help="More sources provide more comprehensive results"
            )
            
            if st.button("Clear Cache"):
                clear_chroma_db()
                st.cache_resource.clear()
                st.success("Cache cleared successfully!")
        
        # Main search interface
        with st.form("search_form"):
            query = st.text_input(
                "Enter your search query:",
                placeholder="What would you like to learn about?"
            )
            submitted = st.form_submit_button("Search")
        
        if submitted and query:
            try:
                # Validate query first
                is_valid, cleaned_query = validate_and_clean_query(query)
                if not is_valid:
                    st.error(cleaned_query)
                    return
                    
                with st.spinner("🌐 Searching for relevant sources..."):
                    search_results = search_with_fallback(cleaned_query, num_results)
                
                if not search_results:
                    st.warning("No results found. Please try a different query.")
                    return
                
                # Initialize progress
                progress_bar = st.progress(0)
                st.session_state['progress'] = 0
                
                # Process results
                with st.spinner("📝 Processing sources..."):
                    processed_results = process_and_store_content(
                        search_results,
                        collection,
                        encoder
                    )
                    progress_bar.progress(1.0)
                
                # Generate AI summary if LLM is available
                if llm:
                    with st.spinner("🤖 Generating AI summary..."):
                        answer = generate_answer_gemini(query, processed_results, llm)
                        if answer:
                            st.markdown("### 🤖 AI Summary")
                            # Check if the response contains code blocks
                            if "```" in answer:
                                st.code(answer)
                            else:
                                st.markdown(f">{answer}")
                
                # Display results
                st.markdown("### 📚 Source Details")
                for i, result in enumerate(processed_results, 1):
                    with st.expander(f"{i}. {result['title']}", expanded=True):
                        st.markdown(f"**Source:** [{result['url']}]({result['url']})")
                        st.markdown("**Summary:**")
                        st.markdown(result['snippet'])
                        st.markdown("**Content Preview:**")
                        st.markdown(result['content'][:500] + "...")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.exception("Error in main execution")
    
    except Exception as e:
        st.error("Failed to initialize the application. Please check the logs for details.")
        logger.critical(f"Application initialization failed: {str(e)}")

if __name__ == "__main__":
    main()