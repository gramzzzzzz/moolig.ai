import streamlit as st
import os
import requests
import re
import numpy as np
import PyPDF2
import io
import asyncio
import nest_asyncio
import threading

from together import Together
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import time

# Apply nest_asyncio to allow running async code in Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# API Keys (Replace with your actual keys)
TOGETHER_API_KEY = "8b41e536935c171ab7eef4bfe5e9dea15fe5a105277fc2b49f8c1e389678a319"
PINECONE_API_KEY = "pcsk_7VH2Wt_M8DP8vcE3vJ9cMMTTKFrsBFUc4PimQSBYE416cHz5nPgxD6eSYVoTJ3PXLBZW5"

# Initialize clients safely
client = Together(api_key=TOGETHER_API_KEY)

# Create a function to initialize Pinecone safely
def init_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        time.sleep(1)
        index = pc.Index("mooligai")
        time.sleep(1)
        return pc, index
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        return None, None

# Initialize Pinecone outside of any async context
pc, index = init_pinecone()

class MooligAI:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.web_searcher = DDGS()
        self.pdf_vectorstore = None
        # Create a thread-local storage for event loops
        self.thread_local = threading.local()

    def extract_pdf_text(self, pdf_file):
        """Extract text from uploaded PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting PDF text: {e}")
            return ""

    def create_pdf_vectorstore(self, pdf_text):
        """Create vector store from PDF text"""
        try:
            # Split text into chunks
            text_chunks = [pdf_text[i:i+1000] for i in range(0, len(pdf_text), 1000)]
            
            # Create vector store
            self.pdf_vectorstore = FAISS.from_texts(
                text_chunks, 
                self.embeddings
            )
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return False

    def get_pdf_context(self, query):
        """Retrieve context from PDF vector store"""
        if not self.pdf_vectorstore:
            return "No PDF loaded. Please upload a PDF first."
        
        try:
            # Retrieve similar documents
            similar_docs = self.pdf_vectorstore.similarity_search(query, k=3)
            context = " ".join([doc.page_content for doc in similar_docs])
            return context
        except Exception as e:
            st.error(f"Error retrieving PDF context: {e}")
            return "Error retrieving context from PDF."

    def get_context(self, query):
        """Retrieve context from Pinecone vector database"""
        try:
            if not index:
                return "Pinecone index is not available. Please check your connection."
                
            query_embeddings = self.embeddings.embed_query(query)
            embedding_numpy = np.array(query_embeddings)
            retrieval_data = index.query(
                namespace="ayurveda",
                vector=embedding_numpy.tolist(),
                top_k=5,
                include_values=False,
                include_metadata=True
            )

            context_list = [
                match.get("metadata", {}).get("source_text", "").strip()
                for match in retrieval_data.get("matches", [])
                if match.get("metadata", {}).get("source_text", "").strip()
            ]

            return " ".join(context_list) if context_list else "No relevant Ayurvedic context found."
        except Exception as e:
            st.error(f"Error retrieving Pinecone context: {e}")
            return f"Error retrieving context: {str(e)}"

    def web_search(self, query, num_results=3):
        """Perform web search using DuckDuckGo"""
        try:
            search_results = list(self.web_searcher.text(query, max_results=num_results))
            return "\n\n".join([
                f"🌐 **Source {i+1}:** {result['title']}\n🔗 {result['href']}\n📖 {result['body']}"
                for i, result in enumerate(search_results)
            ])
        except Exception as e:
            st.error(f"Web search error: {e}")
            return "Unable to perform web search at this time."

    def validate_response(self, original_answer, query):
        """Cross-validate response with web search results"""
        try:
            # Perform web search
            web_results = self.web_search(query)
            
            # Generate validation prompt
            validation_prompt = f"""
            You are an AI validator. Compare the following proposed answer with web search results:

            Original Answer:
            {original_answer}

            Web Search Results:
            {web_results}

            Task: 
            1. Determine the accuracy of the original answer
            2. Rate the answer's reliability (High/Medium/Low)
            3. Highlight any discrepancies or additional insights
            4. Provide a concise validation summary

            Response Format:
            - Reliability Rating: [High/Medium/Low]
            - Validation Summary: 
            - Key Insights/Corrections (if any):
            """

            # Generate validation
            validation = self.generate_response(validation_prompt)
            return validation
        except Exception as e:
            return f"Validation error: {e}"

    def get_event_loop(self):
        """Get or create an event loop for the current thread"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        except RuntimeError:
            # If there's no event loop in the current thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def generate_response(self, prompt):
        """Generate response using Together AI"""
        try:
            # Ensure we have an event loop
            loop = self.get_event_loop()
            
            # Run in the event loop
            if loop.is_running():
                # If the loop is already running, we create a future
                future = asyncio.run_coroutine_threadsafe(self._async_generate_response(prompt), loop)
                return future.result()
            else:
                # If the loop is not running, we run our coroutine
                return loop.run_until_complete(self._async_generate_response(prompt))
                
        except Exception as e:
            print(f"Response generation error: {e}")
            return f"I apologize, but I couldn't generate a response at the moment. Error: {str(e)}"

    async def _async_generate_response(self, prompt):
        """Asynchronous version of generate_response"""
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=False
        )
        return response.choices[0].message.content

def main():
    # Custom Styling
    st.markdown("""
    <style>
        body { background-color: #F5F1E3; font-family: 'Arial', sans-serif; }
        .title { 
            text-align: center; 
            color: #4A7856; 
            font-size: 40px; 
            font-weight: bold; 
            margin: 10px 0;
        }
        .leaf-bg {
            background-image: url('https://www.transparenttextures.com/patterns/leaf.png');
            padding: 20px;
            border-radius: 15px;
        }
        .custom-button {
            background-color: #4A7856 !important;
            color: white !important;
            font-weight: bold !important;
        }
        .sidebar .sidebar-content {
            background-color: #E6F3E6;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    st.sidebar.image("https://via.placeholder.com/150", caption="Moolig.AI", use_container_width=True)
    
    # Initialize RAG System
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MooligAI()
    
    # Store last answer in session state
    if 'last_answer' not in st.session_state:
        st.session_state.last_answer = ""

    # Sidebar PDF Upload
    st.sidebar.header("📚 PDF Knowledge Base")
    uploaded_pdf = st.sidebar.file_uploader("Upload Ayurveda PDF", type=['pdf'])
    
    if uploaded_pdf is not None:
        # Extract and process PDF
        with st.sidebar.status("Processing PDF..."):
            pdf_text = st.session_state.rag_system.extract_pdf_text(uploaded_pdf)
            if pdf_text:
                if st.session_state.rag_system.create_pdf_vectorstore(pdf_text):
                    st.sidebar.success("PDF Successfully Processed!")
            else:
                st.sidebar.error("Failed to process PDF")

    # Main App Title
    st.markdown("""
    <div class='leaf-bg'>
        <p class='title'>🌿 Moolig.AI 2.0 🌿</p>
        <p class='title'>Advanced Ayurveda Knowledge Hub</p>
    </div>
    """, unsafe_allow_html=True)

    # User Query Section
    prompt1 = st.text_input("🌿 Ask about Ayurveda:", placeholder="Enter your holistic health question...")

    # Create columns for different actions
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💬 Ayurvedic Query", key="query_button", help="Retrieve Ayurvedic wisdom"):
            if prompt1:
                with st.spinner("Retrieving Ayurvedic wisdom..."):
                    try:
                        # Combine contexts from Pinecone and PDF (if available)
                        pinecone_context = st.session_state.rag_system.get_context(prompt1)
                        pdf_context = st.session_state.rag_system.get_pdf_context(prompt1) if st.session_state.rag_system.pdf_vectorstore else ""
                        
                        combined_context = f"{pinecone_context}\n\nPDF Context:\n{pdf_context}"

                        retrieval_prompt = f"""
        You are Moolig.AI, an expert in Ayurveda, the ancient Indian system of natural healing.
        Provide an **accurate, concise, and holistic** answer based on Ayurvedic knowledge.

        ### Context:
        {combined_context}

        ### User Question:
        {prompt1}

        Response Guidelines:
        - Speak like a friendly guide
        - Base answers on Ayurvedic principles
        - Provide direct and actionable advice
        - Reference classical sources when possible
        - Promote safe, traditional remedies
        """
                        answer = st.session_state.rag_system.generate_response(retrieval_prompt)
                        
                        # Store the answer for validation
                        st.session_state.last_answer = answer

                        # Display Answer
                        st.markdown("### 🌱 Moolig.AI Response")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        st.info("Please try again or rephrase your question.")
    
    with col2:
        if st.button("🌐 Web Search", key="search_button", help="Search the web for Ayurvedic knowledge"):
            if prompt1:
                with st.spinner("Searching the web..."):
                    try:
                        web_results = st.session_state.rag_system.web_search(prompt1)
                        st.markdown("### 🌎 Web Search Results")
                        st.markdown(web_results, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Web search error: {str(e)}")
                        st.info("Please try again later.")
    
    with col3:
        if st.button("✅ Validate Response", key="validate_button", help="Cross-validate Ayurvedic advice"):
            if prompt1 and st.session_state.last_answer:
                with st.spinner("Validating response..."):
                    try:
                        # Validate the response
                        validation_result = st.session_state.rag_system.validate_response(
                            st.session_state.last_answer, prompt1
                        )
                        
                        # Display Validation
                        st.markdown("### 🔍 Response Validation")
                        st.markdown(validation_result)
                    except Exception as e:
                        st.error(f"Validation error: {str(e)}")
                        st.info("Unable to validate response at this time.")
            elif prompt1 and not st.session_state.last_answer:
                st.warning("Please generate a response first using the Ayurvedic Query button.")
            else:
                st.warning("Please enter a query first.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")