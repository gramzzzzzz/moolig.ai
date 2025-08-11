import streamlit as st
import os
import requests
from io import BytesIO
from together import Together
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import json
import base64
from pinecone import Pinecone, ServerlessSpec
import time
from sentence_transformers import SentenceTransformer


# API Key for Together AI
TOGETHER_API_KEY = "8b41e536935c171ab7eef4bfe5e9dea15fe5a105277fc2b49f8c1e389678a319"
PINECONE_API_KEY = "pcsk_7VH2Wt_M8DP8vcE3vJ9cMMTTKFrsBFUc4PimQSBYE416cHz5nPgxD6eSYVoTJ3PXLBZW5"

client = Together(api_key=TOGETHER_API_KEY)
pc = Pinecone(api_key="pcsk_7VH2Wt_M8DP8vcE3vJ9cMMTTKFrsBFUc4PimQSBYE416cHz5nPgxD6eSYVoTJ3PXLBZW5")
index_name = "mooligai"
time.sleep(1)
index = pc.Index(index_name)
time.sleep(1)

# Custom Styling
st.markdown(
    """
    <style>
        /* Background and fonts */
        body {
            background-color: #F5F1E3;
            font-family: 'Arial', sans-serif;
        }
        /* Title styling */
        .title {
            text-align: center;
            color: #4A7856;
            font-size: 35px;
            font-weight: bold;
            margin-top: 0px;
            margin-bottom: 0px;
        }
        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background-color: #E9E4D3;
            padding: 20px;
            border-radius: 10px;
        }
        /* Text Input */
        .stTextInput>div>div>input {
            border: 2px solid #4A7856;
            border-radius: 10px;
            font-size: 18px;
            padding: 8px;
        }
        /* Button */
        .stButton>button {
            background-color: #4A7856;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
            border: none;
        }
        /* Chat History */
        .chat-box {
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to generate AI responses
def generate_response(prompt):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>","<|eom_id|>"],
        stream=False
    )
    return response.choices[0].message.content

def get_context(query):
    import re
    import numpy as np

    query_embeddings = st.session_state.embeddings.embed_query(query)
    embedding_numpy = np.array(query_embeddings)
    retrieval_data = index.query(
        namespace="ayurveda",
        vector=embedding_numpy.tolist(),
        top_k=5,
        include_values=False,
        include_metadata=True
    )

    context_list = []

    for match in retrieval_data.get("matches", []):
        source_text = match.get("metadata", {}).get("source_text", "")
        cleaned_text = re.sub(r'\n+', '\n', source_text)
        meaningful_sentences = [sentence.strip() for sentence in cleaned_text.split("\n") if len(sentence.strip()) > 20]

        if meaningful_sentences:
            context_list.append("\n".join(meaningful_sentences))

    print(context_list)

    return " ".join(context_list) if context_list else "No relevant Ayurvedic context found."

# Function to handle PDF embeddings
def vector_embedding(uploaded_file=None):
    if "vectors" not in st.session_state:
        os.makedirs("pdfs", exist_ok=True)
        if uploaded_file:
            with open(os.path.join("pdfs", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.loader = PyPDFDirectoryLoader("pdfs")
        else:
            st.session_state.loader = PyPDFDirectoryLoader("pdfs")
        
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = []

        if st.session_state.docs:
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if st.session_state.final_documents:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        else:
            st.warning("No documents found to embed.")

# Function to summarize a document
def summarize_document():
    if "final_documents" not in st.session_state or not st.session_state.final_documents:
        return "Please upload a document first."
    try:
        full_text = " ".join([doc.page_content for doc in st.session_state.final_documents])
        summary_prompt = f"""
        Please provide a concise summary of the following document:
        {full_text[:4000]}
        Summary:
        """
        return generate_response(summary_prompt)
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return "An unexpected error occurred."

def summarize_context(response):
    try:
        if "context_summary" not in st.session_state:
            st.session_state.context_summary = ""
        
        summary_prompt = f"""
        Given the existing summary:
        {st.session_state.context_summary}
        
        Update the summary based on the following new response:
        {response}
        
        Final Summary:
        """
        
        updated_summary = generate_response(summary_prompt)
        st.session_state.context_summary = updated_summary
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return "An unexpected error occurred."

# App Title
st.markdown("<p class='title'>🌿 Moolig.AI🌿</p>", unsafe_allow_html=True)
st.markdown("<p class='title'>Ayurveda Knowledge Hub</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📄 Upload Ayurveda PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if uploaded_file and st.sidebar.button("Upload & Embed"):
    with st.spinner("Processing document..."):
        vector_embedding(uploaded_file)
        st.sidebar.success("Document uploaded and embedded! ✅")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "context_summary" not in st.session_state:
    st.session_state.context_summary = ""

# Summarization Button
if st.sidebar.button("Summarize Document"):
    with st.spinner("Generating summary..."):
        summary = summarize_document()
        st.markdown("### 📝 Document Summary")
        st.write(summary)

# User Query Section
prompt1 = st.text_input("🌿 Ask about Ayurveda:", placeholder="Enter your question here...")

if st.button("💬 Ask Moolig.AI"):
    if prompt1:
        context = get_context(prompt1)
        retrieval_prompt = f"""
You are Moolig.AI, an expert in Ayurveda, the ancient Indian system of natural healing.
Your goal is to provide **accurate, concise, and holistic** answers based on Ayurvedic knowledge.

### Instructions:
- **Speak Like a Friendly Guide**: Keep your tone warm, engaging, and easy to understand—like a wise friend sharing valuable insights.  
- **Stick to Authentic Ayurveda**: Base responses on time-tested Ayurvedic principles, including Doshas (Vata, Pitta, Kapha), herbs, and holistic wellness techniques.  
- **Keep It Short & Useful**: Provide **direct and actionable** answers while ensuring clarity. If needed, you can offer a brief explanation before the remedy.  
- **Mention Classical Sources**: Where relevant, reference Ayurvedic texts (Charaka Samhita, Sushruta Samhita, etc.) to support your response.  
- **Be Safe & Responsible**: Avoid unverified claims and always promote remedies that are traditionally backed and safe for general use. When necessary, encourage consulting a qualified Ayurvedic practitioner.  
- **Bridge Tradition & Science**: Highlight traditional remedies while acknowledging modern research when applicable.  
- **Encourage a Holistic Approach**: Ayurveda is about balance—mention diet, lifestyle, and mindfulness along with herbal remedies when relevant.  

### Conversational Summary:
{st.session_state.context_summary}

### Context:
{context}

### User Question:
{prompt1}

Response:
        """

        with st.spinner("Fetching Ayurveda wisdom..."):
            answer = generate_response(retrieval_prompt)
            st.session_state.chat_history.append({"question": prompt1, "answer": answer})

            summarize_context(answer)
            print(f"Summary: {answer}")
            
            st.markdown("### 🌱 Answer from Moolig.AI")
            st.write(answer)
            
            # Display Chat History
            with st.expander("📜 Chat History"):
                for i, chat in enumerate(st.session_state.chat_history):
                    st.markdown(f"<div class='chat-box'><b>Q{i+1}:</b> {chat['question']}<br><b>A{i+1}:</b> {chat['answer']}</div>", unsafe_allow_html=True)
