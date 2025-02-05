import streamlit as st
import os
from typing import List, Tuple
import PyPDF2
from io import BytesIO
import faiss
import numpy as np
from groq import Groq
from cohere import Client
import pickle
import tiktoken

# Initialize clients
groq_client = st.secrets["GROQ_API_KEY"]
cohere_client = st.secrets["COHERE_API_KEY"]

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    # Explicitly use a known encoding
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500):
        self.chunks = []
        self.chunk_size = chunk_size
        
    def process_pdf(self, pdf_file: BytesIO) -> List[str]:
        """Process PDF file and return chunks of text"""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        # Improved chunking with overlap
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size = count_tokens(" ".join(current_chunk))
            
            if current_size >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                # Keep last 50 words for overlap
                current_chunk = current_chunk[-50:]
                current_size = count_tokens(" ".join(current_chunk))
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        self.chunks = chunks
        return chunks

class VectorStore:
    def __init__(self):
        self.index = None
        self.texts = []
        
    def create_index(self, texts: List[str]):
        """Create FAISS index from text chunks"""
        embeddings = cohere_client.embed(
            texts=texts,
            model='embed-english-v3.0',
            input_type='search_query'
        ).embeddings
        
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        self.texts = texts
        
    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar chunks"""
        query_embedding = cohere_client.embed(
            texts=[query],
            model='embed-english-v3.0',
            input_type='search_query'
        ).embeddings[0]
        
        D, I = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        
        results = [(self.texts[i], D[0][idx]) for idx, i in enumerate(I[0])]
        return results

    def save(self, path: str):
        """Save index and texts"""
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)
    
    def load(self, path: str):
        """Load index and texts"""
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/texts.pkl", "rb") as f:
            self.texts = pickle.load(f)

def generate_response(query: str, context: str, max_tokens: int = 4000) -> str:
    """Generate response using Groq with token limit management"""
    # Create base prompt
    base_prompt = """You are a helpful assistant. Use the following context to answer the question. 
    If you cannot answer the question based on the context, say so.
    
    Context: {context}
    
    Question: {query}
    
    Answer:"""
    
    # Count tokens in base prompt and query
    prompt_tokens = count_tokens(base_prompt.format(context="", query=query))
    available_tokens = max_tokens - prompt_tokens - 1024  # Reserve 1024 tokens for response
    
    # Truncate context if needed
    context_tokens = count_tokens(context)
    if context_tokens > available_tokens:
        encoding = tiktoken.get_encoding("cl100k_base")
        context_ids = encoding.encode(context)
        truncated_ids = context_ids[:available_tokens]
        context = encoding.decode(truncated_ids)
    
    prompt = base_prompt.format(context=context, query=query)
    
    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.3,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating the response. Please try asking a different question or uploading a smaller document."

# Streamlit Interface
st.title("📚Document Chat Assistant")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# File upload
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            # Process the document
            processor = DocumentProcessor(chunk_size=500)  # Smaller chunk size
            chunks = processor.process_pdf(uploaded_file)
            
            # Create index
            st.session_state.vector_store.create_index(chunks)
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your document"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Search for relevant chunks
            if st.session_state.vector_store.index is not None:
                try:
                    results = st.session_state.vector_store.search(query, k=2)  # Reduced number of chunks
                    context = "\n".join([chunk for chunk, _ in results])
                    
                    # Generate response
                    response = generate_response(query, context)
                    st.markdown(response)
                    
                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.error("Please upload a document first!")

# Save and load functionality
if st.sidebar.button("Save Index"):
    if st.session_state.vector_store.index is not None:
        st.session_state.vector_store.save("saved_index")
        st.sidebar.success("Index saved successfully!")

if st.sidebar.button("Load Index"):
    try:
        st.session_state.vector_store.load("saved_index")
        st.sidebar.success("Index loaded successfully!")
    except:
        st.sidebar.error("No saved index found!")
