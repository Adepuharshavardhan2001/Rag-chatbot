import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import create_rag_chain
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

st.set_page_config(page_title="Mobility RAG Chatbot", page_icon="🚀", layout="wide")
st.title("🚗 RAG Q&A Chatbot - Mobility Domain")
st.caption("Ask questions about your PDF/CSV documents")

# Load vector store
@st.cache_resource
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="mobility_rag"
    )
    return vector_store

vector_store = load_vector_store()
chain = create_rag_chain(vector_store)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask anything about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})