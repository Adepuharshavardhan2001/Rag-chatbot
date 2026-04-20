import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ================== LOAD DOCUMENTS ==================
def load_documents(file_path: str):
    """Load PDF or CSV file"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(" Only .pdf and .csv files are supported!")
    
    docs = loader.load()
    print(f" Loaded {len(docs)} document(s) from {file_path}")
    return docs


# ================== SPLIT DOCUMENTS ==================
def split_documents(docs):
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    print(f" Split into {len(splits)} chunks")
    return splits


# ================== CREATE VECTOR STORE ==================
def create_vector_store(splits, persist_dir="./chroma_db"):
    """Create and save Chroma vector database"""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")   # ← Updated
    
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="mobility_rag"
    )
    print(f" Vector store saved to {persist_dir}")
    return vector_store


# ================== CREATE RAG CHAIN ==================
def create_rag_chain(vector_store):
    """Create the Retrieval-Augmented Generation chain"""
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",        # Best free option right now
    temperature=0.1
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    template = """You are a helpful assistant specialized in mobility, EV, and public transport.
    Answer the question using only the provided context.
    If you don't know the answer, just say "I don't have enough information."

    Context:
    {context}

    Question: {question}
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ================== EXECUTION ==================
if __name__ == "__main__":
    # Correct file path based on your image
    FILE_PATH = "data/Andhra_Pradesh_Sustainable_Electric_Mobility_Policy_4_0.pdf"

    try:
        print(" Starting RAG Pipeline...")
        
        # 1. Load
        docs = load_documents(FILE_PATH)
        
        # 2. Split
        splits = split_documents(docs)
        
        # 3. Embed and Store
        vector_store = create_vector_store(splits)
        
        # 4. Build Chain
        chain = create_rag_chain(vector_store)

        # 5. Ask a test question
        query = "What are the key objectives of the Andhra Pradesh Electric Mobility Policy?"
        print(f"\n Question: {query}")
        
        response = chain.invoke(query)
        
        print("\n Answer:")
        print(response)

    except Exception as e:
        print(f"\n An error occurred: {e}")