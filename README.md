#                                                              Mobility AI: EV Policy RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) application designed to analyze mobility domain documents. This tool specifically processes the Andhra Pradesh Sustainable Electric Mobility Policy to provide instant, grounded answers to complex regulatory questions.

# Live Preview

Tip: Record a quick GIF of you typing a question in Streamlit and add it here!

# Key Features

Conversational AI: Interactive chat interface built with Streamlit.

Domain Specific: Specialized in Mobility, EVs, and Public Transport.

Context-Aware: Uses Google Gemini 1.5 Flash to answer questions based only on the provided documents.

Persistent Memory: Uses ChromaDB to store document embeddings locally for instant retrieval.

Hybrid Data Support: Handles both .pdf and .csv files.

# Architecture

The system follows a modular pipeline to ensure data privacy and accuracy:

Ingestion: PyPDFLoader & CSVLoader extract text from the data/ folder.

Vectorization: gemini-embedding-001 converts text into high-dimensional vectors.

Storage: Chroma indexes these vectors for semantic search.

# Tech Stack

Category,Tools
LLM & Embeddings,Google Gemini (GenAI)
Orchestration,LangChain
Vector Database,ChromaDB
Frontend,Streamlit
Document Processing,"PyPDF, SciPy, Pandas"
Retrieval: The system retrieves the 4 most relevant context snippets for every user query.

Generation: A custom ChatPromptTemplate guides the LLM to provide factual responses.
