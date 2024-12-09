import streamlit as st
import ollama
import logging
import os
import onnxruntime as ort
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


# Configure logging
logging.basicConfig(level=logging.INFO)
print("ONNX Runtime version:", ort.__version__)

# Constants
DOC_PATH = "./data/SPATIAL MODEL IN GIS 814.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}.")
        st.error("PDF file not found.")
        return None
    
def split_documents(documents):
    """Split documents into smamller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    logging.info(f"Sample chunk: {chunks[0]}")
    logging.info(f"Chunks: {len(chunks)} created.")

    return chunks

@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)
    
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        logging.info("Loaded existing Vector database.")
    else:
        # oad and process the PDF document
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None
        
        # Split the documents into chunks
        chunks = split_documents(data)
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector databse created and persists")
    return vector_db

def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your is to generate five
    different versions of the given user question to retrieve relavant documents from
    a vector database. By generating multiple perspectives on the user questions, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original questions: {question}""",
    )
    
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever

def create_chain(retriever, llm):
    """Create the chain"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context: 
{context}
Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logging.info("Chain created with preserved syntax.")
    return chain

def main():
    st.title("Document Assistant")
    
    # User input
    user_input = st.text_input("Enter your question:")
    
    if user_input:
        with st.spinner("Generating response..."):
            try: 
                # Initialize the language model 
                llm = ChatOllama(model=MODEL_NAME)
                
                # Find and process the PDF document
                data = ingest_pdf(DOC_PATH)
                
                # Split the documents into chunks
                chunks = split_documents(data)
    
                # Load the vector database
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Error loading vector database")
                    return
                
                # Create the retriever
                retriever = create_retriever(vector_db, llm)
    
                # Create the chain with the preserved syntax
                chain = create_chain(retriever, llm)
                
                # Get the response
                response = chain.invoke(input=user_input)
                
                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
    else:
        st.info("Please enter a question to get started")
                # # Get the response
                # res = chain.invoke(input=question)
                # print("Response:")
                # print(res)
                
                
                
    
    # if data is None:
    #     return
    
   
    # # Create the vector database
    # vector_db = create_vector_db(chunks)
    
    
    
    
    
if __name__=="__main__":
    main()