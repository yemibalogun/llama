import streamlit as st
import ollama
import logging
import os
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "./data/Service Paper Handbook.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"

def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}.")
        return None
    
def split_documents(documents):
    """Split documents into smamller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    logging.info(f"Sample chunk: {chunks[0]}")
    logging.info(f"Chunks: {len(chunks)} created.")

    return chunks

def create_vector_db(chunks):
    """Create a vector database from document chunks."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("Vector database created.")
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
    template = """Answer the question bsed ONLY on the following context: 
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
    
    logging.info("Chain created successfully.")
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
                
                # Load the vector database
                vector_db = create_vector_db(llm)
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
                
                
                
    # # Find and process the PDF document
    # data = ingest_pdf(DOC_PATH)
    # if data is None:
    #     return
    
    # # Split the documents into chunks
    # chunks = split_documents(data)
    
    # # Create the vector database
    # vector_db = create_vector_db(chunks)
    
    
    
    
    
if __name__=="__main__":
    main()