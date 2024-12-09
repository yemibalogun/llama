from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import logging
import os

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
    # Find and process the PDF document
    data = ingest_pdf(DOC_PATH)
    if data is None:
        return
    
    # Split the documents into chunks
    chunks = split_documents(data)
    
    # Create the vector database
    vector_db = create_vector_db(chunks)
    
    # Initialize the language model 
    llm = ChatOllama(model=MODEL_NAME)
    
    # Create the retriever
    retriever = create_retriever(vector_db, llm)
    
    # Create the chain with the preserved syntax
    chain = create_chain(retriever, llm)
    
    # Example query
    question = "How to write introduction paragraph?"
    
    # Get the response
    res = chain.invoke(input=question)
    print("Response:")
    print(res)
    
if __name__=="__main__":
    main()






# doc_path = "./data/Service Paper Handbook.pdf"
# model = "llama3.2"

# # Local PDF file uploads
# if doc_path:
#     loader = PyPDFLoader(file_path=doc_path)
#     data = loader.load()
#     print("done loading...")
# else:
#     print("Upload a PDF file")
    
# content = data[0].page_content
# # print(content[:100])

# # === End of PDF Ingestion ===

# # === Extract Text from PDF Fies and Split into Small Chunks ===

# # Split and chunk
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
# chunks = text_splitter.split_documents(data)
# print("done splitting...")

# # print(f"Number of chunks: {len(chunks)}")
# # print(f"Example chunk: {chunks[0]}")

# # ===== Add to vector database ====

# ollama.pull("nomic-embed-text")

# vector_db = Chroma.from_documents(
#     documents=chunks,
#     embedding=OllamaEmbeddings(model="nomic-embed-text"),
#     collection_name="simple-rag",
# )
# print("done adding to vector database...")

# # === Retrieval ===

# # Set up our model to use
# llm = ChatOllama(model=model)

# # a simple technique to generate multiple questions from a single
# # based on those questions, getting the best of both worlds.
# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an AI language model assistant. Your is to generate five
#     different versions of the given user question to retrieve relavant documents from
#     a vector database. By generating multiple perspectives on the user questions, your
#     goal is to help the user overcome some of the limitations of the distance-based
#     similarity search. Provide these alternative questions separated by newlines.
#     Original questions: {question}""",
# )

# retriever = MultiQueryRetriever.from_llm(
#     vector_db.as_retriever(), llm, prompt=QUERY_PROMPT 
# )

# # RAG prompt
# template = """Answer the question bsed ONLY on the following context: 
# {context}
# Question: {question}
# """

# prompt = ChatPromptTemplate.from_template(template)

# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# res = chain.invoke(input=("what is the document about?",))

# print(res)
