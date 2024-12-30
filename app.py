"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""

#imports all from the virtual environment
import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

import warnings
warnings.filterwarnings('ignore') #ignore warnings

#enviorment variables
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

#streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

#logger
logger = logging.getLogger(__name__)

def create_vector_db(file_upload):
    """
    Create a vector database from an uploaded PDF file.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = PyPDFLoader(path) #load the pdf file using Pypdfloader
        data = loader.load()

    #split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    #updated embeddings configuration
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="myRAG"
    )
    logger.info("Vector DB created")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db

def process_question(question, vector_db, selected_model):
    """
    Process a user question using the vector database and selected language model.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    #initialize LLM
    llm = ChatOllama(model=selected_model)
    
    # Query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    #set up retriever which fetches relevant context from the vector database.
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    #RAG prompt template
    template = """Answer the question based ONLY on the following context: {context}
    Question: {question}
    """
    #retrieved context and the original question are then 
    #formatted into a prompt using a ChatPromptTemplate.
    prompt = ChatPromptTemplate.from_template(template)

    #create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

@st.cache_data
def extract_all_pages_as_images(file_upload):
    """
    Extract all pages from a PDF file as images.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

def delete_vector_db(vector_db):
    """
    Delete the vector database and clear related session state.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")

def main():
    """
    Main function to run the Streamlit application.
    """
    st.subheader("Ollama PDF Reader", divider="gray", anchor=False)

    #get available models
    models_info = ollama.list()
    selected_model = "llama3"

    #create layout
    col1, col2 = st.columns([1.5, 2])

    #initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    #file uploader
    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", 
        type="pdf", 
        accept_multiple_files=False
    )

    if file_upload:
        if st.session_state["vector_db"] is None:
            with st.spinner("Processing uploaded PDF..."):
                st.session_state["vector_db"] = create_vector_db(file_upload)
                pdf_pages = extract_all_pages_as_images(file_upload)
                st.session_state["pdf_pages"] = pdf_pages

    #display PDF if pages are available
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        # PDF display controls
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=1000, 
            value=700, 
            step=50
        )

        #display PDF pages
        with col1:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

    #delete collection button
    delete_collection = col1.button(
        "⚠️ Delete collection", 
        type="secondary"
    )

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    #chat interface
    with col2:
        message_container = st.container(height=500, border=True)

        #display chat history
        for i, message in enumerate(st.session_state["messages"]):
            with message_container.chat_message(message["role"]):
                st.markdown(message["content"])

        #chat input and processing
        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                #add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user"):
                    st.markdown(prompt)

                #process and display assistant response
                with message_container.chat_message("assistant"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                #add assistant response to chat history
                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e)
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to begin chat...")
                
if __name__ == "__main__":
    main()
