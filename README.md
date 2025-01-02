# Ollama PDF RAG Streamlit UI

## Overview
This Streamlit application leverages Ollama and LangChain to create a PDF-based Retrieval-Augmented Generation (RAG) system. Users can upload PDF documents and ask questions about their content, utilizing language models to generate answers based on the document's text.

## Features
- PDF Upload: Users can upload PDF documents to be processed.
- Question Processing: Ask questions about the content of the uploaded PDFs.
- Document Viewing: View uploaded PDFs directly in the application.
- Vector Database Creation: Automatically creates a vector database from the uploaded PDF to facilitate document retrieval and question answering.
- RAG Pipeline implemented in order to comply to everything above

## Installation

1. Clone this repository.
2. Ensure you have Python 3.8 or newer installed.
3. Install Ollama locally(https://ollama.com/download/mac)
4. Pull Ollama:
     ```bash
   ollama pull llama3
6. Install required packages:
   ```bash
   pip install streamlit pdfplumber ollama langchain

To start the application:
streamlit run app.py
