
## Overview

This project is an AI-powered system that allows users to upload PDF documents and ask questions based on the content of the documents. The system then processes the PDF, extracts the text, and uses a combination of Langchain, Pinecone, and Streamlit to provide relevant answers.

## Architecture

The system leverages a sophisticated architecture combining the latest in natural language processing and vector database technologies.

- Text from PDFs is extracted and split into manageable chunks.
- These chunks are then embedded into a high-dimensional space using embedding models from OpenAI or Sentence Transformers.
- The embeddings are stored in a Pinecone vector database, which allows for efficient similarity search when queries are made.
- Upon receiving a query, the system retrieves the most relevant document chunks and uses an AI model like GPT-3 or GPT-4 to generate an answer.


Here is a preview of the application in action.

## How to Run

To run this application on your local machine, follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies using the following command:

   ```sh
   pip install -r requirements.txt
3. Run the Streamlit application
   ```sh
   python -m streamlit run app.py
