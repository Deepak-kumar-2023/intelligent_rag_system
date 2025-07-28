# config.py
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

# Azure OpenAI Configuration
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4o"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_4o"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_4o"),
)

# Vector Embeddings
embedding_model = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
)

# Qdrant Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "intelligent_documents"