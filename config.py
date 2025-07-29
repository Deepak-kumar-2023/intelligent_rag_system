# config.py
import os
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT_4")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY_4") 
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION_4")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "intelligent_query_system")

try:
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    print("✅ Azure OpenAI client initialized successfully")
except Exception as e:
    print(f"❌ Error initializing Azure OpenAI client: {e}")
    raise

# Initialize embedding model
try:
    embedding_model = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
    )
    print("✅ Embedding model initialized successfully")
except Exception as e:
    print(f"❌ Error initializing embedding model: {e}")
    raise