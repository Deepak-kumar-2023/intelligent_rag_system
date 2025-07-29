# test_azure_connection.py
from dotenv import load_dotenv
import os

load_dotenv()

def test_azure_connection():
    """Test Azure OpenAI connection"""
    try:
        print("testing Azure OpenAI connection...")
        from config import client, embedding_model
        
        print("🔍 Testing Azure OpenAI Chat...")
        response = client.chat.completions.create(
            model='gpt-4.1',
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=50
        )
        print(f"✅ Chat test successful: {response.choices[0].message.content}")
        
        print("\n🔍 Testing Azure OpenAI Embeddings...")
        test_embedding = embedding_model.embed_query("This is a test query")
        print(f"✅ Embedding test successful: Got {len(test_embedding)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ Azure connection test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_azure_connection()
    if success:
        print("\n🎉 All Azure OpenAI services are working correctly!")
    else:
        print("\n⚠️  Please check your Azure OpenAI configuration.")