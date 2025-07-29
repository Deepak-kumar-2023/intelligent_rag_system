# test_system.py
import os
from dotenv import load_dotenv
from intelligent_rag_system import IntelligentRAGSystem, RAGSystemUtils

# Load environment variables
load_dotenv()

def test_system():
    """Test the RAG system with sample operations"""
    
    print("🚀 Testing Intelligent RAG System...")
    
    # Initialize the system
    try:
        rag_system = IntelligentRAGSystem()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    # Test document ingestion (if you have a sample document)
    sample_doc_path = "sample_document.pdf"  # Replace with your test document
    if os.path.exists(sample_doc_path):
        print(f"\n📁 Testing document ingestion with: {sample_doc_path}")
        result = rag_system.ingest_document(sample_doc_path)
        print(f"Ingestion result: {result['status']}")
        if result['status'] == 'success':
            print(f"   - Document type: {result['document_type']}")
            print(f"   - Total pages: {result['total_pages']}")
            print(f"   - Total chunks: {result['total_chunks']}")
            print(f"   - Total clauses: {result['total_clauses']}")
    else:
        print(f"\n⚠️  Sample document not found at: {sample_doc_path}")
        print("   Create a sample PDF document to test ingestion")
    
    # Test querying (even without documents, system should handle gracefully)
    print("\n🔍 Testing query processing...")
    test_queries = [
        "What coverage is provided under this policy?",
        "Are there any exclusions?",
        "What are the payment terms?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = rag_system.query_with_context(query)
            print(f"✅ Response generated - Confidence: {response.confidence_score:.2f}")
            print(f"   Answer preview: {response.answer[:100]}...")
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    print("\n🎉 System test completed!")

if __name__ == "__main__":
    test_system()