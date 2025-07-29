# demo.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from intelligent_rag_system import IntelligentRAGSystem, RAGSystemUtils

# Load environment variables
load_dotenv()

class RAGDemo:
    def __init__(self):
        self.rag_system = IntelligentRAGSystem()
        self.responses = []
    
    def ingest_documents(self, documents_path: str):
        """Ingest documents from a directory or single file"""
        path = Path(documents_path)
        
        if path.is_file():
            print(f"📄 Ingesting single document: {path.name}")
            result = self.rag_system.ingest_document(str(path))
            self.print_ingestion_result(result)
            return [result]
        
        elif path.is_dir():
            print(f"📁 Ingesting documents from directory: {documents_path}")
            results = RAGSystemUtils.batch_ingest_documents(self.rag_system, documents_path)
            for result in results:
                self.print_ingestion_result(result)
            return results
        
        else:
            print(f"❌ Path not found: {documents_path}")
            return []
    
    def print_ingestion_result(self, result):
        """Print formatted ingestion result"""
        status = result['status']
        if status == 'success':
            print(f"  ✅ {result['file_name']}")
            print(f"     Type: {result['document_type']}")
            print(f"     Pages: {result['total_pages']}, Chunks: {result['total_chunks']}")
            print(f"     Clauses: {result['total_clauses']}")
        else:
            print(f"  ❌ {result.get('file_path', 'Unknown file')}")
            print(f"     Error: {result.get('error_message', 'Unknown error')}")
    
    def interactive_query_session(self):
        """Interactive query session"""
        print("\n🔍 Interactive Query Session")
        print("Type 'quit' to exit, 'export' to save responses")
        
        while True:
            query = input("\n💬 Enter your query: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'export':
                self.export_responses()
                continue
            elif not query:
                continue
            
            try:
                print("🤔 Processing query...")
                response = self.rag_system.query_with_context(query)
                self.responses.append(response)
                
                print(f"\n📋 Response:")
                print(f"Answer: {response.answer}")
                print(f"Confidence: {response.confidence_score:.2f}")
                print(f"Relevant Clauses: {len(response.relevant_clauses)}")
                
                if response.relevant_clauses:
                    print(f"\n📝 Top Clauses:")
                    for i, clause in enumerate(response.relevant_clauses[:3]):
                        print(f"  {i+1}. Type: {clause.clause_type.value}")
                        print(f"     Section: {clause.section}")
                        print(f"     Confidence: {clause.confidence:.2f}")
                        print(f"     Content: {clause.content[:150]}...")
                
                print(f"\n💡 Rationale: {response.decision_rationale}")
                
                if response.recommendations:
                    print(f"\n📌 Recommendations:")
                    for rec in response.recommendations:
                        print(f"  • {rec}")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
    
    def export_responses(self):
        """Export all responses"""
        if not self.responses:
            print("📭 No responses to export")
            return
        
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Export individual responses
        for i, response in enumerate(self.responses):
            filename = f"response_{i+1}.json"
            filepath = output_dir / filename
            RAGSystemUtils.export_query_response(response, str(filepath))
        
        # Create comprehensive report
        report_path = output_dir / "comprehensive_report.json"
        RAGSystemUtils.create_query_report(self.responses, str(report_path))
        
        print(f"📁 Exported {len(self.responses)} responses to: {output_dir}")
    
    def demo_with_sample_queries(self):
        """Run demo with predefined queries"""
        sample_queries = [
            "What coverage is provided under this policy?",
            "Are there any exclusions I should be aware of?",
            "What are the payment terms and conditions?",
            "How can this policy be terminated?",
            "What are my obligations under this contract?",
            "Is there a waiting period for coverage?",
            "What happens if I miss a payment?",
            "Are pre-existing conditions covered?"
        ]
        
        print("\n🎯 Running demo queries...")
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            try:
                response = self.rag_system.query_with_context(query)
                self.responses.append(response)
                
                print(f"✅ Confidence: {response.confidence_score:.2f}")
                print(f"📄 Sources: {len(response.sources)}")
                print(f"🔍 Clauses: {len(response.relevant_clauses)}")
                print(f"💬 Answer: {response.answer[:200]}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n🎉 Demo completed! Processed {len(self.responses)} queries")

def main():
    """Main demo function"""
    print("🚀 Welcome to the Intelligent RAG System Demo!")
    
    # Initialize demo
    demo = RAGDemo()
    
    # Menu system
    while True:
        print("\n" + "="*50)
        print("📋 MENU OPTIONS:")
        print("1. 📁 Ingest documents")
        print("2. 🎯 Run sample queries demo")
        print("3. 💬 Interactive query session")
        print("4. 📊 Export results")
        print("5. 🗑️  Delete collection (reset)")
        print("6. ❌ Exit")
        print("="*50)
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            path = input("Enter document path or directory: ").strip()
            if path:
                demo.ingest_documents(path)
        
        elif choice == '2':
            demo.demo_with_sample_queries()
        
        elif choice == '3':
            demo.interactive_query_session()
        
        elif choice == '4':
            demo.export_responses()
        
        elif choice == '5':
            confirm = input("⚠️  This will delete all data. Confirm? (yes/no): ")
            if confirm.lower() == 'yes':
                result = demo.rag_system.delete_collection()
                print(f"Result: {result['status']}")
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option. Please try again.")

if __name__ == "__main__":
    main()