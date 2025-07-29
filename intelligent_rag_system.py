# intelligent_rag_system.py
# intelligent_rag_system.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import os
import re
import json
from datetime import datetime
from pathlib import Path
import logging

# Updated imports to fix deprecation warnings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import spacy

from config import client, embedding_model, QDRANT_URL, COLLECTION_NAME

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DocumentType(Enum):
    INSURANCE_POLICY = "insurance_policy"
    LEGAL_CONTRACT = "legal_contract"
    HR_POLICY = "hr_policy"
    COMPLIANCE_DOCUMENT = "compliance_document"
    EMAIL = "email"
    GENERAL = "general"

class ClauseType(Enum):
    COVERAGE = "coverage"
    EXCLUSION = "exclusion"
    LIMITATION = "limitation"
    OBLIGATION = "obligation"
    TERMINATION = "termination"
    PAYMENT = "payment"
    LIABILITY = "liability"
    CONFIDENTIALITY = "confidentiality"
    DISPUTE_RESOLUTION = "dispute_resolution"
    GENERAL = "general"

@dataclass
class ExtractedClause:
    clause_type: ClauseType
    content: str
    section: str
    page_number: int
    confidence: float
    keywords: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "clause_type": self.clause_type.value,
            "content": self.content,
            "section": self.section,
            "page_number": self.page_number,
            "confidence": self.confidence,
            "keywords": self.keywords
        }

@dataclass
class QueryResponse:
    query: str
    answer: str
    relevant_clauses: List[ExtractedClause]
    sources: List[Dict[str, Any]]
    confidence_score: float
    decision_rationale: str
    recommendations: List[str]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_json(self) -> str:
        """Convert response to structured JSON format"""
        response_dict = {
            "query": self.query,
            "answer": self.answer,
            "relevant_clauses": [clause.to_dict() for clause in self.relevant_clauses],
            "sources": self.sources,
            "confidence_score": self.confidence_score,
            "decision_rationale": self.decision_rationale,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
            "total_clauses_found": len(self.relevant_clauses),
            "processing_metadata": {
                "system_version": "1.0",
                "model_used": "gpt-4o-mini",
                "embedding_model": "text-embedding-ada-002"
            }
        }
        return json.dumps(response_dict, indent=2, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return json.loads(self.to_json())

class IntelligentRAGSystem:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.clause_patterns = self._initialize_clause_patterns()
        self._initialize_vector_store()
        logger.info("IntelligentRAGSystem initialized successfully")
        
    def _initialize_vector_store(self):
        """Initialize or connect to existing Qdrant collection"""
        try:
            # Create Qdrant client
            qdrant_client = QdrantClient(url=QDRANT_URL)
            
            # Check if collection exists
            collections = qdrant_client.get_collections().collections
            collection_exists = any(col.name == COLLECTION_NAME for col in collections)
            
            if not collection_exists:
                logger.info(f"Creating new Qdrant collection: {COLLECTION_NAME}")
                # Create collection with proper vector size
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=1536,  # Ada-002 embedding size
                        distance=Distance.COSINE
                    )
                )
            
            # Initialize vector store with correct parameters
            self.vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=COLLECTION_NAME,
                embedding=embedding_model  # Note: 'embeddings' not 'embedding'
            )
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            # Fallback: Create new vector store
            try:
                qdrant_client = QdrantClient(url=QDRANT_URL)
                self.vector_store = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=COLLECTION_NAME,
                    embeddings=embedding_model
                )
                logger.info("Fallback vector store initialization successful")
            except Exception as fallback_error:
                logger.error(f"Fallback initialization also failed: {fallback_error}")
                raise fallback_error
        
    def _initialize_clause_patterns(self) -> Dict[ClauseType, List[str]]:
        """Initialize regex patterns for different clause types"""
        return {
            ClauseType.COVERAGE: [
                r"(?i)(coverage|covered|insured|benefits?|protection)",
                r"(?i)(policy\s+covers?|we\s+will\s+pay|included\s+in)",
                r"(?i)(eligible|entitled|provided|compensat\w+)"
            ],
            ClauseType.EXCLUSION: [
                r"(?i)(exclud\w+|except\w+|not\s+cover\w+)",
                r"(?i)(limitation|restrict\w+|outside\s+the\s+scope)",
                r"(?i)(does\s+not\s+apply|not\s+included|void\s+if)"
            ],
            ClauseType.LIMITATION: [
                r"(?i)(limit\w+|maximum|cap|ceiling|threshold)",
                r"(?i)(up\s+to|not\s+exceed|subject\s+to)",
                r"(?i)(minimum|floor|deductible)"
            ],
            ClauseType.OBLIGATION: [
                r"(?i)(must|shall|required|obligat\w+|duty|duties)",
                r"(?i)(responsible|comply|adherence|mandatory)"
            ],
            ClauseType.TERMINATION: [
                r"(?i)(terminat\w+|end|expir\w+|cancel\w+)",
                r"(?i)(dissolution|void|nullif\w+|cease)"
            ],
            ClauseType.PAYMENT: [
                r"(?i)(payment|pay|premium|fee|cost|price)",
                r"(?i)(invoice|billing|due\s+date|amount|financial)"
            ],
            ClauseType.LIABILITY: [
                r"(?i)(liabil\w+|responsible|accountab\w+|indemni\w+)",
                r"(?i)(damages|loss|harm|injury|fault)"
            ],
            ClauseType.CONFIDENTIALITY: [
                r"(?i)(confident\w+|proprietary|non-disclosure|secret)",
                r"(?i)(private|sensitive|classified|privileged)"
            ],
            ClauseType.DISPUTE_RESOLUTION: [
                r"(?i)(dispute|arbitrat\w+|mediat\w+|litigation)",
                r"(?i)(conflict|resolution|settlement|court|legal\s+action)"
            ]
        }
    
    def detect_document_type(self, text: str) -> DocumentType:
        """Detect document type based on content"""
        text_lower = text.lower()
        
        type_keywords = {
            DocumentType.INSURANCE_POLICY: [
                "insurance", "policy", "premium", "coverage", "insured", 
                "beneficiary", "claim", "deductible", "underwriter", "policyholder"
            ],
            DocumentType.LEGAL_CONTRACT: [
                "agreement", "contract", "party", "parties", "whereas", 
                "hereinafter", "covenant", "breach", "governing law", "jurisdiction"
            ],
            DocumentType.HR_POLICY: [
                "employee", "human resources", "hr policy", "workplace", 
                "conduct", "benefits", "leave", "performance", "disciplinary", "personnel"
            ],
            DocumentType.COMPLIANCE_DOCUMENT: [
                "compliance", "regulation", "regulatory", "audit", 
                "standard", "requirement", "control", "risk assessment", "governance"
            ],
            DocumentType.EMAIL: [
                "from:", "to:", "subject:", "date:", "sent:", "received:", "cc:", "bcc:"
            ]
        }
        
        scores = {}
        total_words = len(text_lower.split())
        
        for doc_type, keywords in type_keywords.items():
            score = sum(text_lower.count(keyword.lower()) for keyword in keywords)
            # Normalize by document length
            normalized_score = (score / max(total_words, 1)) * 1000
            scores[doc_type] = normalized_score
        
        max_type = max(scores, key=scores.get)
        return max_type if scores[max_type] > 0.1 else DocumentType.GENERAL
    
    def extract_clauses(self, text: str, page_num: int = 0) -> List[ExtractedClause]:
        """Extract and classify clauses from text"""
        clauses = []
        
        try:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
            
            current_clause = []
            for sent in sentences:
                current_clause.append(sent)
                
                # Process clause when we have enough content or reach a sentence end
                if len(' '.join(current_clause)) > 100 or sent.endswith(('.', ':', ';', '!')):
                    clause_text = ' '.join(current_clause).strip()
                    
                    if len(clause_text) > 50:  # Only process substantial clauses
                        clause_type = self._classify_clause(clause_text)
                        
                        if clause_type != ClauseType.GENERAL or len(clause_text) > 200:
                            keywords = self._extract_keywords(clause_text)
                            confidence = self._calculate_confidence(clause_text, clause_type)
                            
                            clause = ExtractedClause(
                                clause_type=clause_type,
                                content=clause_text,
                                section=self._extract_section(text, clause_text),
                                page_number=page_num,
                                confidence=confidence,
                                keywords=keywords
                            )
                            clauses.append(clause)
                    
                    current_clause = []
            
            # Process any remaining clause
            if current_clause:
                clause_text = ' '.join(current_clause).strip()
                if len(clause_text) > 50:
                    clause_type = self._classify_clause(clause_text)
                    keywords = self._extract_keywords(clause_text)
                    confidence = self._calculate_confidence(clause_text, clause_type)
                    
                    clause = ExtractedClause(
                        clause_type=clause_type,
                        content=clause_text,
                        section=self._extract_section(text, clause_text),
                        page_number=page_num,
                        confidence=confidence,
                        keywords=keywords
                    )
                    clauses.append(clause)
        
        except Exception as e:
            logger.error(f"Error extracting clauses: {e}")
        
        return clauses
    
    def _classify_clause(self, text: str) -> ClauseType:
        """Classify clause based on patterns"""
        best_match = ClauseType.GENERAL
        highest_score = 0
        
        for clause_type, patterns in self.clause_patterns.items():
            score = sum(len(re.findall(pattern, text)) for pattern in patterns)
            if score > highest_score:
                highest_score = score
                best_match = clause_type
        
        return best_match
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from clause"""
        try:
            doc = nlp(text)
            keywords = []
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ["ORG", "MONEY", "DATE", "PERCENT", "QUANTITY", "PERSON", "GPE"]:
                    keywords.append(ent.text.strip())
            
            # Extract important nouns and verbs
            for token in doc:
                if (token.pos_ in ["NOUN", "VERB"] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 3 and
                    token.text.isalpha()):
                    keywords.append(token.lemma_.lower())
            
            # Remove duplicates and limit
            return list(set(keywords))[:15]
        
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _extract_section(self, full_text: str, clause_text: str) -> str:
        """Extract section header for the clause"""
        try:
            clause_pos = full_text.find(clause_text[:100])  # Use first 100 chars for matching
            if clause_pos == -1:
                return "General Section"
            
            preceding_text = full_text[:clause_pos]
            lines = preceding_text.split('\n')
            
            section_patterns = [
                r'^\d+\.?\s+[A-Z].*',
                r'^[A-Z][A-Z\s]+$',
                r'^Article\s+\d+.*',
                r'^Section\s+\d+.*',
                r'^Chapter\s+\d+.*',
                r'^\d+\.\d+.*',
                r'^[IVX]+\.\s+.*'
            ]
            
            # Look for section headers in the last 10 lines
            for i in range(len(lines) - 1, max(0, len(lines) - 10), -1):
                line = lines[i].strip()
                if len(line) > 0 and len(line) < 100:
                    for pattern in section_patterns:
                        if re.match(pattern, line):
                            return line
            
            return "General Section"
        
        except Exception as e:
            logger.error(f"Error extracting section: {e}")
            return "Unknown Section"
    
    def _calculate_confidence(self, text: str, clause_type: ClauseType) -> float:
        """Calculate confidence score for clause classification"""
        try:
            patterns = self.clause_patterns.get(clause_type, [])
            matches = sum(len(re.findall(pattern, text)) for pattern in patterns)
            
            # Base confidence from pattern matches
            base_confidence = min(matches * 0.25, 0.8)
            
            # Length factor (longer clauses tend to be more reliable)
            length_factor = min(len(text) / 1000, 0.15)
            
            # Keyword density factor
            keywords = self._extract_keywords(text)
            keyword_factor = min(len(keywords) * 0.02, 0.1)
            
            total_confidence = base_confidence + length_factor + keyword_factor
            return min(max(total_confidence, 0.1), 1.0)  # Ensure between 0.1 and 1.0
        
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Process and ingest document into vector store"""
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file type and load document
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_extension == '.eml':
                loader = UnstructuredEmailLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            logger.info(f"Loading document: {file_path}")
            
            # Load documents
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content could be extracted from the document")
            
            # Extract full text for analysis
            full_text = "\n".join([doc.page_content for doc in documents])
            
            # Detect document type
            doc_type = self.detect_document_type(full_text)
            
            # Extract clauses from each page
            all_clauses = []
            enhanced_documents = []
            
            for i, doc in enumerate(documents):
                page_num = doc.metadata.get('page', i + 1)
                clauses = self.extract_clauses(doc.page_content, page_num)
                all_clauses.extend(clauses)
                
                # Enhanced metadata for each document chunk
                clause_info = [
                    {
                        "type": clause.clause_type.value,
                        "content_preview": clause.content[:100] + "..." if len(clause.content) > 100 else clause.content,
                        "keywords": clause.keywords[:5],  # Limit keywords in metadata
                        "confidence": clause.confidence
                    }
                    for clause in clauses
                ]
                
                doc.metadata.update({
                    "document_type": doc_type.value,
                    "file_name": os.path.basename(file_path),
                    "file_path": file_path,
                    "page_clauses": clause_info,
                    "num_clauses": len(clauses),
                    "ingestion_date": datetime.now().isoformat(),
                    "page_number": page_num
                })
                
                enhanced_documents.append(doc)
            
            # Split documents for vector store
            splits = self.text_splitter.split_documents(enhanced_documents)
            
            # Add to vector store
            logger.info(f"Adding {len(splits)} document chunks to vector store")
            self.vector_store.add_documents(splits)
            
            # Return ingestion summary
            result = {
                "status": "success",
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "document_type": doc_type.value,
                "total_pages": len(documents),
                "total_chunks": len(splits),
                "total_clauses": len(all_clauses),
                "clause_breakdown": self._summarize_clauses(all_clauses),
                "ingestion_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Document ingestion completed successfully: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "file_path": file_path,
                "error_timestamp": datetime.now().isoformat()
            }
    
    def _summarize_clauses(self, clauses: List[ExtractedClause]) -> Dict[str, int]:
        """Summarize clause types found"""
        summary = {}
        for clause in clauses:
            clause_type = clause.clause_type.value
            summary[clause_type] = summary.get(clause_type, 0) + 1
        return summary
    
    def query_with_context(self, query: str, k: int = 5) -> QueryResponse:
        """Enhanced query processing with contextual decision making"""
        try:
            logger.info(f"Processing query: {query}")
            
            # Search for relevant documents
            search_results = self.vector_store.similarity_search_with_score(query, k=k)
            
            if not search_results:
                return QueryResponse(
                    query=query,
                    answer="No relevant documents found in the knowledge base.",
                    relevant_clauses=[],
                    sources=[],
                    confidence_score=0.0,
                    decision_rationale="No matching documents were found for this query.",
                    recommendations=["Please ensure documents are properly ingested into the system."]
                )
            
            # Extract relevant clauses from search results
            relevant_clauses = []
            sources = []
            
            for doc, score in search_results:
                # Re-extract clauses from the content for more precise matching
                doc_clauses = self.extract_clauses(
                    doc.page_content, 
                    doc.metadata.get('page_number', 0)
                )
                
                # Filter clauses relevant to query
                query_doc = nlp(query.lower())
                query_tokens = set([token.lemma_ for token in query_doc 
                                   if not token.is_stop and len(token.text) > 2])
                
                for clause in doc_clauses:
                    clause_tokens = set([token.lower() for token in clause.keywords])
                    # Check for token overlap or direct content matching
                    if (query_tokens & clause_tokens or 
                        any(token in clause.content.lower() for token in query_tokens)):
                        clause.confidence = min(clause.confidence * (1 - score), 1.0)  # Adjust by search score
                        relevant_clauses.append(clause)
                
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": {
                        "file_name": doc.metadata.get('file_name', 'Unknown'),
                        "page_number": doc.metadata.get('page_number', 'N/A'),
                        "document_type": doc.metadata.get('document_type', 'general')
                    },
                    "relevance_score": float(1 - score)  # Convert distance to similarity
                })
            
            # Sort clauses by confidence
            relevant_clauses.sort(key=lambda x: x.confidence, reverse=True)
            relevant_clauses = relevant_clauses[:10]  # Top 10 most relevant clauses
            
            # Generate context for LLM
            context = self._build_context(search_results, relevant_clauses)
            
            # Generate answer using LLM
            answer, rationale, recommendations = self._generate_answer(
                query, context, relevant_clauses
            )
            
            # Calculate overall confidence
            confidence = self._calculate_query_confidence(search_results, relevant_clauses)
            
            response = QueryResponse(
                query=query,
                answer=answer,
                relevant_clauses=relevant_clauses,
                sources=sources,
                confidence_score=confidence,
                decision_rationale=rationale,
                recommendations=recommendations
            )
            
            logger.info(f"Query processed successfully with confidence: {confidence}")
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResponse(
                query=query,
                answer=f"An error occurred while processing your query: {str(e)}",
                relevant_clauses=[],
                sources=[],
                confidence_score=0.0,
                decision_rationale="System error occurred during processing.",
                recommendations=["Please try rephrasing your query or contact support."]
            )
    
    def _build_context(self, search_results: List, clauses: List[ExtractedClause]) -> str:
        """Build context for LLM from search results and clauses"""
        context_parts = []
        
        # Add document excerpts
        context_parts.append("=== RELEVANT DOCUMENT EXCERPTS ===")
        for i, (doc, score) in enumerate(search_results[:3]):  # Top 3 results
            context_parts.append(f"\n📄 Source {i+1}: {doc.metadata.get('file_name', 'Unknown')}")
            context_parts.append(f"📘 Page: {doc.metadata.get('page_number', 'N/A')}")
            context_parts.append(f"📊 Document Type: {doc.metadata.get('document_type', 'general')}")
            context_parts.append(f"🎯 Relevance Score: {1-score:.2f}")
            context_parts.append(f"Content: {doc.page_content}\n")
        
        # Add relevant clauses
        if clauses:
            context_parts.append("\n=== RELEVANT CLAUSES ===")
            for i, clause in enumerate(clauses[:5]):  # Top 5 clauses
                context_parts.append(f"\n🔍 Clause {i+1} - Type: {clause.clause_type.value}")
                context_parts.append(f"📝 Section: {clause.section}")
                context_parts.append(f"📄 Page: {clause.page_number}")
                context_parts.append(f"🎯 Confidence: {clause.confidence:.2f}")
                context_parts.append(f"💡 Content: {clause.content}")
                context_parts.append(f"🔑 Keywords: {', '.join(clause.keywords[:10])}\n")
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str, 
                        clauses: List[ExtractedClause]) -> tuple:
        """Generate answer using Azure OpenAI"""
        try:
            # Determine query intent
            query_intent = self._analyze_query_intent(query)
            
            # Build system prompt based on document types in context
            system_prompt = f"""You are an expert AI assistant specializing in {query_intent} analysis.
            You help users understand complex documents including insurance policies, legal contracts, 
            HR policies, and compliance documents.
            
            Your task is to:
            1. Answer the user's query accurately based on the provided context
            2. Cite specific clauses, sections, and page numbers when relevant
            3. Provide clear decision rationale explaining your reasoning
            4. Offer actionable recommendations
            
            Format your response as follows:
            ANSWER: [Your comprehensive answer here]
            
            RATIONALE: [Explain your reasoning and decision-making process]
            
            RECOMMENDATIONS: [Provide actionable recommendations, one per line]
            
            Always be precise, factual, and helpful. If information is unclear or missing, state that clearly.
            Reference specific page numbers and sections when available.
            """
            
            # Build user prompt
            user_prompt = f"""Based on the following context, please answer this query: "{query}"
            
            {context}
            
            Please provide a comprehensive response following the specified format."""
            
            # Get response from Azure OpenAI
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent responses
                max_tokens=1500
            )
            
            # Parse the response
            full_response = response.choices[0].message.content
            return self._parse_llm_response(full_response)
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return (
                "I apologize, but I encountered an error while generating the response.",
                f"System error: {str(e)}",
                ["Please try rephrasing your query or contact support."]
            )
    
    def _parse_llm_response(self, response_text: str) -> tuple:
        """Parse structured LLM response"""
        try:
            # Look for structured markers
            answer_match = re.search(r"ANSWER:\s*(.*?)(?=RATIONALE:|$)", response_text, re.DOTALL)
            rationale_match = re.search(r"RATIONALE:\s*(.*?)(?=RECOMMENDATIONS:|$)", response_text, re.DOTALL)
            recommendations_match = re.search(r"RECOMMENDATIONS:\s*(.*?)$", response_text, re.DOTALL)
            
            # Extract sections
            answer = answer_match.group(1).strip() if answer_match else response_text.split('\n')[0]
            rationale = rationale_match.group(1).strip() if rationale_match else "Based on document analysis and clause matching"
            recommendations_text = recommendations_match.group(1).strip() if recommendations_match else ""
            
            # Parse recommendations
            if recommendations_text:
                recommendations = [
                    r.strip().lstrip('- ').lstrip('• ').lstrip('* ')
                    for r in recommendations_text.split('\n') 
                    if r.strip()
                ]
            else:
                recommendations = ["Review the complete source documents for additional details"]
            
            # Clean up the answer
            answer = re.sub(r'\n+', ' ', answer).strip()
            rationale = re.sub(r'\n+', ' ', rationale).strip()
            
            return answer, rationale, recommendations
        
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return (
                response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "Analysis completed based on available context",
                ["Review source documents for complete information"]
            )
    
    def _analyze_query_intent(self, query: str) -> str:
        """Analyze the intent of the query"""
        query_lower = query.lower()
        
        intent_patterns = {
            "insurance coverage": ["coverage", "cover", "included", "benefit", "protected", "insured"],
            "policy exclusions": ["exclude", "exclusion", "not cover", "not included", "except"],
            "payment and financial": ["pay", "payment", "premium", "cost", "fee", "price", "financial"],
            "termination and cancellation": ["terminate", "cancel", "end", "expire", "dissolution"],
            "liability and responsibility": ["liable", "liability", "responsible", "accountable", "fault"],
            "compliance and obligations": ["comply", "obligation", "must", "shall", "required", "duty"],
            "confidentiality": ["confident", "secret", "private", "proprietary", "disclosure"],
            "dispute resolution": ["dispute", "arbitration", "mediation", "court", "legal action"]
        }
        
        best_intent = "document analysis"
        max_matches = 0
        
        for intent, keywords in intent_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > max_matches:
                max_matches = matches
                best_intent = intent
        
        return best_intent
    
    def _calculate_query_confidence(self, search_results: List, 
                                  clauses: List[ExtractedClause]) -> float:
       """Calculate confidence score for query response"""
       try:
           if not search_results:
               return 0.0
           
           # Base confidence on search scores (convert distance to similarity)
           search_scores = [1 - score for _, score in search_results[:3]]
           avg_search_relevance = sum(search_scores) / len(search_scores)
           
           # Boost confidence based on number and quality of relevant clauses
           if clauses:
               clause_confidence = sum(clause.confidence for clause in clauses[:5]) / min(5, len(clauses))
               clause_boost = min(len(clauses) * 0.05, 0.2)
           else:
               clause_confidence = 0.0
               clause_boost = 0.0
           
           # Combine factors
           final_confidence = (avg_search_relevance * 0.6) + (clause_confidence * 0.3) + clause_boost
           
           # Ensure confidence is between 0.1 and 1.0
           return min(max(final_confidence, 0.1), 1.0)
       
       except Exception as e:
           logger.error(f"Error calculating query confidence: {e}")
           return 0.5
   
    def get_document_summary(self) -> Dict[str, Any]:
       """Get summary of all ingested documents"""
       try:
           # This would require implementing a method to retrieve document metadata
           # For now, return a placeholder
           return {
               "status": "success",
               "message": "Use query_with_context to search documents",
               "timestamp": datetime.now().isoformat()
           }
       except Exception as e:
           logger.error(f"Error getting document summary: {e}")
           return {"status": "error", "error_message": str(e)}
   
    def delete_collection(self) -> Dict[str, Any]:
       """Delete the entire Qdrant collection (use with caution)"""
       try:
           qdrant_client = QdrantClient(url=QDRANT_URL)
           qdrant_client.delete_collection(COLLECTION_NAME)
           logger.info(f"Collection {COLLECTION_NAME} deleted successfully")
           return {
               "status": "success",
               "message": f"Collection {COLLECTION_NAME} deleted successfully",
               "timestamp": datetime.now().isoformat()
           }
       except Exception as e:
           logger.error(f"Error deleting collection: {e}")
           return {"status": "error", "error_message": str(e)}

# Usage example and main execution
def main():
   """Main function demonstrating system usage"""
   # Initialize the RAG system
   print("🚀 Initializing Intelligent RAG System...")
   rag_system = IntelligentRAGSystem()
   
   # Example document ingestion
   print("\n📁 Example: Ingesting a document...")
   # Uncomment and modify the path below to test with your documents
   # result = rag_system.ingest_document("path/to/your/document.pdf")
   # print("Ingestion Result:")
   # print(json.dumps(result, indent=2))
   
   # Example query
   print("\n🔍 Example: Querying the system...")
   # response = rag_system.query_with_context("What coverage is provided under this policy?")
   # print("Query Response:")
   # print(response.to_json())
   
   print("\n✅ System initialized successfully!")
   print("💡 To use the system:")
   print("   1. Call ingest_document(file_path) to add documents")
   print("   2. Call query_with_context(query) to ask questions")
   print("   3. Use response.to_json() to get structured JSON output")

if __name__ == "__main__":
   main()

# Additional utility functions
class RAGSystemUtils:
   """Utility functions for the RAG system"""
   
   @staticmethod
   def validate_file_format(file_path: str) -> bool:
       """Validate if file format is supported"""
       supported_extensions = ['.pdf', '.docx', '.eml']
       return Path(file_path).suffix.lower() in supported_extensions
   
   @staticmethod
   def batch_ingest_documents(rag_system: IntelligentRAGSystem, 
                             directory_path: str) -> List[Dict[str, Any]]:
       """Batch ingest all supported documents from a directory"""
       results = []
       directory = Path(directory_path)
       
       if not directory.exists():
           return [{"status": "error", "error_message": f"Directory not found: {directory_path}"}]
       
       for file_path in directory.iterdir():
           if file_path.is_file() and RAGSystemUtils.validate_file_format(str(file_path)):
               print(f"Processing: {file_path.name}")
               result = rag_system.ingest_document(str(file_path))
               results.append(result)
       
       return results
   
   @staticmethod
   def export_query_response(response: QueryResponse, output_path: str):
       """Export query response to a JSON file"""
       try:
           with open(output_path, 'w', encoding='utf-8') as f:
               f.write(response.to_json())
           print(f"Response exported to: {output_path}")
       except Exception as e:
           print(f"Error exporting response: {e}")
   
   @staticmethod
   def create_query_report(responses: List[QueryResponse], 
                          output_path: str):
       """Create a comprehensive report from multiple queries"""
       try:
           report = {
               "report_metadata": {
                   "generated_at": datetime.now().isoformat(),
                   "total_queries": len(responses),
                   "system_version": "1.0"
               },
               "queries": [response.to_dict() for response in responses],
               "summary": {
                   "avg_confidence": sum(r.confidence_score for r in responses) / len(responses) if responses else 0,
                   "total_clauses_analyzed": sum(len(r.relevant_clauses) for r in responses),
                   "unique_document_types": list(set(
                       source.get("metadata", {}).get("document_type", "unknown")
                       for response in responses
                       for source in response.sources
                   ))
               }
           }
           
           with open(output_path, 'w', encoding='utf-8') as f:
               json.dump(report, f, indent=2, ensure_ascii=False)
           
           print(f"Query report exported to: {output_path}")
       except Exception as e:
           print(f"Error creating query report: {e}")

# Example usage script
def example_usage():
   """Example script showing how to use the system"""
   
   # Initialize system
   rag_system = IntelligentRAGSystem()
   
   # Example 1: Single document ingestion
   print("=== Example 1: Document Ingestion ===")
   # document_path = "sample_insurance_policy.pdf"
   # if os.path.exists(document_path):
   #     result = rag_system.ingest_document(document_path)
   #     print(json.dumps(result, indent=2))
   
   # Example 2: Batch document ingestion
   print("\n=== Example 2: Batch Ingestion ===")
   # documents_directory = "documents/"
   # if os.path.exists(documents_directory):
   #     results = RAGSystemUtils.batch_ingest_documents(rag_system, documents_directory)
   #     for result in results:
   #         print(f"File: {result.get('file_name', 'Unknown')} - Status: {result['status']}")
   
   # Example 3: Various types of queries
   print("\n=== Example 3: Sample Queries ===")
   sample_queries = [
       "What coverage is provided under this insurance policy?",
       "Are there any exclusions I should be aware of?",
       "What are the payment terms and conditions?",
       "How can this policy be terminated?",
       "What are my obligations under this contract?"
   ]
   
   responses = []
   for query in sample_queries:
       print(f"\nQuery: {query}")
       # response = rag_system.query_with_context(query)
       # responses.append(response)
       # print(f"Answer: {response.answer[:100]}...")
       # print(f"Confidence: {response.confidence_score:.2f}")
       # print(f"Relevant Clauses: {len(response.relevant_clauses)}")
   
   # Example 4: Export results
   print("\n=== Example 4: Export Results ===")
   # if responses:
   #     RAGSystemUtils.create_query_report(responses, "query_report.json")
   #     RAGSystemUtils.export_query_response(responses[0], "sample_response.json")
   
   print("\n✅ Example usage completed!")

if __name__ == "__main__":
   # Run the main function
   main()
   
   # Uncomment to run the example usage
   # example_usage()