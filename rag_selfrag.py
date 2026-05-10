"""
Self-RAG Implementation with LangGraph

More sophisticated grading approach with:
- Document relevance grading (ISREL)
- Generation support grading (ISSUP) 
- Usefulness grading (ISUSE)
- Automatic re-retrieval on poor quality
"""

from typing import Literal, List
from pydantic import BaseModel, Field
from langchain_community.document_loaders import (
    PyPDFLoader, DirectoryLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import TypedDict
import os
import json

# ============================================================================
# Self-RAG Grading Models
# ============================================================================

class ISRELGrade(BaseModel):
    """Document relevance grading (ISREL token)"""
    relevant: str = Field(
        description="Is the document relevant to the question? 'relevant' or 'irrelevant'"
    )

class ISSUPGrade(BaseModel):
    """Generation support grading (ISSUP token)"""
    supported: str = Field(
        description="Is the generation supported by the document? 'fully_supported', 'partially_supported', or 'no_support'"
    )

class ISUSEGrade(BaseModel):
    """Usefulness grading (ISUSE token)"""
    score: int = Field(
        description="How useful is this generation? Score 1-5 where 5 is most useful"
    )

# ============================================================================
# State Definition
# ============================================================================

class SelfRAGState(TypedDict):
    """State for Self-RAG graph"""
    question: str
    documents: List[Document]
    relevant_documents: List[Document]
    generation: str
    generation_supported: bool
    generation_useful: bool
    rewrite_count: int
    max_rewrites: int

# ============================================================================
# Document Processor (reusable)
# ============================================================================

class DocumentProcessor:
    """Load and process documents"""
    
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore = None
        self.retriever = None
    
    def load_documents(self, path: str) -> List[Document]:
        """Load documents from file or directory"""
        documents = []
        
        if os.path.isfile(path):
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
                documents = loader.load()
            elif path.endswith('.txt'):
                loader = TextLoader(path)
                documents = loader.load()
        elif os.path.isdir(path):
            pdf_loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
            txt_loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
            documents.extend(pdf_loader.load())
            documents.extend(txt_loader.load())
        
        return documents
    
    def create_index(self, documents: List[Document]):
        """Create FAISS vector index"""
        splits = self.splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        print(f"✓ Indexed {len(splits)} document chunks")

# ============================================================================
# Self-RAG Nodes
# ============================================================================

def retrieve_documents(state: SelfRAGState, retriever) -> Command:
    """Retrieve documents from vectorstore"""
    print(f"\n📖 Retrieving documents...")
    
    documents = retriever.invoke(state["question"])
    
    return Command(
        update={
            "documents": documents
        }
    )


def grade_documents_isrel(state: SelfRAGState, llm) -> Command:
    """Grade each document for relevance (ISREL)"""
    print(f"\n⭐ Grading document relevance (ISREL)...")
    
    question = state["question"]
    relevant_docs = []
    
    for i, doc in enumerate(state["documents"]):
        prompt = f"""You are a relevance grader. Assess if the following document is relevant to the user question.

Question: {question}

Document:
{doc.page_content[:500]}

Is this document relevant? Answer only 'relevant' or 'irrelevant'."""
        
        response = llm.invoke(prompt)
        is_relevant = "relevant" in response.content.lower() and "irrelevant" not in response.content.lower()
        
        status = "✓" if is_relevant else "✗"
        print(f"  Doc {i}: {status}")
        
        if is_relevant:
            relevant_docs.append(doc)
    
    print(f"  → {len(relevant_docs)}/{len(state['documents'])} documents relevant")
    
    return Command(
        update={
            "relevant_documents": relevant_docs
        }
    )


def generate_answer(state: SelfRAGState, llm) -> Command:
    """Generate answer from relevant documents"""
    print(f"\n✍️  Generating answer...")
    
    if not state["relevant_documents"]:
        print("  ⚠️  No relevant documents found")
        generation = "I could not find relevant information to answer this question."
    else:
        question = state["question"]
        docs_str = "\n\n".join([f"[{i}] {doc.page_content}" 
                                 for i, doc in enumerate(state["relevant_documents"])])
        
        prompt = f"""Answer the user question based ONLY on the provided documents. 
Be concise and accurate.

Question: {question}

Documents:
{docs_str}

Answer:"""
        
        response = llm.invoke(prompt)
        generation = response.content.strip()
    
    print(f"  Generated: {generation[:150]}...")
    
    return Command(
        update={
            "generation": generation
        }
    )


def grade_generation_issup(state: SelfRAGState, llm) -> Command:
    """Grade if generation is supported by documents (ISSUP)"""
    print(f"\n🔍 Checking if generation is supported by documents (ISSUP)...")
    
    if not state["relevant_documents"]:
        return Command(update={"generation_supported": False})
    
    docs_str = "\n\n".join([doc.page_content for doc in state["relevant_documents"]])
    
    prompt = f"""You are a support grader. Assess if the generated answer is fully supported by the provided documents.

Documents:
{docs_str}

Generated Answer:
{state["generation"]}

Is the answer fully supported by the documents? Answer only 'yes' or 'no'."""
    
    response = llm.invoke(prompt)
    is_supported = "yes" in response.content.lower()
    
    status = "✓" if is_supported else "✗"
    print(f"  {status} Generation supported by documents")
    
    return Command(
        update={
            "generation_supported": is_supported
        }
    )


def grade_generation_isuse(state: SelfRAGState, llm) -> Command:
    """Grade usefulness of generation (ISUSE)"""
    print(f"\n📊 Grading usefulness of generation (ISUSE)...")
    
    prompt = f"""You are a usefulness grader. Rate how useful the generated answer is for answering the original question.

Question: {state["question"]}

Generated Answer:
{state["generation"]}

Rate usefulness on a scale of 1-5 where:
5 = Fully answers the question with good detail
4 = Mostly answers the question
3 = Partially answers the question
2 = Minimally relevant to the question
1 = Not useful for answering the question

Respond with only the number."""
    
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip()[-1])
    except:
        score = 3
    
    is_useful = score >= 4
    print(f"  Score: {score}/5 - {'✓ Useful' if is_useful else '✗ Not useful'}")
    
    return Command(
        update={
            "generation_useful": is_useful
        }
    )


def rewrite_query(state: SelfRAGState, llm) -> Command:
    """Rewrite query for better retrieval"""
    print(f"\n🔄 Rewriting query...")
    
    prompt = f"""You are a query optimizer. The previous retrieval didn't find enough relevant documents.
Rewrite the user question to be more specific and search-friendly.

Original question: {state["question"]}

Rewritten question:"""
    
    response = llm.invoke(prompt)
    new_question = response.content.strip()
    
    print(f"  Original: {state['question']}")
    print(f"  Rewritten: {new_question}")
    
    return Command(
        update={
            "question": new_question,
            "rewrite_count": state["rewrite_count"] + 1
        }
    )


# ============================================================================
# Conditional Routing
# ============================================================================

def should_rewrite_or_end(state: SelfRAGState) -> Literal["retrieve", "end"]:
    """Decide to re-retrieve with rewritten query or end"""
    max_rewrites = state["max_rewrites"]
    
    # End conditions
    if state["generation_supported"] and state["generation_useful"]:
        print("\n✅ Generation is good! Ending.")
        return "end"
    
    if state["rewrite_count"] >= max_rewrites:
        print(f"\n⚠️  Max rewrites ({max_rewrites}) reached. Ending.")
        return "end"
    
    # If not supported or not useful, rewrite and retry
    if not state["generation_supported"] or not state["generation_useful"]:
        print("\n🔁 Generation needs improvement. Rewriting query...")
        return "retrieve"
    
    return "end"


# ============================================================================
# Build Self-RAG Graph
# ============================================================================

def build_selfrag_graph(retriever, llm) -> StateGraph:
    """Build the Self-RAG workflow graph"""
    
    graph = StateGraph(SelfRAGState)
    
    # Add nodes
    graph.add_node(
        "retrieve",
        lambda state: retrieve_documents(state, retriever)
    )
    graph.add_node(
        "grade_documents",
        lambda state: grade_documents_isrel(state, llm)
    )
    graph.add_node(
        "generate",
        lambda state: generate_answer(state, llm)
    )
    graph.add_node(
        "grade_support",
        lambda state: grade_generation_issup(state, llm)
    )
    graph.add_node(
        "grade_usefulness",
        lambda state: grade_generation_isuse(state, llm)
    )
    graph.add_node(
        "rewrite",
        lambda state: rewrite_query(state, llm)
    )
    
    # Build edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("grade_documents", "generate")
    graph.add_edge("generate", "grade_support")
    graph.add_edge("grade_support", "grade_usefulness")
    graph.add_conditional_edges(
        "grade_usefulness",
        should_rewrite_or_end,
        {
            "retrieve": "rewrite",
            "end": END
        }
    )
    graph.add_edge("rewrite", "retrieve")
    
    return graph.compile()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point"""
    
    print("=" * 60)
    print("Self-RAG with LangGraph")
    print("=" * 60)
    
    # Check for documents
    doc_path = "./documents"
    if not os.path.exists(doc_path):
        print(f"\n⚠️  No documents found in '{doc_path}'")
        return
    
    # Initialize components
    print("\n1️⃣  Loading and indexing documents...")
    processor = DocumentProcessor(embedding_model="nomic-embed-text")
    documents = processor.load_documents(doc_path)
    
    if not documents:
        print("No documents found.")
        return
    
    processor.create_index(documents)
    
    print("\n2️⃣  Setting up local Ollama LLM...")
    llm = ChatOllama(
        model="mistral",
        temperature=0,
        num_ctx=4096
    )
    
    print("\n3️⃣  Building Self-RAG graph...")
    app = build_selfrag_graph(processor.retriever, llm)
    
    # Test queries
    print("\n" + "=" * 60)
    print("Testing Self-RAG System")
    print("=" * 60)
    
    test_questions = [
        "What is the main topic of the documents?",
        "Explain the key concepts discussed"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        initial_state = {
            "question": question,
            "documents": [],
            "relevant_documents": [],
            "generation": "",
            "generation_supported": False,
            "generation_useful": False,
            "rewrite_count": 0,
            "max_rewrites": 2
        }
        
        # Run the graph
        result = app.invoke(initial_state)
        
        print(f"\n📋 Final Answer:")
        print(result["generation"])
        print(f"\nℹ️  Sources: {len(result['relevant_documents'])} relevant documents")


if __name__ == "__main__":
    main()