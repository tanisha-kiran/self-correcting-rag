"""
Self-Reflective RAG with LangGraph
Implements both CRAG and Self-RAG approaches with local Ollama
"""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from langchain_community.document_loaders import (
    PyPDFLoader, DirectoryLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import TypedDict
import json
import os


# ============================================================================

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="yes or no")


class RAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search_needed: bool
    retrieve_attempts: int
    rewrite_attempts: int


# ============================================================================

class DocumentProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        self.vectorstore = None
        self.retriever = None

    def load_documents(self, path: str) -> List[Document]:
        documents = []

        if os.path.isfile(path):
            if path.endswith(".pdf"):
                documents = PyPDFLoader(path).load()
            elif path.endswith(".txt"):
                documents = TextLoader(path).load()

        elif os.path.isdir(path):
            documents.extend(DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader).load())
            documents.extend(DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader).load())

        return documents

    def create_index(self, documents: List[Document]):
        splits = self.splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        print(f"✓ Indexed {len(splits)} chunks")


# ============================================================================

def setup_llm(model="tinyllama", temperature=0):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
        num_ctx=4096
    )
    return llm


# ============================================================================

def retrieve_documents(state: RAGState, retriever) -> Command:
    print("\n📖 Retrieving...")
    docs = retriever.invoke(state["question"])

    return Command(update={
        "documents": docs,
        "retrieve_attempts": state.get("retrieve_attempts", 0) + 1
    })


# ============================================================================

def grade_documents(state: RAGState, llm) -> Command:
    print("⭐ Grading documents...")

    question = state["question"]
    docs_str = "\n\n".join([doc.page_content[:300] for doc in state["documents"]])

    prompt = f"""
You are a strict grader.

Question: {question}

Documents:
{docs_str}

Answer ONLY yes or no:
Are these documents relevant?
"""

    response = llm.invoke(prompt)

    is_relevant = "yes" in response.content.lower()

    return Command(update={
        "web_search_needed": not is_relevant
    })


# ============================================================================

def rewrite_query(state: RAGState, llm) -> Command:
    print("🔄 Rewriting query...")

    prompt = f"""
Rewrite this question to improve retrieval:

{state['question']}
"""

    response = llm.invoke(prompt)

    return Command(update={
        "question": response.content.strip(),
        "rewrite_attempts": state.get("rewrite_attempts", 0) + 1
    })


# ============================================================================

def web_search(state: RAGState) -> Command:
    print("🌐 Web search...")

    if not os.getenv("TAVILY_API_KEY"):
        return Command(update={"documents": state.get("documents", [])})

    search = TavilySearchResults(max_results=3)
    results = search.invoke(state["question"])

    docs = [
        Document(page_content=r.get("content", ""), metadata={"source": r.get("url")})
        for r in results
    ]

    return Command(update={
        "documents": docs,
        "web_search_needed": False
    })


# ============================================================================

def generate_answer(state: RAGState, llm) -> Command:
    print("✍️ Generating...")

    docs_str = "\n\n".join([d.page_content for d in state["documents"]])

    prompt = f"""
Answer ONLY using context.

Question: {state['question']}

Context:
{docs_str}

Answer:
"""

    response = llm.invoke(prompt)

    return Command(update={
        "generation": response.content
    })


# ============================================================================

def should_web_search(state: RAGState):
    return "web_search" if state.get("web_search_needed") else "generate"


def should_retrieve_again(state: RAGState):
    return "retrieve" if state.get("rewrite_attempts", 0) < 2 else "end"


# ============================================================================

def build_crag_graph(retriever, llm):
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", lambda s: retrieve_documents(s, retriever))
    graph.add_node("grade", lambda s: grade_documents(s, llm))
    graph.add_node("rewrite", lambda s: rewrite_query(s, llm))
    graph.add_node("web_search", lambda s: web_search(s))
    graph.add_node("generate", lambda s: generate_answer(s, llm))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        should_web_search,
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )

    graph.add_edge("web_search", "generate")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


# ============================================================================

def main():
    print("=" * 50)

    doc_path = "./documents"
    if not os.path.exists(doc_path):
        print("Add documents first")
        return

    processor = DocumentProcessor()
    docs = processor.load_documents(doc_path)

    processor.create_index(docs)

    print("🔧 Loading TinyLlama...")
    llm = setup_llm(model="tinyllama")

    print("⚙️ Building graph...")
    app = build_crag_graph(processor.retriever, llm)

    questions = [
        "What are the main topics?",
        "Summarize key ideas"
    ]

    for q in questions:
        print("\nQ:", q)

        state = {
            "question": q,
            "documents": [],
            "generation": "",
            "web_search_needed": False,
            "retrieve_attempts": 0,
            "rewrite_attempts": 0
        }

        result = app.invoke(state)

        print("\nAnswer:", result["generation"])


if __name__ == "__main__":
    main()