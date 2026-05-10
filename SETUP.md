# Self-Reflective RAG with LangGraph - Setup Guide

## 📋 Overview

Two complete implementations of advanced RAG systems using LangGraph:

1. **CRAG (Corrective RAG)** - `rag_system.py`
   - Retrieves documents and grades their relevance
   - Falls back to web search if needed
   - Rewrites queries for better results

2. **Self-RAG** - `rag_selfrag.py`  
   - Fine-grained grading of documents and generations
   - Checks if answer is supported by documents
   - Rates usefulness and auto-refines

## 🛠️ Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install & Run Ollama

For macOS/Linux:
```bash
# Install from https://ollama.ai
ollama serve
```

For Windows, download from https://ollama.ai/download/windows

### Step 3: Pull a Model

In a new terminal:
```bash
ollama pull mistral
# Or try: ollama pull neural-chat, ollama pull orca-mini
```

## 📁 Project Structure

```
.
├── rag_system.py          # CRAG implementation
├── rag_selfrag.py         # Self-RAG implementation  
├── requirements.txt       # Python dependencies
├── documents/             # Your documents go here
│   ├── doc1.pdf
│   ├── doc2.txt
│   └── ...
└── README.md
```

## 📚 Adding Your Documents

1. Create a `documents/` directory in the same folder as the scripts
2. Add your PDF or TXT files:

```bash
mkdir documents
cp /path/to/your/files/* documents/
```

Supported formats:
- `.pdf` - PDF documents
- `.txt` - Plain text files
- `.md` - Markdown files (save as .txt)

## 🚀 Running the Systems

### CRAG System

```bash
python rag_system.py
```

Output:
```
==============================================
Self-Reflective RAG with LangGraph (CRAG)
==============================================

1️⃣  Loading and indexing documents...
✓ Indexed 25 document chunks

2️⃣  Setting up local Ollama LLM...
  (Make sure Ollama is running: 'ollama serve')

3️⃣  Building CRAG graph...

============================================================
Question: What are the main topics?
============================================================

📖 Retrieving documents for: What are the main topics?
⭐ Grading 4 documents...
  → Documents relevant: yes
✍️  Generating answer...
  Generated: The main topics covered include...

📋 Final Answer:
The documents discuss several key areas...

ℹ️  Sources: 4 documents
```

### Self-RAG System

```bash
python rag_selfrag.py
```

Output includes detailed grading at each stage:
```
📖 Retrieving documents...
⭐ Grading document relevance (ISREL)...
  Doc 0: ✓
  Doc 1: ✓
  Doc 2: ✗
  → 3/4 documents relevant

✍️  Generating answer...
  Generated: The answer is...

🔍 Checking if generation is supported (ISSUP)...
  ✓ Generation supported by documents

📊 Grading usefulness (ISUSE)...
  Score: 5/5 - ✓ Useful

📋 Final Answer:
```

## 🔌 Optional: Enable Web Search

For CRAG web search fallback:

1. Get a Tavily API key from https://tavily.com
2. Set environment variable:

```bash
export TAVILY_API_KEY="your_key_here"
```

Without this, the system will gracefully skip web search.

## 🎯 Architecture Comparison

### CRAG Flow
```
Retrieve → Grade Docs → Web Search? 
              ↓          ↓
            YES        NO
              ↓          ↓
         Web Search   Generate
              ↓          ↓
               ← Generate ←
                    ↓
                  END
```

### Self-RAG Flow
```
Retrieve → Grade (ISREL) → Generate → Grade (ISSUP) → Grade (ISUSE)
                                            ↓              ↓
                                         Pass?          Useful?
                                      ↙     ↖         ↙    ↖
                                    YES     NO      YES    NO
                                     ↓               ↓      ↓
                                    END          Rewrite or END
                                                    ↓
                                                Retrieve
```

## 🧪 Testing with Sample Documents

Create a test document:

```bash
cat > documents/test.txt << 'EOF'
# AI and Machine Learning

## Introduction
Artificial intelligence (AI) is transforming industries worldwide.
Machine learning (ML) is a subset of AI that focuses on data-driven learning.

## Key Concepts
- Neural Networks: Inspired by biological neural systems
- Deep Learning: Multiple layers of neural networks
- Natural Language Processing: Understanding human language
- Computer Vision: Understanding images and video

## Applications
AI and ML are used in:
- Healthcare diagnostics
- Autonomous vehicles
- Recommendation systems
- Natural language understanding
EOF
```

Then test with questions like:
- "What is machine learning?"
- "What are applications of AI?"
- "Explain neural networks"

## ⚙️ Configuration

### Model Selection

Change the model in the code:
```python
llm = ChatOllama(
    model="mistral",  # Change to neural-chat, orca-mini, etc.
    temperature=0,     # 0 = deterministic, 1 = creative
    num_ctx=4096      # Context window size
)
```

Available models:
- `mistral` - Fast, good quality (recommended)
- `neural-chat` - Optimized for chat
- `orca-mini` - Smaller, faster
- `llama2` - Meta's Llama 2
- `dolphin-mixtral` - Higher quality

### Chunk Size

In DocumentProcessor:
```python
self.splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Larger = more context
    chunk_overlap=200     # Overlap helps with boundaries
)
```

### Retrieval Settings

In DocumentProcessor:
```python
self.retriever = self.vectorstore.as_retriever(
    search_kwargs={"k": 4}  # Number of documents to retrieve
)
```

## 🐛 Troubleshooting

### "Connection refused" when running
- Make sure Ollama is running: `ollama serve`
- Check it's accessible: `curl http://localhost:11434`

### "Model 'mistral' not found"
- Pull the model: `ollama pull mistral`
- List available models: `ollama list`

### "No documents found"
- Check documents are in `./documents/` directory
- Supported formats: .pdf, .txt only
- Check file permissions

### Out of memory
- Reduce `num_ctx` in ChatOllama (e.g., 2048)
- Reduce `chunk_size` in RecursiveCharacterTextSplitter
- Use smaller model (e.g., orca-mini)

### Slow performance
- Use smaller model (orca-mini instead of mistral)
- Reduce chunk_size and chunk_overlap
- Reduce number of documents retrieved (k=2 or k=3)

## 🎓 Learning Resources

- [LangGraph Docs](https://python.langchain.com/docs/langgraph)
- [CRAG Paper](https://arxiv.org/pdf/2401.15884.pdf)
- [Self-RAG Paper](https://arxiv.org/pdf/2310.11511.pdf)
- [Ollama Models](https://ollama.ai/library)

## 📝 Next Steps

1. **Custom Document Loader**: Extend to load from URLs, databases, etc.
2. **Persistent Storage**: Store embeddings to avoid re-indexing
3. **Multi-turn Conversations**: Add chat history to the state
4. **Custom Grading**: Create domain-specific grading prompts
5. **Evaluation**: Add metrics to measure RAG quality
6. **API Server**: Wrap with FastAPI for production deployment

## 📧 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify Ollama is running and has the model
3. Check that documents are in the correct format
4. Review the output for specific error messages

---

**Happy RAG building! 🚀**