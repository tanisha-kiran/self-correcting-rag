# Agentic Multimodal RAG Framework for Privacy-Preserving Document Intelligence

**Phase 1: Foundational Text-Based RAG Architecture**

A research implementation establishing the core retrieval-augmented generation capabilities for a comprehensive sovereign AI platform designed to process and understand complex documents while maintaining complete data privacy.

## Executive Summary

This project represents the foundational layer of an enterprise-grade document intelligence system that achieves centralized performance while ensuring 100% data privacy and operational resilience in air-gapped environments. Our research proposes an offline, agentic multimodal RAG framework that leverages:

- Advanced reasoning through agentic orchestration via LangGraph
- On-device processing with 4-bit quantized Vision-Language Models
- Privacy-preserving document analysis without cloud dependency
- Autonomous query refinement and multi-step verification
- Cross-modal alignment fine-tuned on document understanding datasets
- Complete sovereignty over data processing and model inference

This Phase 1 implementation validates the foundational text-based RAG architecture that will serve as the backbone for the complete multimodal system.

---

## Strategic Vision and Roadmap

### Phase 1: Foundational Text-Based RAG Architecture (Current)

Establishing core retrieval and generation capabilities with local language models.

**Current Capabilities**:
- CRAG (Corrective RAG): Document relevance evaluation with web search fallback
- Self-RAG: Reflection tokens for quality assessment (RETRIEVE, ISREL, ISSUP, ISUSE)
- Adaptive-RAG: Complexity-based routing for efficiency optimization
- Local-first architecture using Ollama
- Document ingestion pipeline for text extraction and chunking
- Vector-based retrieval using FAISS and embeddings
- Real-time query processing and answer generation

**Research Foundation**:
- Self-RAG (Asai et al., 2023): Fine-grained quality control through reflection tokens
- Adaptive-RAG (Jeong et al., 2024): Intelligent routing based on query complexity
- CRAG (Shi et al., 2024): Corrective mechanisms for retrieval errors

**Validation Status**: Operational and tested on document collections

### Phase 2: Multimodal Document Understanding

Extending the system to interpret complex, spatially-dependent visual and structural content.

**Planned Components**:

1. Vision-Language Model Integration
   - 4-bit quantized VLM deployment for efficiency
   - Image understanding and analysis
   - OCR capabilities for scanned documents
   - Cross-modal embedding alignment

2. Structured Content Processing
   - Hierarchical table parsing and interpretation
   - Schematic diagram analysis and understanding
   - Chart and graph comprehension
   - Layout-aware text extraction

3. Fine-tuning on Document Understanding
   - QLoRA-based fine-tuning for domain adaptation
   - Training on DocVQA and similar datasets
   - Cross-modal alignment optimization
   - Transfer learning from pretrained VLMs

4. Multimodal Fusion
   - Seamless integration of text and visual understanding
   - Joint reasoning over multiple modalities
   - Spatial relationship understanding
   - Context-aware content interpretation

**Expected Outcomes**:
- Support for documents with mixed content (text, tables, images, diagrams)
- Accurate extraction of information from complex layouts
- Improved performance on spatially-dependent queries
- Foundation for agentic reasoning across modalities

### Phase 3: Agentic Reasoning and Orchestration 

Implementing autonomous agents for complex query resolution and multi-step verification.

**Planned Components**:

1. LangGraph-Based Orchestration
   - Workflow state management and persistence
   - Autonomous query decomposition
   - Dynamic tool selection and execution
   - Error recovery and adaptation

2. Agentic Reasoning Capabilities
   - Autonomous query refinement and reformulation
   - Multi-step verification workflows
   - Evidence aggregation across documents
   - Confidence assessment and uncertainty handling

3. Specialized Processing Agents
   - Document content analyzer (text, tables, images)
   - Citation verification agent
   - Cross-document reasoning agent
   - Quality assurance and validation agent

4. Advanced Verification Framework
   - Multi-step verification chains
   - Consistency checking across sources
   - Confidence scoring for answers
   - Audit trails for reasoning process

**Architecture**:
```
User Query
    |
    v
Query Decomposition Agent
    |
    +---> Document Retrieval Agent
    |         |
    |         v
    |     Multimodal Content Analyzer
    |         |
    |         v
    |     Information Extraction
    |
    +---> Evidence Aggregation Agent
    |         |
    |         v
    |     Cross-Document Reasoning
    |
    +---> Verification Agent
            |
            v
        Quality Assessment
            |
            v
        Final Answer with Citations
```

**Key Technologies**:
- LangGraph for agentic workflow orchestration
- Tool-use framework for specialized processors
- Reflection tokens for quality control
- State machines for complex reasoning patterns

## Current Implementation Details

### System Overview

This Phase 1 implementation provides  complementary RAG approaches, each optimized for different use cases:

**CRAG (Corrective RAG)**
- Document relevance evaluation before generation
- Web search fallback mechanism (optional)
- Query rewriting for improved retrieval
- Optimized for general question-answering

**Self-RAG**
- Reflection tokens for fine-grained quality assessment
- Four token types: RETRIEVE, ISREL, ISSUP, ISUSE
- Segment-level evaluation and ranking
- Best for citation-heavy and quality-critical applications


### Architecture

```
Input Documents
    |
    v
Text Extraction & Chunking
    |
    v
Vector Embedding (Ollama)
    |
    v
FAISS Vector Index
    |
    v
Query Processing
    |
    |
    +---> Retrieval Engine
    |         |
    |         v
    |     Document Ranking
    |
    +---> Self-RAG Evaluation
    |         |
    |         v
    |     Reflection Token Generation
    |         |
    |         v
    |     Quality Verification
    |
    +---> Answer Generation
            |
            v
        Output with Confidence
```

### Reflection Tokens

The Self-RAG component uses four types of tokens for quality assessment:

**RETRIEVE Token**: Determines if external documents are needed
- Values: yes, no, continue
- Enables on-demand retrieval based on content

**ISREL Token**: Assesses document relevance to query
- Values: relevant, irrelevant
- Filters documents before processing

**ISSUP Token**: Verifies generation support by documents
- Values: fully_supported, partially_supported, no_support
- Prevents hallucination through evidence verification

**ISUSE Token**: Rates answer usefulness
- Scale: 1-5 (1: useless, 5: excellent)
- Triggers refinement when below threshold

---

## Technical Stack

### Core Dependencies

```
Language Models
- Ollama (local inference)
- Mistral, Neural-Chat, or alternative open models
- 4-bit quantization support (future phases)

Vector Database & Embeddings
- FAISS (vector similarity search)
- Ollama Embeddings or HuggingFace models

Framework & Orchestration
- LangChain (LLM integration)
- LangGraph (agentic workflows - Phase 3)
- Pydantic (data validation)

Document Processing
- PyPDF (PDF extraction)
- TextLoader (text file handling)
- RecursiveCharacterTextSplitter (document chunking)

Build & Testing
- Python 3.8+
- pytest for testing
- Docker for deployment
```

### Requirements

- Python 3.8 or higher
- Ollama (for local LLM inference)
- 8GB RAM minimum (16GB recommended)
- GPU support optional (for acceleration in Phases 3-5)

---

## Installation and Setup

### Step 1: Install Ollama

Download and install from https://ollama.ai

### Step 2: Clone Repository

```bash
git clone https://github.com/yourusername/agentic-multimodal-rag.git
cd agentic-multimodal-rag
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Language Model

```bash
ollama pull mistral
```

Alternative models:
```bash
ollama pull neural-chat    # Better for document understanding
ollama pull orca-mini      # Smaller, faster
ollama pull llama2         # Alternative option
```

### Step 5: Verify Installation

```bash
# Start Ollama service
ollama serve

# In another terminal, test setup
python3 -c "from langchain_ollama import ChatOllama; llm = ChatOllama(model='mistral'); print(llm.invoke('Hello'))"
```

---

## Usage

### Prepare Documents

```bash
mkdir documents
# Add PDF or TXT files to documents/ folder
```

### Run RAG System

Choose your preferred RAG implementation:

```bash
# CRAG (Corrective RAG)
python3 rag_system.py

# Self-RAG (Self-Reflective)
python3 rag_selfrag.py
```

### Programmatic Usage

```python
from rag_system import DocumentProcessor, build_crag_graph
from langchain_ollama import ChatOllama

# Setup
processor = DocumentProcessor()
documents = processor.load_documents("./documents")
processor.create_index(documents)

# Initialize LLM
llm = ChatOllama(model="mistral", temperature=0, num_ctx=4096)

# Build system
app = build_crag_graph(processor.retriever, llm, llm_with_tools)

# Execute query
initial_state = {
    "question": "What are the main topics covered?",
    "documents": [],
    "generation": "",
    "web_search_needed": False,
    "retrieve_attempts": 0,
    "rewrite_attempts": 0
}

result = app.invoke(initial_state)
print("Answer:", result["generation"])
```

---

## Research Foundation

This work builds upon peer-reviewed research in retrieval-augmented generation and multimodal learning:

### Primary References

1. **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**
   - Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023)
   - arXiv: 2310.11511
   - Introduces reflection tokens for quality assessment
   - Demonstrates superior performance with smaller models

2. **Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity**
   - Jeong, S., Baek, J., Cho, S., Hwang, S.J., & Park, J.C. (2024)
   - arXiv: 2403.14403
   - Complexity-based routing for efficiency optimization
   - Balances accuracy and latency across query types

3. **Corrective Retrieval Augmented Generation**
   - Shi, W., et al. (2024)
   - arXiv: 2401.15884
   - Error correction mechanisms for retrieval failures
   - Web search integration for knowledge gaps

### Related Work

- **Dense Passage Retrieval for Open-Domain Question Answering** (Karpukhin et al., 2020)
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020)
- **Retrieval Augmented Language Model Pre-Training** (Guu et al., 2020)
- **Vision Transformers for Document Understanding** (Li et al., 2021)
- **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023)

---

## Documentation

Additional documentation files provide detailed guidance:

- **PAPERS_ANALYSIS.md**: Comprehensive comparison of Self-RAG and Adaptive-RAG approaches
- **CRAG_vs_SelfRAG.md**: Decision framework for choosing between approaches
- **HYBRID_IMPLEMENTATION_GUIDE.md**: Advanced configuration and optimization
- **PAPERS_GUIDE.md**: Instructions for obtaining research papers
- **SETUP.md**: Detailed installation and configuration guide

---

## Future Development Areas

### Multimodal Vision-Language Models (Phase 2)

The next phase will integrate Vision-Language Models for document understanding:

- **Image-based content analysis**: Tables, diagrams, charts, signatures
- **OCR capabilities**: Scanned document processing
- **Spatial reasoning**: Layout-aware information extraction
- **4-bit quantization**: Efficient on-device VLM deployment
- **Fine-tuning**: QLoRA-based adaptation to domain-specific documents

### Agentic Reasoning (Phase 3)

Implementing autonomous agents for complex query resolution:

- **Multi-step reasoning**: Decomposition and verification chains
- **LangGraph orchestration**: Workflow management and persistence
- **Tool use framework**: Specialized processors for different content types
- **Autonomous verification**: Self-checking and confidence assessment
- **Evidence aggregation**: Cross-document reasoning and synthesis

---

## Technical Notes for Phase 2 Transition

Preparation for multimodal integration:

1. **Model Architecture**: Design for multimodal input handling
2. **Data Pipeline**: Support for image and document processing
3. **Embedding Space**: Joint text-image embeddings
4. **Fine-tuning Framework**: QLoRA integration for VLM adaptation
5. **Evaluation Metrics**: Cross-modal performance assessment

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If using this work in research, please cite:

```bibtex
@software{agentic_rag_2024,
  title={Agentic Multimodal RAG Framework for Privacy-Preserving Document Intelligence},
  author={Your Name and Contributors},
  year={2024},
  url={https://github.com/yourusername/agentic-multimodal-rag}
}
```

And the foundational papers:

```bibtex
@article{asai2023selfrag,
  title={Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  author={Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2310.11511},
  year={2023}
}

@article{jeong2024adaptiverag,
  title={Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity},
  author={Jeong, Soyeong and Baek, Jinheon and Cho, Sukmin and Hwang, Sung Ju and Park, Jong C},
  journal={arXiv preprint arXiv:2403.14403},
  year={2024}
}
```

## Acknowledgments

This research builds on the excellent work of:
- University of Washington and Allen Institute for AI (Self-RAG)
- KAIST (Adaptive-RAG)
- Meta AI (CRAG research)
- LangChain and Ollama communities



This README will be updated as phases progress with specific achievements, documentation, and technical details.
