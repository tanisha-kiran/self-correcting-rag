# CRAG vs Self-RAG: Detailed Comparison

## Overview

Both CRAG (Corrective RAG) and Self-RAG are approaches to build more intelligent retrieval-augmented generation systems, but they solve different problems and have different strengths.

## CRAG (Corrective RAG)

### Philosophy
**"Fix bad retrieval before generation"**

CRAG focuses on detecting and correcting poor document retrieval quality early in the pipeline, before the generation stage.

### Key Components

1. **Retrieval Evaluator**
   - Grades whether retrieved documents are relevant to the query
   - Returns a confidence score
   - Binary decision: relevant or not relevant

2. **Web Search Fallback**
   - If documents are deemed ambiguous or irrelevant
   - Supplements with web search results
   - Combines both sources for generation

3. **Knowledge Refinement** (optional)
   - Partitions documents into "knowledge strips"
   - Grades each strip individually
   - Filters out irrelevant sections

### Workflow

```
User Question
    ↓
Retrieve Documents
    ↓
Grade Documents for Relevance
    ├─ YES: Proceed to Generation
    └─ NO: Supplement with Web Search
    ↓
Generate Answer
    ↓
Return Answer
```

### When to Use CRAG

✅ **Use CRAG when:**
- You have some local documents but they might have gaps
- You want fast, confident answers
- You need web search as a safety net
- You care about retrieval quality early
- Your documents have variable relevance

❌ **Avoid CRAG when:**
- All documents are always relevant
- Web search is not available/not allowed
- You need more fine-grained control
- You want generation-level quality checks

### Advantages
- Fast and straightforward
- Good for mixed document quality
- Web search fallback is powerful
- Easy to debug (fewer steps)
- Lower LLM cost (fewer calls for grading)

### Disadvantages
- Doesn't check if generation is good
- Might generate hallucinations
- Less sophisticated feedback loops
- Web search dependency

## Self-RAG

### Philosophy
**"Self-correct through reflection at every stage"**

Self-RAG implements multiple reflection tokens that assess quality at document, generation, and usefulness levels, enabling fine-grained self-correction.

### Key Components

1. **ISREL Token** (Is Relevant)
   - Grades each document for relevance to query
   - Per-document evaluation
   - Filters out irrelevant documents

2. **ISSUP Token** (Is Supported)
   - Checks if generated answer is supported by documents
   - Guards against hallucinations
   - Validates claim-level support

3. **ISUSE Token** (Is Useful)
   - Scores answer usefulness (1-5)
   - Evaluates whether it actually answers the question
   - Can trigger re-retrieval if low score

4. **Iterative Refinement**
   - If answer isn't good, rewrite query
   - Retrieve new documents
   - Generate improved answer
   - Repeat until quality threshold met

### Workflow

```
User Question
    ↓
Retrieve Documents
    ↓
Grade Each Document (ISREL)
    ├─ Keep relevant documents
    └─ Filter irrelevant ones
    ↓
Generate Answer
    ↓
Check if Supported (ISSUP)
    ├─ YES: Continue
    └─ NO: Might rewrite
    ↓
Rate Usefulness (ISUSE)
    ├─ Score ≥ 4: Done
    └─ Score < 4: Rewrite Query & Retry
    ↓
Return Answer
```

### When to Use Self-RAG

✅ **Use Self-RAG when:**
- You need high-quality, verified answers
- Hallucinations are unacceptable
- You have the LLM budget for extra calls
- Your domain needs careful verification
- You want to understand why answers are generated
- You need progressive quality improvement

❌ **Avoid Self-RAG when:**
- Speed is critical
- LLM cost is a concern
- Your documents are already curated
- You don't need answer verification

### Advantages
- Sophisticated quality checks at multiple levels
- Guards against hallucinations
- Per-document relevance grading
- Iterative improvement loops
- Detailed visibility into decision-making
- Better for critical applications

### Disadvantages
- More expensive (multiple LLM calls)
- Slower due to multiple grading steps
- More complex to debug
- Can loop indefinitely if poorly configured
- Requires careful prompt engineering for graders

## Side-by-Side Comparison

| Aspect | CRAG | Self-RAG |
|--------|------|----------|
| **Primary Focus** | Retrieval quality | Overall quality (retrieval + generation) |
| **Key Strength** | Web search fallback | Hallucination prevention |
| **Grading Points** | Documents only | Documents + Generation (2 steps) |
| **Iteration Strategy** | Query rewrite before generation | Query rewrite after generation |
| **LLM Calls** | ~3-5 per query | ~6-10+ per query |
| **Speed** | Fast (2-3 seconds) | Slower (5-10 seconds) |
| **Cost** | Lower | Higher |
| **Complexity** | Simple | Complex |
| **Best For** | General QA with web fallback | Critical/specialized domains |
| **Hallucination Control** | None | Strong (ISSUP check) |
| **Usefulness Check** | No | Yes (ISUSE) |
| **Loop Prevention** | Rewrite attempts limit | Rewrite + usefulness score |

## Hybrid Approach

You can combine both strategies:

```python
# Start with CRAG's retrieval quality check
grade_documents()

if documents_relevant:
    # Then use Self-RAG's generation checks
    generate_answer()
    check_support()
    check_usefulness()
    if not useful:
        rewrite_and_retry()
else:
    # Fall back to CRAG's web search
    web_search()
```

## Decision Tree

```
Do you need to check if answers are true?
├─ YES → Self-RAG (ISSUP checks for hallucinations)
└─ NO → CRAG (faster, simpler)

Do you need real-time, fast answers?
├─ YES → CRAG
└─ NO → Self-RAG (quality over speed)

Is web search important?
├─ YES → CRAG
└─ NO → Either (Self-RAG is better for pure QA)

Are your documents always relevant?
├─ YES → Self-RAG (focus on generation quality)
└─ NO → CRAG (catch irrelevant docs early)
```

## Example Scenarios

### Scenario 1: FAQ System for Product Support
- **Chosen**: CRAG
- **Reason**: Documents are curated FAQs (always relevant), speed matters, web search not needed
- **Configuration**: Skip web search, focus on retrieval grading

### Scenario 2: Medical Information System
- **Chosen**: Self-RAG
- **Reason**: Hallucinations are dangerous, quality is critical, cost is acceptable
- **Configuration**: Strict usefulness thresholds (≥4), multiple rewrite attempts

### Scenario 3: Legal Document Analysis
- **Chosen**: Self-RAG with strict ISSUP
- **Reason**: Answers must be supported by documents, hallucinations unacceptable
- **Configuration**: Only accept "fully_supported", aggressive filtering

### Scenario 4: General Knowledge QA
- **Chosen**: CRAG
- **Reason**: Documents provide primary knowledge, web search for unknown topics
- **Configuration**: Enable web search, standard grading thresholds

## Performance Metrics to Track

### CRAG
- Retrieval precision (% of retrieved docs that are relevant)
- Web search fallback rate (when should it trigger)
- Answer correctness vs. without web search

### Self-RAG
- Document relevance filtering rate
- Hallucination detection rate (ISSUP accuracy)
- Usefulness score distribution
- Rewrite loop frequency

## Configuration Tips

### CRAG
```python
# Balance between strict and lenient
grade_threshold = 0.5  # What confidence means "relevant"?

# Web search only as fallback
web_search_only_if = "all_documents_irrelevant"

# Query rewrite attempts
max_rewrites = 1  # Usually 1-2 is enough
```

### Self-RAG
```python
# Useful score threshold
useful_threshold = 4  # 1-5 scale

# Support requirement
require_full_support = True  # vs partially_supported

# Maximum correction attempts
max_rewrites = 3  # Usually 2-3

# Document relevance requirement
min_relevant_docs = 1  # How many good docs do we need?
```

## Advanced: Extending Both

Both systems can be extended:

### Add Cost Tracking
```python
track_llm_calls = True
track_api_costs = True
```

### Add Caching
```python
cache_embeddings = True
cache_generations = True
```

### Add Custom Metrics
```python
def custom_grade_accuracy(generation, question, docs):
    # Your domain-specific grading
    pass
```

### Add Multi-turn Context
```python
class ConversationalRAGState(RAGState):
    chat_history: List[Message]
    conversational_context: str
```

## Conclusion

- **Choose CRAG** for fast, reliable systems where document quality varies
- **Choose Self-RAG** for high-stakes applications where quality and accuracy are paramount
- **Hybrid approach** for best of both worlds: fast retrieval filtering + generation verification

Start with CRAG, move to Self-RAG if quality issues emerge!