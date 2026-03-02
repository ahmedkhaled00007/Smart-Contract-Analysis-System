# Design Document: Groq API Migration

## Overview

This design outlines the migration of the Smart Contract Assistant's chat completion functionality from OpenAI to Groq API. The migration maintains a dual-provider architecture where Groq handles chat completions (fast inference) and OpenAI handles embeddings (compatibility with existing vector store). This approach minimizes code changes while leveraging Groq's performance benefits.

The migration is surgical: only the chat model instantiation in `rag_chain.py` and configuration in `config.py` need modification. All other components (ingestion, server, UI, evaluation) remain unchanged due to LangChain's provider abstraction.

## Architecture

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG System                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  Ingestion   │      │  RAG Chain   │                    │
│  │              │      │              │                    │
│  │  OpenAI      │      │  OpenAI      │                    │
│  │  Embeddings  │──────│  Chat Model  │                    │
│  │              │      │              │                    │
│  └──────────────┘      └──────────────┘                    │
│         │                      │                            │
│         ▼                      ▼                            │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ FAISS Index  │      │  Responses   │                    │
│  └──────────────┘      └──────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG System                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  Ingestion   │      │  RAG Chain   │                    │
│  │              │      │              │                    │
│  │  OpenAI      │      │    Groq      │                    │
│  │  Embeddings  │──────│  Chat Model  │                    │
│  │              │      │              │                    │
│  └──────────────┘      └──────────────┘                    │
│         │                      │                            │
│         ▼                      ▼                            │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ FAISS Index  │      │  Responses   │                    │
│  └──────────────┘      └──────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

1. **Dual-Provider Strategy**: Use Groq for chat, OpenAI for embeddings
   - Rationale: Groq doesn't provide embeddings; re-indexing all documents would be expensive and unnecessary
   - Trade-off: Requires two API keys but maintains compatibility

2. **Minimal Code Changes**: Leverage LangChain's abstraction layer
   - Rationale: LangChain provides consistent interfaces across providers
   - Impact: Only `rag_chain.py` and `config.py` need modification

3. **Configuration-Driven**: Use environment variables for provider selection
   - Rationale: Allows easy switching between providers without code changes
   - Implementation: Add GROQ_API_KEY and GROQ_MODEL to configuration

## Components and Interfaces

### Modified Components

#### 1. Configuration Module (`config.py`)

**Changes Required**:
- Add `GROQ_API_KEY` environment variable loading
- Add `GROQ_MODEL` environment variable with default value (e.g., "llama-3.3-70b-versatile")
- Update `validate_api_key()` to check both OpenAI and Groq keys
- Rename existing function to `validate_api_keys()` for clarity

**New Configuration Variables**:
```python
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
```

**Updated Validation Function**:
```python
def validate_api_keys():
    """Validates that both OpenAI and Groq API keys are configured."""
    errors = []
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set (required for embeddings)")
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set (required for chat completions)")
    
    if errors:
        raise ValueError(
            "API key configuration errors:\n" + "\n".join(f"  - {e}" for e in errors) +
            "\n\nCreate a .env file from .env.example and add your keys."
        )
    logger.info("API keys configured: OpenAI (embeddings), Groq (chat)")
```

#### 2. RAG Chain Module (`rag_chain.py`)

**Changes Required**:
- Replace `from langchain_openai import ChatOpenAI` with `from langchain_groq import ChatGroq`
- Replace `ChatOpenAI(model=CHAT_MODEL, temperature=0)` with `ChatGroq(model=GROQ_MODEL, temperature=0)`
- Update import to use `GROQ_MODEL` from config instead of `CHAT_MODEL`
- Update log messages to reference Groq

**Modified Function**:
```python
def build_rag_chain(vectorstore: FAISS):
    """
    Creates a RAG chain that returns the answer and source documents.
    Uses Groq for chat completions and OpenAI for embeddings.
    """
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    
    # ... rest remains unchanged
    
    logger.info(f"RAG chain built (chat_model=Groq/{GROQ_MODEL}, k={RETRIEVER_K}).")
    return RunnableLambda(_chain_fn)
```

### Unchanged Components

The following components require NO code changes due to LangChain's abstraction:

1. **Ingestion Pipeline** (`ingestion.py`): Continues using OpenAI embeddings
2. **API Server** (`server.py`): Works with any LangChain-compatible chain
3. **Gradio UI** (`app.py`): Interacts through chain interface
4. **Evaluation Module** (`evaluation.py`): Tests chain behavior, not provider
5. **Main Entry Point** (`main.py`): No changes needed

### Interface Contracts

All existing interfaces remain unchanged:

```python
# RAG Chain Interface (unchanged)
def build_rag_chain(vectorstore: FAISS) -> Runnable
def query(chain, question: str) -> Dict[str, Any]

# Ingestion Interface (unchanged)
def ingest_documents(file_paths: List[str]) -> Optional[FAISS]
def save_index(vectorstore: FAISS, path: str = FAISS_INDEX_PATH) -> None
def load_index(path: str = FAISS_INDEX_PATH) -> Optional[FAISS]
```

## Data Models

No data model changes are required. The migration maintains all existing data structures:

### Response Format (Unchanged)
```python
{
    "answer": str,           # Generated answer text
    "sources": List[str],    # List of source citations
    "formatted": str         # Formatted answer with citations
}
```

### Chain Input Format (Unchanged)
```python
{
    "input": str  # User question
}
```

### Chain Output Format (Unchanged)
```python
{
    "input": str,              # Original question
    "context": List[Document], # Retrieved documents
    "answer": str              # Generated answer
}
```

## 
Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: RAG chain uses Groq for chat completions

*For any* RAG chain built by the system, the chat model should be a ChatGroq instance configured with a Groq model name (llama-3.3-70b-versatile, mixtral-8x7b-32768, etc.)

**Validates: Requirements 1.1, 1.2**

### Property 2: Queries return answers with citations

*For any* valid user query submitted through the Gradio UI or API, the response should contain both an answer string and a non-empty list of source citations

**Validates: Requirements 5.1**

### Property 3: Response format consistency

*For any* query processed by the RAG system, the response dictionary should contain exactly the keys 'answer', 'sources', and 'formatted', with 'answer' and 'formatted' as strings and 'sources' as a list

**Validates: Requirements 5.2, 5.5, 7.4**

### Property 4: Ingestion creates valid embeddings

*For any* list of valid document file paths, the ingestion pipeline should successfully create a FAISS vector store with a count of vectors greater than zero

**Validates: Requirements 5.3**

## Error Handling

### API Key Validation

The system must validate both API keys at startup:

1. **Missing Keys**: Raise `ValueError` with clear message indicating which key(s) are missing
2. **Key Format**: Log warnings if keys don't match expected format (sk-... for OpenAI, gsk_... for Groq)
3. **Validation Timing**: Perform validation in `validate_api_keys()` called by server.py and app.py before initializing chains

### Runtime Error Handling

1. **Groq API Errors**: Catch and re-raise with context: "Groq API error: {original_message}"
2. **OpenAI Embedding Errors**: Catch and re-raise with context: "OpenAI Embeddings error: {original_message}"
3. **Rate Limiting**: Propagate rate limit errors with provider information
4. **Network Failures**: Let LangChain's built-in retry logic handle transient failures

### Error Propagation Strategy

```python
# Example error handling in rag_chain.py
try:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
except Exception as e:
    logger.error(f"Failed to initialize Groq chat model: {e}")
    raise ValueError(f"Groq initialization error: {e}") from e
```

## Testing Strategy

### Dual Testing Approach

The migration requires both unit tests and property-based tests to ensure correctness:

- **Unit tests**: Verify specific configuration examples, edge cases (missing keys), and integration points
- **Property tests**: Verify universal properties across all inputs (response formats, query handling)

Both testing approaches are complementary and necessary for comprehensive coverage. Unit tests catch concrete configuration bugs, while property tests verify that the system behaves correctly across all possible inputs.

### Property-Based Testing Configuration

We will use **Hypothesis** (Python's property-based testing library) for implementing correctness properties:

- Each property test must run a minimum of 100 iterations
- Each test must include a comment tag: `# Feature: groq-api-migration, Property {N}: {property_text}`
- Each correctness property from this design must be implemented by a single property-based test
- Property tests should generate random inputs (queries, document lists) to verify universal behavior

### Unit Testing Focus

Unit tests should focus on:

1. **Configuration Examples**:
   - Verify GROQ_API_KEY is loaded from environment
   - Verify GROQ_MODEL has correct default value
   - Verify both API keys are validated

2. **Edge Cases**:
   - Missing Groq API key raises appropriate error
   - Missing OpenAI API key raises appropriate error
   - Invalid API key formats are handled

3. **Integration Points**:
   - RAG chain builds successfully with Groq
   - Ingestion still uses OpenAI embeddings
   - Server and UI can initialize with new configuration

### Test Organization

```python
# tests/test_config.py - Configuration unit tests
# tests/test_rag_chain.py - RAG chain unit tests
# tests/test_properties.py - Property-based tests using Hypothesis
```

### Manual Testing Checklist

After automated tests pass, perform manual verification:

1. Start Gradio UI and submit test queries
2. Start API server and call endpoints via /docs
3. Run evaluation.py to verify evaluation still works
4. Ingest new documents to verify embedding generation
5. Check logs for proper provider attribution (Groq vs OpenAI)

## Migration Risks and Mitigations

### Risk 1: Model Behavior Differences

**Risk**: Groq models may generate different response styles than GPT-4o-mini

**Mitigation**: 
- Run evaluation.py before and after migration to compare quality
- Use temperature=0 for deterministic responses
- Test with representative queries from production usage

### Risk 2: API Compatibility

**Risk**: Groq API may have different rate limits or request formats

**Mitigation**:
- Review Groq API documentation for rate limits
- Test with high query volume during development
- Implement proper error handling for rate limit errors

### Risk 3: Dependency Conflicts

**Risk**: langchain-groq may conflict with existing langchain packages

**Mitigation**:
- Test in clean virtual environment
- Pin compatible versions if conflicts arise
- Document any version constraints in requirements.txt

### Risk 4: Configuration Errors

**Risk**: Users may forget to add GROQ_API_KEY to .env file

**Mitigation**:
- Clear validation error messages
- Updated .env.example with prominent Groq configuration
- README documentation with setup instructions

## Documentation Updates

### README.md Changes

1. Update title/description to mention Groq integration
2. Add "Dual-Provider Architecture" section explaining OpenAI (embeddings) + Groq (chat)
3. Update setup instructions to include Groq API key acquisition
4. Add link to https://console.groq.com for API key creation
5. Update architecture diagram if present
6. Add note about model selection (available Groq models)

### .env.example Changes

```bash
# OpenAI API Key (Required for embeddings)
OPENAI_API_KEY=sk-proj-your-key-here

# Groq API Key (Required for chat completions)
GROQ_API_KEY=gsk_your-key-here

# Model Configuration (Optional - defaults shown)
EMBEDDING_MODEL=text-embedding-3-small
GROQ_MODEL=llama-3.3-70b-versatile
```

### Code Comments

Update comments in modified files to reflect dual-provider architecture:

- `config.py`: Update section headers to distinguish OpenAI and Groq settings
- `rag_chain.py`: Update docstrings to mention Groq for chat, OpenAI for embeddings
