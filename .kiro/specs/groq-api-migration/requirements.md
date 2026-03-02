# Requirements Document

## Introduction

This document specifies the requirements for migrating the Smart Contract Assistant from using OpenAI's API for chat completions to using Groq's API, while maintaining OpenAI embeddings and all existing functionality. The migration aims to leverage Groq's fast LLM inference capabilities while preserving the system's RAG architecture and user experience.

## Glossary

- **RAG_System**: The Smart Contract Assistant application that uses retrieval-augmented generation to answer questions about smart contracts
- **Chat_Provider**: The LLM service used for generating conversational responses (currently OpenAI, migrating to Groq)
- **Embedding_Provider**: The service used for generating vector embeddings (remains OpenAI)
- **Configuration_Module**: The config.py file that manages environment variables and system settings
- **RAG_Chain**: The rag_chain.py module that orchestrates retrieval and generation
- **API_Server**: The FastAPI server (server.py) that exposes REST endpoints
- **Gradio_UI**: The web interface (app.py) for user interactions
- **Ingestion_Pipeline**: The ingestion.py module that processes and indexes documents
- **Evaluation_Module**: The evaluation.py module that tests system performance

## Requirements

### Requirement 1: Groq API Integration

**User Story:** As a system administrator, I want to integrate Groq API for chat completions, so that the system can leverage Groq's fast inference capabilities.

#### Acceptance Criteria

1. WHEN the RAG_Chain initializes the chat model, THE RAG_System SHALL use the Groq API client instead of OpenAI's chat API
2. WHEN the RAG_Chain generates responses, THE RAG_System SHALL use Groq's LLM models (llama-3.3-70b-versatile or mixtral-8x7b-32768)
3. THE Configuration_Module SHALL support a GROQ_API_KEY environment variable
4. THE Configuration_Module SHALL support a GROQ_MODEL environment variable with a default value
5. WHEN the Groq API key is missing, THE RAG_System SHALL raise a descriptive validation error

### Requirement 2: Preserve OpenAI Embeddings

**User Story:** As a system administrator, I want to continue using OpenAI embeddings, so that the existing vector store remains compatible without re-indexing.

#### Acceptance Criteria

1. THE Ingestion_Pipeline SHALL continue using OpenAI's text-embedding-3-small model for document embeddings
2. THE RAG_Chain SHALL continue using OpenAI embeddings for query vectorization
3. WHEN the system initializes, THE RAG_System SHALL require both OPENAI_API_KEY and GROQ_API_KEY
4. THE Configuration_Module SHALL maintain separate configuration for embedding and chat models

### Requirement 3: Dependency Management

**User Story:** As a developer, I want updated dependencies, so that the system includes the necessary Groq integration packages.

#### Acceptance Criteria

1. THE requirements.txt file SHALL include the langchain-groq package
2. THE requirements.txt file SHALL maintain all existing dependencies
3. WHEN dependencies are installed, THE RAG_System SHALL have access to both langchain-openai and langchain-groq packages

### Requirement 4: Configuration Updates

**User Story:** As a system administrator, I want clear configuration documentation, so that I can properly set up API keys and model selections.

#### Acceptance Criteria

1. THE .env.example file SHALL include GROQ_API_KEY with example format (gsk_...)
2. THE .env.example file SHALL include GROQ_MODEL with a recommended default value
3. THE .env.example file SHALL maintain OPENAI_API_KEY for embeddings
4. THE README.md file SHALL document the dual-provider architecture
5. THE README.md file SHALL include instructions for obtaining a Groq API key from https://console.groq.com

### Requirement 5: Functional Preservation

**User Story:** As a user, I want all existing features to work after migration, so that my workflow is not disrupted.

#### Acceptance Criteria

1. WHEN a user submits a query through the Gradio_UI, THE RAG_System SHALL return relevant answers with source citations
2. WHEN a client calls the API_Server endpoints, THE RAG_System SHALL process requests and return responses in the same format as before
3. WHEN the Ingestion_Pipeline processes documents, THE RAG_System SHALL create and store embeddings successfully
4. WHEN the Evaluation_Module runs tests, THE RAG_System SHALL execute evaluation queries and return results
5. THE RAG_System SHALL maintain the same response format including answer text and source citations

### Requirement 6: Error Handling

**User Story:** As a developer, I want clear error messages for API issues, so that I can quickly diagnose and fix configuration problems.

#### Acceptance Criteria

1. WHEN the Groq API key is invalid or missing, THE RAG_System SHALL raise a descriptive error message indicating the Groq configuration issue
2. WHEN the OpenAI API key is invalid or missing, THE RAG_System SHALL raise a descriptive error message indicating the OpenAI configuration issue
3. WHEN Groq API rate limits are exceeded, THE RAG_System SHALL propagate the error with context about which provider failed
4. WHEN network errors occur with Groq, THE RAG_System SHALL provide clear error messages distinguishing Groq failures from OpenAI failures

### Requirement 7: Backward Compatibility

**User Story:** As a developer, I want minimal code changes, so that the migration is low-risk and easy to review.

#### Acceptance Criteria

1. THE RAG_Chain interface SHALL maintain the same function signatures (build_rag_chain, query)
2. THE Configuration_Module SHALL maintain all existing configuration variables
3. WHEN other modules import from RAG_Chain, THE RAG_System SHALL not require changes to those import statements
4. THE RAG_System SHALL maintain the same data structures for responses (Dict with 'answer', 'sources', 'formatted')
