# Implementation Plan: Groq API Migration

## Overview

This plan outlines the step-by-step migration from OpenAI to Groq for chat completions while maintaining OpenAI embeddings. The migration follows a surgical approach with minimal code changes, focusing on configuration and the RAG chain module. Each task builds incrementally with validation checkpoints to ensure the system remains functional throughout the migration.

## Tasks

- [x] 1. Update project dependencies
  - Add langchain-groq to requirements.txt
  - Verify all existing dependencies are preserved
  - _Requirements: 3.1, 3.2_

- [ ]* 1.1 Write unit test to verify dependency presence
  - Test that requirements.txt contains langchain-groq
  - Test that requirements.txt contains langchain-openai
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 2. Update configuration module for dual-provider support
  - [x] 2.1 Add Groq configuration variables to config.py
    - Add GROQ_API_KEY environment variable loading
    - Add GROQ_MODEL environment variable with default "llama-3.3-70b-versatile"
    - Update section comments to distinguish OpenAI (embeddings) and Groq (chat) settings
    - _Requirements: 1.3, 1.4, 2.4_

  - [x] 2.2 Update API key validation function
    - Rename validate_api_key() to validate_api_keys()
    - Check for both OPENAI_API_KEY and GROQ_API_KEY
    - Provide descriptive error messages indicating which key is missing
    - Update log message to mention both providers
    - _Requirements: 1.5, 2.3, 6.1, 6.2_

  - [ ]* 2.3 Write unit tests for configuration
    - Test GROQ_API_KEY is loaded from environment
    - Test GROQ_MODEL has correct default value
    - Test validate_api_keys() raises error when Groq key missing
    - Test validate_api_keys() raises error when OpenAI key missing
    - Test validate_api_keys() succeeds when both keys present
    - _Requirements: 1.3, 1.4, 1.5, 2.3_

- [x] 3. Migrate RAG chain to use Groq
  - [x] 3.1 Update rag_chain.py imports and model initialization
    - Replace ChatOpenAI import with ChatGroq from langchain_groq
    - Update import to use GROQ_MODEL instead of CHAT_MODEL from config
    - Replace ChatOpenAI instantiation with ChatGroq in build_rag_chain()
    - Update log messages to reference Groq and the specific model
    - _Requirements: 1.1, 1.2, 7.1_

  - [ ]* 3.2 Write unit test for RAG chain model type
    - Test that build_rag_chain() creates a chain using ChatGroq
    - Test that the model name matches GROQ_MODEL configuration
    - _Requirements: 1.1, 1.2_

  - [ ]* 3.3 Write property test for response format consistency
    - **Property 3: Response format consistency**
    - **Validates: Requirements 5.2, 5.5, 7.4**

- [x] 4. Verify ingestion pipeline unchanged
  - [x] 4.1 Confirm ingestion.py still uses OpenAI embeddings
    - Review _get_embeddings() function
    - Verify it returns OpenAIEmbeddings with EMBEDDING_MODEL
    - No code changes should be needed
    - _Requirements: 2.1, 2.4_

  - [ ]* 4.2 Write property test for ingestion functionality
    - **Property 4: Ingestion creates valid embeddings**
    - **Validates: Requirements 5.3**

- [x] 5. Update configuration documentation
  - [x] 5.1 Update .env.example file
    - Add GROQ_API_KEY with example format (gsk_...)
    - Add GROQ_MODEL with default value
    - Update comments to clarify OpenAI is for embeddings
    - Add comment clarifying Groq is for chat completions
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 5.2 Update README.md documentation
    - Add section explaining dual-provider architecture
    - Add instructions for obtaining Groq API key from https://console.groq.com
    - Update setup instructions to include both API keys
    - List available Groq models (llama-3.3-70b-versatile, mixtral-8x7b-32768)
    - Update any architecture diagrams or descriptions
    - _Requirements: 4.4, 4.5_

- [x] 6. Checkpoint - Verify configuration and basic functionality
  - Install updated dependencies in a clean environment
  - Set both API keys in .env file
  - Verify config.py loads both keys successfully
  - Verify validate_api_keys() passes with both keys
  - Ensure all tests pass, ask the user if questions arise

- [ ] 7. Integration testing across all components
  - [ ]* 7.1 Write property test for end-to-end query functionality
    - **Property 2: Queries return answers with citations**
    - **Validates: Requirements 5.1**

  - [ ]* 7.2 Write integration test for API server
    - Test that server starts successfully with Groq configuration
    - Test that /contract-assistant/invoke endpoint returns correct format
    - _Requirements: 5.2_

  - [ ]* 7.3 Write integration test for Gradio UI
    - Test that UI initializes without errors
    - Test that query submission returns formatted response
    - _Requirements: 5.1_

  - [ ]* 7.4 Write integration test for evaluation module
    - Test that evaluation.py runs successfully with Groq
    - Verify evaluation results are returned in expected format
    - _Requirements: 5.4_

- [x] 8. Update validation function calls
  - [x] 8.1 Update server.py to call validate_api_keys()
    - Change validate_api_key() to validate_api_keys() in create_app()
    - _Requirements: 2.3_

  - [x] 8.2 Update app.py to call validate_api_keys()
    - Change validate_api_key() to validate_api_keys() in launch()
    - _Requirements: 2.3_

- [x] 9. Final checkpoint - Complete system verification
  - Run all unit tests and property tests
  - Start API server and verify it responds to queries
  - Start Gradio UI and submit test queries
  - Run evaluation.py and compare results with pre-migration baseline
  - Verify logs show Groq for chat and OpenAI for embeddings
  - Ensure all tests pass, ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster migration
- The migration is designed to be low-risk with minimal code changes
- All changes are isolated to config.py and rag_chain.py
- Existing vector store remains compatible (no re-indexing needed)
- Property tests ensure behavior consistency across the migration
- Manual testing should verify response quality hasn't degraded
