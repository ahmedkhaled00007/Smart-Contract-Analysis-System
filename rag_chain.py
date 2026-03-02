"""
RAG Pipeline with Citations.
Builds a retrieval-augmented generation chain that answers questions
using retrieved document chunks and cites the sources.

Compatible with langchain v0.3.x and v1.x+
"""

import os
from typing import Dict, Any, List
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

from config import (
    GROQ_MODEL,
    RETRIEVER_K,
    SYSTEM_PROMPT,
    logger,
)


def _format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore: FAISS):
    """
    Creates a RAG chain that returns the answer and source documents.
    Uses Groq for chat completions and OpenAI for embeddings.

    Args:
        vectorstore: FAISS vector store to retrieve from.

    Returns:
        A callable chain that accepts {"input": str} and returns
        {"input", "context", "answer"}.
    """
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    # Build chain using LCEL (LangChain Expression Language)
    # Step 1: Retrieve documents
    # Step 2: Format docs into context string
    # Step 3: Pass context + input to LLM
    # Step 4: Parse output

    def _chain_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["input"]

        # Retrieve relevant documents
        docs = retriever.invoke(question)

        # Format context
        context_str = _format_docs(docs)

        # Format the system prompt with context
        formatted_system_prompt = SYSTEM_PROMPT.format(context=context_str)

        # Create messages in the correct format for Groq
        messages = [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Generate answer
        response = llm.invoke(messages)
        answer = response.content

        return {
            "input": question,
            "context": docs,
            "answer": answer,
        }

    logger.info(f"RAG chain built (chat_model=Groq/{GROQ_MODEL}, k={RETRIEVER_K}).")
    return RunnableLambda(_chain_fn)


def query(chain, question: str) -> Dict[str, Any]:
    """
    Runs a question through the RAG chain and formats the result.

    Args:
        chain: The RAG chain to invoke.
        question: The user's question.

    Returns:
        Dict with 'answer', 'sources' (list), and 'formatted' (display string).
    """
    response = chain.invoke({"input": question})

    answer = response.get("answer", "No answer generated.")

    # Extract and deduplicate sources
    sources = []
    for doc in response.get("context", []):
        source_file = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        source_str = f"- {os.path.basename(source_file)} (Page {page})"
        sources.append(source_str)

    unique_sources = list(set(sources))

    formatted = f"{answer}\n\n**Sources:**\n" + "\n".join(unique_sources)

    return {
        "answer": answer,
        "sources": unique_sources,
        "formatted": formatted,
    }
