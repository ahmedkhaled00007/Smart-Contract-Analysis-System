"""
LangServe Microservice — FastAPI + LangServe API Server.
Serves the RAG chain as a REST API with auto-generated docs and playground.

Run with:  python -m smart_contract_assistant.server
       or: python main.py serve
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from config import (
    validate_api_keys,
    API_HOST,
    API_PORT,
    logger,
)
from ingestion import load_or_create_empty_index
from rag_chain import build_rag_chain


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    validate_api_keys()

    # Load vector store
    vectorstore = load_or_create_empty_index()

    # Build RAG chain
    chain = build_rag_chain(vectorstore)

    # Create FastAPI app
    app = FastAPI(
        title="Smart Contract Analysis API",
        description="Smart Contract Analysis API",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add LangServe routes
    try:
        add_routes(app, chain, path="/contract-assistant")
        logger.info("LangServe routes added at /contract-assistant")
    except Exception as e:
        logger.warning(f"Could not add LangServe routes: {e}")
        logger.info("The chain is still accessible programmatically.")

        # Add a simple fallback endpoint
        from pydantic import BaseModel

        class QueryRequest(BaseModel):
            input: str

        @app.post("/contract-assistant/invoke")
        async def invoke_chain(request: QueryRequest):
            result = chain.invoke({"input": request.input})
            return {"output": result}

    logger.info("FastAPI app created successfully.")
    return app


# Create the app instance (used by uvicorn)
app = create_app()


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server at http://{API_HOST}:{API_PORT}")
    logger.info(f"API docs:   http://{API_HOST}:{API_PORT}/docs")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
