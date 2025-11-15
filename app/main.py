# app/main.py
"""
FastAPI application entry point for Text Generator API
This is the file that uvicorn runs: uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import your routes
from app.api.routes import router

# # Import configuration
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    Replaces deprecated @app.on_event("startup") and @app.on_event("shutdown")
    """
    # Startup
    print("🚀 Starting up Text Generator API...")
    # print(f"📦 API Version: {settings.VERSION}")
    # print(f"🔧 Environment: {settings.ENVIRONMENT}")
    
    # Optional: Pre-load a default model
    # from app.models.generator import model_cache
    # await model_cache.get_or_load(settings.DEFAULT_MODEL_NAME)
    # print(f"✅ Default model loaded: {settings.DEFAULT_MODEL_NAME}")
    
    yield  # Application runs here
    
    # Shutdown
    print("🛑 Shutting down Text Generator API...")
    # Cleanup if needed
    from app.models.generator import model_cache
    model_cache.clear()
    print("✅ Cleanup complete")


def create_application() -> FastAPI:
    """
    Application factory pattern
    Creates and configures the FastAPI application
    """
    application = FastAPI(
        title=settings.PROJECT_NAME,
        description="Text Generation API using Language Models with multiple decoding strategies",
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware - allows requests from Streamlit and other origins
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes with prefix
    application.include_router(
        router,
        prefix=settings.API_V1_PREFIX,
        tags=["text-generation"]
    )

    return application


# Create the FastAPI app instance
# This is what uvicorn looks for when you run: uvicorn app.main:app
app = create_application()


# Root endpoint
@app.get("/", tags=["root"])
def read_root():
    """
    Root endpoint - API information
    """
    return {
        "message": "Text Generator API",
        "version": settings.VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "api_base": settings.API_V1_PREFIX
    }


# Health check endpoint
@app.get("/health", tags=["health"])
def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }


# Optional: Add a ping endpoint
@app.get("/ping", tags=["health"])
def ping():
    """
    Simple ping endpoint
    """
    return {"ping": "pong"}


if __name__ == "__main__":
    # This allows you to run the file directly with: python app/main.py
    # Though normally you'd use: uvicorn app.main:app --reload
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )