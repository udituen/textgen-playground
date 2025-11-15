from fastapi import APIRouter, HTTPException, Depends
from app.schemas.request import GenerateRequest, GenerateResponse, ModelInfo
from app.models.generator import model_cache, TextGenerator

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text based on input prompt with specified model and parameters
    
    Supports multiple models and decoding strategies:
    - greedy: Deterministic, always picks highest probability token
    - top_k: Samples from top K most likely tokens
    - top_p: Nucleus sampling, samples from smallest set with cumulative probability >= p
    - beam: Beam search for more coherent but less diverse outputs
    """
    try:
        # Get or load model from cache
        generator = await model_cache.get_or_load(
            model_name=request.model_name,
            device="auto"
        )
        
        # Generate text
        generated_text = await generator.generate(
            prompt=request.prompt,
            strategy=request.strategy,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
            top_k=request.top_k,
            top_p=request.top_p,
            num_beams=request.num_beams,
        )
        
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            model_name=request.model_name,
            strategy=request.strategy,
            temperature=request.temperature,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@router.get("/models/info/{model_name:path}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """
    Get information about a specific model
    Use model_name with path (e.g., Qwen/Qwen-7B-Chat)
    """
    try:
        generator = await model_cache.get_or_load(model_name=model_name)
        info = generator.get_info()
        
        return ModelInfo(
            model_name=info["model_name"],
            model_type=info["model_type"],
            is_loaded=info["is_loaded"],
            device=info["device"],
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.post("/models/load/{model_name:path}")
async def load_model(model_name: str):
    """
    Pre-load a model into cache
    """
    try:
        generator = await model_cache.get_or_load(model_name=model_name)
        
        return {
            "status": "success",
            "message": f"Model {model_name} loaded successfully",
            "model_info": generator.get_info()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@router.post("/models/clear-cache")
async def clear_model_cache():
    """
    Clear all models from cache to free up memory
    """
    try:
        model_cache.clear()
        return {
            "status": "success",
            "message": "Model cache cleared"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get("/strategies")
async def list_strategies():
    """
    List available decoding strategies and their descriptions
    """
    return {
        "strategies": [
            {
                "name": "greedy",
                "description": "Deterministic decoding, always picks highest probability token",
                "parameters": ["temperature (ignored)"]
            },
            {
                "name": "top_k",
                "description": "Samples from top K most likely tokens",
                "parameters": ["temperature", "top_k"]
            },
            {
                "name": "top_p",
                "description": "Nucleus sampling, samples from smallest set with cumulative probability >= p",
                "parameters": ["temperature", "top_p"]
            },
            {
                "name": "beam",
                "description": "Beam search for more coherent outputs",
                "parameters": ["num_beams"]
            }
        ]
    }