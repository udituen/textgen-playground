from pydantic import BaseModel, Field
from typing import Optional, Literal

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000, description="Input prompt")
    model_name: str = Field(..., description="Model name (e.g., 'gpt2', 'Qwen/Qwen-7B-Chat')")
    strategy: Literal["greedy", "top_k", "top_p", "beam"] = Field(
        "greedy",
        description="Decoding strategy"
    )
    temperature: float = Field(
        1.0,
        ge=0.1,
        le=2.0,
        description="Sampling temperature"
    )
    max_new_tokens: int = Field(
        100,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate"
    )
    top_k: Optional[int] = Field(
        50,
        ge=1,
        le=100,
        description="Top-k sampling parameter (used when strategy='top_k')"
    )
    top_p: Optional[float] = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (used when strategy='top_p')"
    )
    num_beams: Optional[int] = Field(
        4,
        ge=1,
        le=10,
        description="Number of beams (used when strategy='beam')"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Once upon a time in a distant land",
                "model_name": "gpt2",
                "strategy": "top_p",
                "temperature": 0.8,
                "max_new_tokens": 150,
                "top_p": 0.9
            }
        }


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    model_name: str
    strategy: str
    temperature: float
    tokens_generated: Optional[int] = None


class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    is_loaded: bool
    device: str

