# app/models/generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any
import asyncio
from functools import lru_cache
import accelerate


class TextGenerator:
    """
    Text generation model wrapper supporting multiple models and decoding strategies
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model_type = self._detect_model_type(model_name)
        self.is_loaded = False
        self._model = None
        self._tokenizer = None
        
    def _detect_model_type(self, model_name: str) -> str:
        """Detect model type from model name"""
        model_name_lower = model_name.lower()
        if "qwen" in model_name_lower:
            return "qwen"
        elif "gpt" in model_name_lower:
            return "gpt"
        else:
            return "causal_lm"
    
    async def load_model(self):
        """
        Load the model and tokenizer asynchronously
        """
        if self.is_loaded:
            return
        
        # Run the blocking I/O in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)
        
    def _load_model_sync(self):
        """Synchronous model loading"""
        print(f"Loading model: {self.model_name}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=self.device,
            trust_remote_code=True
        )
        
        # Set pad token if not set (common issue with GPT models)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self.is_loaded = True
        print(f"Model {self.model_name} loaded successfully")
    
    def _build_generation_kwargs(
        self,
        strategy: str,
        temperature: float,
        max_new_tokens: int,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build generation kwargs based on decoding strategy
        """
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        
        if strategy == "greedy":
            gen_kwargs.update({
                "do_sample": False
            })
        
        elif strategy == "top_k":
            gen_kwargs.update({
                "do_sample": True,
                "top_k": top_k or 50
            })
        
        elif strategy == "top_p":
            gen_kwargs.update({
                "do_sample": True,
                "top_p": top_p or 0.9
            })
        
        elif strategy == "beam":
            gen_kwargs.update({
                "num_beams": num_beams or 4,
                "num_return_sequences": 1,  # Changed to 1 for API simplicity
                "early_stopping": True,
            })
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return gen_kwargs
    
    async def generate(
        self,
        prompt: str,
        strategy: str = "greedy",
        temperature: float = 1.0,
        max_new_tokens: int = 100,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_beams: Optional[int] = None,
    ) -> str:
        """
        Generate text from prompt
        """
        if not self.is_loaded:
            await self.load_model()
        
        gen_kwargs = self._build_generation_kwargs(
            strategy=strategy,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
        )
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            self._generate_sync,
            prompt,
            gen_kwargs
        )
        
        return output
    
    def _generate_sync(self, prompt: str, gen_kwargs: Dict[str, Any]) -> str:
        """Synchronous generation"""
        
        if self.model_type == "qwen":
            return self._generate_qwen(prompt, gen_kwargs)
        elif self.model_type in ["gpt", "causal_lm"]:
            return self._generate_gpt(prompt, gen_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _generate_qwen(self, prompt: str, gen_kwargs: Dict[str, Any]) -> str:
        """Generate text using Qwen chat template"""
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)
        
        # Decode only the new tokens
        output = self._tokenizer.decode(
            generated_ids[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        return output
    
    def _generate_gpt(self, prompt: str, gen_kwargs: Dict[str, Any]) -> str:
        """Generate text using standard causal LM approach"""
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)
        
        # Decode the full output
        output = self._tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        return output
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "is_loaded": self.is_loaded,
            "device": str(self._model.device) if self.is_loaded else self.device,
        }


# Model cache for managing multiple models
class ModelCache:
    """
    Cache multiple models to avoid reloading
    """
    def __init__(self, max_models: int = 3):
        self.max_models = max_models
        self._cache: Dict[str, TextGenerator] = {}
    
    async def get_or_load(self, model_name: str, device: str = "auto") -> TextGenerator:
        """
        Get model from cache or load it
        """
        if model_name not in self._cache:
            # If cache is full, remove oldest model
            if len(self._cache) >= self.max_models:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                print(f"Removed {oldest_key} from cache")
            
            # Load new model
            generator = TextGenerator(model_name=model_name, device=device)
            await generator.load_model()
            self._cache[model_name] = generator
        
        return self._cache[model_name]
    
    def clear(self):
        """Clear all cached models"""
        self._cache.clear()
        torch.cuda.empty_cache()  # Free GPU memory


# Global model cache instance
model_cache = ModelCache(max_models=2)