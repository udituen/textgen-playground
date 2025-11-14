import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(model_name):
    """
    Input (str) model name
    initiate tokenizer and model based on model name
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    return model



def generate(tokenizer, model, prompt, strategy, temperature, max_new_tokens):
    """
    Input above parameters and outputs the result of the model
    """

    gen_kwargs = {
        "max_new_tokens" : max_new_tokens,
        "temperature":temperature
    }
    if strategy == "greedy":
        gen_kwargs.update({
            "do_sample":False
        })
    elif strategy == "top_k":
        gen_kwargs.update({
            "do_sample":True,
            "top_k" : 50
        })
    elif strategy == "top_p":
        gen_kwargs.update({
            "do_sample":True,
            "top_p": 0.9
        })
    elif strategy == "beam":
        gen_kwargs.update({
            "num_beams":4,
            "num_return_sequences":4,
        })
    
    if model == "Qwen":

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        print(f"kwargs : {gen_kwargs}")

        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, **gen_kwargs)
        output = tokenizer.decode(generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        return output
    
    elif model == "gpt":
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, **gen_kwargs)
        print(f"kwargs : {gen_kwargs}")
  
        return tokenizer.decode(output[0])
