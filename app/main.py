"""
This file contains the main app logic.
we will be using Autotenizer, automodelforcausalLM

"""

import torch
from utils.utils import get_model, generate


# parameters: model, temperature (0 to 2), strategy, 
# tokenizer is defined from the model name

