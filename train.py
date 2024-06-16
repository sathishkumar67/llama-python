from __future__ import annotations
import torch
from transformers import LlamaTokenizer

from architecture import ModelArgs, Transformer
from dataset import DataArgs, TokenDataset

# function to convert dict to json
def dict_to_json(dictionary):
    import json
    return json.dumps(dictionary)