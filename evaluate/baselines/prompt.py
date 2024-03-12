from typing import Any
from torch import nn
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel 

class PromptedModel:
    def __init__(self, model : PreTrainedModel, tokenizer, prompt=""):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
    
    def forward(self, input_ids : torch.Tensor, attention_mask, **kwargs):
        lengths = attention_mask.sum(-1)
        inputs = []
        
        for i, x in enumerate(input_ids):
            inp = self.prompt + self.tokenizer.decode(x[:lengths[i]])
            inputs.append(inp)
        inputs_tokens = self.tokenizer(inputs, return_tensors='pt', padding=True).to(input_ids.device)
        return self.model(**inputs_tokens, **kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def generate(self, input_ids, attention_mask, **kwargs):
        inputs_ = []
        
        for x in input_ids:
            inp = self.prompt + self.tokenizer.decode(x).strip(self.tokenizer.pad_token)
            inputs_.append(inp)
        self.tokenizer.padding_side = 'left'
        inputs_tokens = self.tokenizer(inputs_, return_tensors='pt',padding=True).to(input_ids.device)
        self.tokenizer.padding_side = 'right'
        return self.model.generate(**inputs_tokens, **kwargs)
    
    def get_length(self, text : str) -> int:
        """Returns the length in tokens of the concatenation of the model's prompt and the text.

        Args:
            text (str): The text to add to the prompt

        Returns:
            int: The total number of tokens
        """
        text = self.prompt + text
        return len(self.tokenizer.encode(text))
    
        

class PromptHyperParams:
    def __str__(self) -> str:
        return '"No parameters"'
    
    @staticmethod
    def from_json(path):
        return PromptHyperParams()

def build_prompt(records) -> str:
    sfx = '. ' if len(records) else ''
    return ". ".join(x['prompt'].strip().format(x['subject']) + ' ' + x['target_new']['str'] for x in records) + sfx

def prompt_rewrite(
        model : nn.Module,
        tok,
        records,
        hparams,
        copy=False,
        return_orig_weights=True,
        **kwargs,
    ):
    prompt = build_prompt(records)
    if not isinstance(model, PromptedModel):
        model = PromptedModel(model, tok, prompt)
    else:
        model.prompt = prompt
    if return_orig_weights:
        return model, {"__prompt" : ""}
    return model
