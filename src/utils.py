from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from constants import HF_READ_TOKEN

logged_in = False

def load_model(model_name, device_map="auto"):
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    global logged_in
    if not logged_in and HF_READ_TOKEN:
        login(HF_READ_TOKEN)
        logged_in = True

    device = get_default_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else None,
        device_map=device_map if device.type == "cuda" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Match neuron_distillation AddDataset/collate_fn (right-pad); eval/generate use same side.
    tokenizer.padding_side = "right"
    return model, tokenizer


def get_default_device() -> torch.device:
    """Get the default device, preferring CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")