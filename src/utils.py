from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import HF_READ_TOKEN

def load_model(model_name):
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    global logged_in
    if not logged_in and HF_READ_TOKEN:
        login(HF_READ_TOKEN)
        logged_in = True

    device = get_default_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device.type == "cuda" else None,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Match neuron_distillation AddDataset/collate_fn (right-pad); eval/generate use same side.
    tokenizer.padding_side = "right"
    return model, tokenizer