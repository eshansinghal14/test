import argparse
import concurrent.futures
import json

import torch
from datasets import load_dataset
from utils import load_model



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--num_sequences", type=int, required=True, help="Number of problems to generate traces for")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of problems to process in parallel")
    parser.add_argument("--output", default="gsm8k_traces.json", help="Output JSON file path")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


_SYSTEM = (
    "Solve the math problem step by step. "
    "At the end of your solution, write the final numeric answer on its own line in the format: #### <number>"
)

def build_prompt(question, tokenizer):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {_SYSTEM}\n\nQuestion: {question}\nAnswer:"


def _generate_on_device(model, device, tokenizer, sub_batch, max_new_tokens, logits_processor, generate_kwargs):
    from transformers import LogitsProcessorList
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.set_device(device)

    # Reset per-sequence queue state before each generation call.
    if logits_processor is not None and hasattr(logits_processor, "queues"):
        logits_processor.queues = None
    processor_list = LogitsProcessorList([logits_processor]) if logits_processor is not None else None

    prompts = [build_prompt(ex["question"], tokenizer) for ex in sub_batch]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    gen_kwargs.update(generate_kwargs)
    if processor_list is not None:
        gen_kwargs["logits_processor"] = processor_list

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    results = []
    for i, ex in enumerate(sub_batch):
        results.append({
            "question": ex["question"],
            "answer": ex["answer"],
            "generated_response": tokenizer.decode(out_ids[i, prompt_len:], skip_special_tokens=True),
        })
    del out_ids
    torch.cuda.empty_cache()
    return results


def generate_traces(models, tokenizer, examples, max_new_tokens, batch_size=8,
                    logits_processor=None, spec_decoding=False, **generate_kwargs):
    """
    Generate responses for a list of GSM8K examples across multiple GPUs.

    models: list of model replicas, one per GPU. Each batch is split evenly across
    GPUs and sub-batches run in parallel threads.
    Returns a list of dicts with keys: question, answer, generated_response.
    Pass a LogitsProcessor (or list of one per model) via logits_processor to
    intercept/augment logits at each step. A list ensures each GPU thread uses
    its own processor with the correct model's hook.
    Any extra keyword arguments are forwarded to model.generate().
    """
    num_gpus = len(models)
    devices = [next(m.parameters()).device for m in models]
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    traces = []

    # Support per-GPU processor list so each thread uses the hook bound to its model.
    if isinstance(logits_processor, list):
        processors = logits_processor
    else:
        processors = [logits_processor] * num_gpus

    try:
        for batch_start in range(0, len(examples), batch_size):
            batch = examples[batch_start : batch_start + batch_size]

            sub_size = max(1, (len(batch) + num_gpus - 1) // num_gpus)
            sub_batches = [batch[i : i + sub_size] for i in range(0, len(batch), sub_size)]

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = [
                    executor.submit(
                        _generate_on_device,
                        models[idx], devices[idx], tokenizer, sub_batches[idx],
                        max_new_tokens, processors[idx], generate_kwargs,
                    )
                    for idx in range(len(sub_batches))
                ]
                for future in futures:  # iterate in submission order to preserve batch ordering
                    traces.extend(future.result())

            print(f"  [{len(traces)}/{len(examples)}] generated", end="\r", flush=True)
    finally:
        tokenizer.padding_side = orig_side

    return traces


def load_models_on_all_gpus(model_name):
    """Load one model replica per available GPU, sharing a single tokenizer."""
    num_gpus = torch.cuda.device_count()
    models = []
    tokenizer = None
    if num_gpus > 0:
        for gpu_id in range(num_gpus):
            model, tok = load_model(model_name, device_map=f"cuda:{gpu_id}")
            model.eval()
            models.append(model)
            if tokenizer is None:
                tokenizer = tok
    else:
        model, tokenizer = load_model(model_name)
        model.eval()
        models = [model]
    print(f"Loaded {len(models)} model replica(s) across {len(models)} device(s)")
    return models, tokenizer


def main():
    args = parse_args()

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    examples = list(dataset.select(range(args.num_sequences)))

    models, tokenizer = load_models_on_all_gpus(args.model)

    results = []
    for i, trace in enumerate(generate_traces(
        models, tokenizer, examples, args.max_new_tokens,
        batch_size=args.batch_size, do_sample=True, temperature=0.8,
    )):
        trace["problem_index"] = i
        results.append(trace)

        if (i + 1) % args.batch_size == 0 or (i + 1) == len(examples):
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[{i + 1}/{len(examples)}] saved to {args.output}")

    print(f"Done. {len(results)} traces saved to {args.output}")


if __name__ == "__main__":
    main()
