import argparse
import re

import torch
from datasets import load_dataset
from generate_gsm8k_traces import generate_traces, load_models_on_all_gpus


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--spec-decoding", action="store_true",
                        help="Use speculative decoding with SC lm_head; requires --embedding-ckpt and --lm-head-ckpt")
    parser.add_argument("--embedding-ckpt", default=None,
                        help="Path to superchunk_embeddings.pt (required for --spec-decoding)")
    parser.add_argument("--lm-head-ckpt", default=None,
                        help="Path to SC lm_head checkpoint step_N.pt (required for --spec-decoding)")
    return parser.parse_args()


def extract_answer(text):
    m = re.search(r"####\s*([\d,.-]+)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    for n in reversed(nums):
        try:
            return float(n.replace(",", ""))
        except ValueError:
            pass
    return None


def eval_gsm8k_accuracy(traces):
    n_correct = sum(
        1 for t in traces
        if (p := extract_answer(t["generated_response"])) is not None
        and (g := extract_answer(t["answer"])) is not None
        and abs(p - g) < 1e-3
    )
    return n_correct / len(traces), n_correct, len(traces)


def main():
    args = parse_args()

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    examples = list(dataset.select(range(min(args.num_samples, len(dataset)))))

    models, tokenizer = load_models_on_all_gpus(args.model)

    logits_processor = None
    hooks = []
    if args.spec_decoding:
        if args.embedding_ckpt is None or args.lm_head_ckpt is None:
            raise ValueError("--spec-decoding requires --embedding-ckpt and --lm-head-ckpt")
        # Local import to avoid circular dependency (train_superchunk_lm_head imports this module).
        from train_superchunk_lm_head import (
            _build_superchunk_lookups, make_sc_logits_processors, full_accept_rates_by_len,
        )
        emb_ckpt = torch.load(args.embedding_ckpt, map_location="cpu")
        superchunks = emb_ckpt["superchunks"]
        vocab_size = models[0].lm_head.weight.shape[0]
        lm_ckpt = torch.load(args.lm_head_ckpt, map_location="cpu")
        new_weight = torch.nn.Parameter(lm_ckpt["new_weight"])
        new_bias = torch.nn.Parameter(lm_ckpt["new_bias"])
        _, _, sc_idx_to_leaves, _ = _build_superchunk_lookups(superchunks, vocab_size)
        logits_processor, hooks = make_sc_logits_processors(
            models, new_weight, new_bias, sc_idx_to_leaves, vocab_size,
            spec_decoding=True,
        )

    try:
        traces = generate_traces(
            models, tokenizer, examples,
            args.max_new_tokens, batch_size=args.batch_size,
            logits_processor=logits_processor,
            spec_decoding=args.spec_decoding,
        )
    finally:
        for h in hooks:
            h.remove()
    print()

    acc, correct, total = eval_gsm8k_accuracy(traces)
    total_base_tokens = sum(len(tokenizer.encode(t["generated_response"], add_special_tokens=False)) for t in traces)
    queue_pops = sum(p.n_queue_pops for p in logits_processor) if logits_processor else 0
    avg_len = (total_base_tokens - queue_pops) / len(traces)
    print(f"Accuracy: {correct}/{total} = {acc:.2%}  avg_response_len={avg_len:.1f}")
    if args.spec_decoding:
        rates = full_accept_rates_by_len(logits_processor)
        print("full_accept_by_len: " + "  ".join(f"len{k}={a}/{t}" for k, (a, t) in rates.items()))


if __name__ == "__main__":
    main()
