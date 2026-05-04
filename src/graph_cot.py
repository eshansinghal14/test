import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset

from utils import load_model

model_name = 'meta-llama/Llama-3.2-1B-Instruct'
dataset = 'gsm8k'

ds = load_dataset(dataset, "main", split="test")


@dataclass
class Branch:
    input_ids: torch.Tensor
    log_prob: float
    generated_tokens: int = 0


def _model_device(model):
    return next(model.parameters()).device


def _normalize_prompt_ids(prompt, tokenizer, device):
    if isinstance(prompt, str):
        encoded = tokenizer(prompt, return_tensors="pt")
        return encoded["input_ids"][0].to(device)

    if isinstance(prompt, dict):
        input_ids = prompt["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids.squeeze(0).to(device)

    if isinstance(prompt, torch.Tensor):
        return prompt.squeeze(0).to(device)

    return torch.tensor(prompt, dtype=torch.long, device=device)


def _pad_branches(branches, pad_token_id, device):
    lengths = torch.tensor([branch.input_ids.numel() for branch in branches], device=device)
    max_len = int(lengths.max().item())
    batch = torch.full(
        (len(branches), max_len),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )

    for row, branch in enumerate(branches):
        batch[row, : branch.input_ids.numel()] = branch.input_ids

    attention_mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return batch, attention_mask.long(), lengths


def _minimal_threshold_tokens(log_probs_row, node_out_cum_prob, initial_top_k=32):
    vocab_size = log_probs_row.numel()
    top_k = min(initial_top_k, vocab_size)

    while True:
        top_log_probs, token_ids = torch.topk(log_probs_row, k=top_k)
        cumulative_probs = top_log_probs.exp().cumsum(dim=0)
        crosses_threshold = cumulative_probs > node_out_cum_prob

        if bool(crosses_threshold.any()) or top_k == vocab_size:
            if bool(crosses_threshold.any()):
                cutoff = int(torch.nonzero(crosses_threshold, as_tuple=False)[0].item()) + 1
            else:
                cutoff = top_k
            return token_ids[:cutoff], top_log_probs[:cutoff]

        top_k = min(top_k * 2, vocab_size)


def _decode_branch(branch, tokenizer):
    input_ids = branch.input_ids.detach().cpu().tolist()
    generated_ids = input_ids[-branch.generated_tokens :] if branch.generated_tokens else []
    return {
        "input_ids": input_ids,
        "text": tokenizer.decode(input_ids, skip_special_tokens=True),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "log_prob": branch.log_prob,
        "generated_tokens": branch.generated_tokens,
    }


def _parse_answer(text):
    final_answer_match = re.search(r"####\s*([^\n]+)", text)
    if final_answer_match:
        return final_answer_match.group(1).strip()

    number_matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if number_matches:
        return number_matches[-1].replace(",", "")

    return text.strip()


def _answers_match(parsed_answer, real_answer):
    return parsed_answer.strip() == real_answer.strip()


def _best_branch(branches):
    if branches["finished_branches"]:
        return branches["finished_branches"][0]
    if branches["unfinished_branches"]:
        return branches["unfinished_branches"][0]
    return None


def brute_force_cot_tree(
    prompt,
    node_out_cum_prob,
    model,
    tokenizer,
    max_new_tokens=256,
    max_live_branches=None,
):
    if not 0 < node_out_cum_prob < 1:
        raise ValueError("node_out_cum_prob must be between 0 and 1.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if max_live_branches is not None and max_live_branches <= 0:
        raise ValueError("max_live_branches must be positive when provided.")

    device = _model_device(model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define a pad token or an eos token.")

    prompt_ids = _normalize_prompt_ids(prompt, tokenizer, device)
    live_branches = [Branch(input_ids=prompt_ids, log_prob=0.0)]
    finished_branches = []

    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if not live_branches:
                break

            batch, attention_mask, lengths = _pad_branches(live_branches, pad_token_id, device)
            outputs = model(input_ids=batch, attention_mask=attention_mask)
            next_token_logits = outputs.logits[
                torch.arange(len(live_branches), device=device),
                lengths - 1,
                :,
            ]
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

            next_live_branches = []
            for branch_idx, branch in enumerate(live_branches):
                token_ids, token_log_probs = _minimal_threshold_tokens(
                    next_token_log_probs[branch_idx],
                    node_out_cum_prob,
                )

                for token_id, token_log_prob in zip(token_ids.tolist(), token_log_probs.tolist()):
                    child = Branch(
                        input_ids=torch.cat(
                            [branch.input_ids, torch.tensor([token_id], device=device)]
                        ),
                        log_prob=branch.log_prob + float(token_log_prob),
                        generated_tokens=branch.generated_tokens + 1,
                    )

                    if token_id == tokenizer.eos_token_id:
                        finished_branches.append(child)
                    else:
                        next_live_branches.append(child)

            if max_live_branches is not None and len(next_live_branches) > max_live_branches:
                next_live_branches.sort(key=lambda branch: branch.log_prob, reverse=True)
                next_live_branches = next_live_branches[:max_live_branches]

            live_branches = next_live_branches

    finished_branches.sort(key=lambda branch: branch.log_prob, reverse=True)
    live_branches.sort(key=lambda branch: branch.log_prob, reverse=True)
    return {
        "finished_branches": [_decode_branch(branch, tokenizer) for branch in finished_branches],
        "unfinished_branches": [_decode_branch(branch, tokenizer) for branch in live_branches],
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            ""
        ),
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        metavar="K",
        help="Maximum tree depth after the prompt.",
    )
    p.add_argument(
        "--max-problems",
        type=int,
        default=None,
        metavar="N",
        help="HF only: evaluate at most N examples (default: full split)",
    )
    p.add_argument(
        "--node-out-cum-prob",
        type=float,
        default=None,
        required=True
    )
    p.add_argument(
        "--max-live-branches",
        type=int,
        default=None,
        metavar="N",
        help="Optional cap on live branches, keeping highest log-probability branches.",
    )
    p.add_argument(
        "--output-json",
        default="graph_cot_results.json",
        help="Path to write per-problem JSON results. Relative paths are saved in this script's directory.",
    )
    return p.parse_args(argv)

if __name__ == "__main__":
    args = _parse_args()

    model, tokenizer = load_model(model_name)
    max_problems = len(ds) if args.max_problems is None else min(args.max_problems, len(ds))
    results = []
    for i in range(max_problems):
        example = ds[i]
        prompt = example["question"] + " Let's think step by step."
        branches = brute_force_cot_tree(
            prompt,
            args.node_out_cum_prob,
            model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            max_live_branches=args.max_live_branches,
        )

        best_branch = _best_branch(branches)
        best_text = "" if best_branch is None else best_branch["generated_text"]
        full_branch_text = "" if best_branch is None else best_branch["text"]
        log_prob = None if best_branch is None else best_branch["log_prob"]
        parsed_answer = _parse_answer(best_text)
        real_answer = _parse_answer(example["answer"])
        is_correct = _answers_match(parsed_answer, real_answer)

        result = {
            "problem_index": i,
            "question": example["question"],
            "prediction": best_text,
            "full_branch_text": full_branch_text,
            "parsed_answer": parsed_answer,
            "real_answer": real_answer,
            "log_prob": log_prob,
            "is_correct": is_correct,
            "finished_branch_count": len(branches["finished_branches"]),
            "unfinished_branch_count": len(branches["unfinished_branches"]),
        }
        results.append(result)

        print(
            f"Problem {i + 1}/{max_problems}: "
            f"log_prob={log_prob}, parsed_answer={parsed_answer}, real_answer={real_answer}, "
            f"finished={len(branches['finished_branches'])}, "
            f"unfinished={len(branches['unfinished_branches'])}"
        )

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} results to {output_path}")
    correct_count = sum(result["is_correct"] for result in results)
    accuracy = correct_count / len(results) if results else 0.0
    print(f"Final accuracy: {correct_count}/{len(results)} = {accuracy:.2%}")



