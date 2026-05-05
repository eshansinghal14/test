import argparse
import gc
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


def _calculate_coe_c(branch, model, eps=1e-8):
    if branch.generated_tokens == 0:
        return None

    with torch.inference_mode():
        input_ids = branch.input_ids.unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) < 3:
        return None

    output_start = branch.input_ids.numel() - branch.generated_tokens
    layer_means = torch.stack(
        [hidden_state[0, output_start:, :].float().mean(dim=0) for hidden_state in hidden_states[1:]],
        dim=0,
    )
    norms = torch.linalg.vector_norm(layer_means, dim=1)
    mag_changes = torch.abs(norms[1:] - norms[:-1])
    dot_products = (layer_means[:-1] * layer_means[1:]).sum(dim=1)
    cosine = dot_products / (norms[:-1] * norms[1:]).clamp_min(eps)
    ang_changes = torch.acos(cosine.clamp(-1.0, 1.0))

    coe_c = torch.abs(torch.complex(mag_changes.mean(), ang_changes.mean()))
    del outputs, hidden_states, layer_means, norms, mag_changes, dot_products, cosine, ang_changes
    return float(coe_c.item())


def _decode_branch(branch, tokenizer, model):
    input_ids = branch.input_ids.detach().cpu().tolist()
    generated_ids = input_ids[-branch.generated_tokens :] if branch.generated_tokens else []
    return {
        "input_ids": input_ids,
        "text": tokenizer.decode(input_ids, skip_special_tokens=True),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "coe_c": _calculate_coe_c(branch, model),
        "generated_tokens": branch.generated_tokens,
    }


def _release_torch_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


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


def _first_available_branch(branches):
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
    keep_branch_details=False,
):
    if not 0 < node_out_cum_prob < 1:
        raise ValueError("node_out_cum_prob must be between 0 and 1.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    device = _model_device(model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define a pad token or an eos token.")

    prompt_ids = _normalize_prompt_ids(prompt, tokenizer, device)
    live_branches = [Branch(input_ids=prompt_ids)]
    finished_branches = []
    first_finished_branch = None
    finished_branch_count = 0

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
                token_ids, _ = _minimal_threshold_tokens(
                    next_token_log_probs[branch_idx],
                    node_out_cum_prob,
                )

                for token_id in token_ids.tolist():
                    child = Branch(
                        input_ids=torch.cat(
                            [branch.input_ids, torch.tensor([token_id], device=device)]
                        ),
                        generated_tokens=branch.generated_tokens + 1,
                    )

                    if token_id == tokenizer.eos_token_id:
                        decoded_child = None
                        finished_branch_count += 1
                        if first_finished_branch is None:
                            decoded_child = _decode_branch(child, tokenizer, model)
                            first_finished_branch = decoded_child
                        if keep_branch_details:
                            if decoded_child is None:
                                decoded_child = _decode_branch(child, tokenizer, model)
                            finished_branches.append(decoded_child)
                        del child
                    else:
                        next_live_branches.append(child)

            del batch, attention_mask, lengths, outputs, next_token_logits, next_token_log_probs
            live_branches.clear()
            live_branches = next_live_branches

    _release_torch_memory(device)

    unfinished_branch_count = len(live_branches)
    if keep_branch_details:
        decoded_unfinished_branches = [_decode_branch(branch, tokenizer, model) for branch in live_branches]
    elif live_branches:
        decoded_unfinished_branches = [_decode_branch(live_branches[0], tokenizer, model)]
    else:
        decoded_unfinished_branches = []
    live_branches.clear()
    _release_torch_memory(device)

    return {
        "finished_branches": (
            finished_branches
            if keep_branch_details
            else ([first_finished_branch] if first_finished_branch is not None else [])
        ),
        "unfinished_branches": decoded_unfinished_branches,
        "finished_branch_count": finished_branch_count,
        "unfinished_branch_count": unfinished_branch_count,
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
        "--output-json",
        default="graph_cot_results.json",
        help="Path to write per-problem JSON results. Relative paths are saved from the current working directory.",
    )
    p.add_argument(
        "--keep-branch-details",
        action="store_true",
        help="Keep every decoded branch in memory. By default only the first branch and counts are retained.",
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
            keep_branch_details=args.keep_branch_details,
        )

        selected_branch = _first_available_branch(branches)
        selected_text = "" if selected_branch is None else selected_branch["generated_text"]
        full_branch_text = "" if selected_branch is None else selected_branch["text"]
        coe_c = None if selected_branch is None else selected_branch["coe_c"]
        parsed_answer = _parse_answer(selected_text)
        real_answer = _parse_answer(example["answer"])
        is_correct = _answers_match(parsed_answer, real_answer)

        result = {
            "problem_index": i,
            "question": example["question"],
            "prediction": selected_text,
            "full_branch_text": full_branch_text,
            "parsed_answer": parsed_answer,
            "real_answer": real_answer,
            "coe_c": coe_c,
            "is_correct": is_correct,
            "finished_branch_count": branches["finished_branch_count"],
            "unfinished_branch_count": branches["unfinished_branch_count"],
        }
        results.append(result)

        print(
            f"Problem {i + 1}/{max_problems}: "
            f"coe_c={coe_c}, parsed_answer={parsed_answer}, real_answer={real_answer}, "
            f"finished={branches['finished_branch_count']}, "
            f"unfinished={branches['unfinished_branch_count']}"
        )
        del branches, selected_branch, selected_text, full_branch_text
        _release_torch_memory(_model_device(model))

    output_path = Path(args.output_json)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} results to {output_path}")
    correct_count = sum(result["is_correct"] for result in results)
    accuracy = correct_count / len(results) if results else 0.0
    print(f"Final accuracy: {correct_count}/{len(results)} = {accuracy:.2%}")



