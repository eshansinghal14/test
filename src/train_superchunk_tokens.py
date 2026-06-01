import argparse
import json
import random
import re

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW

from utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train superchunk token predictions via unembedding extension")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--traces", required=True, help="Path to train gsm8k_traces.json")
    parser.add_argument("--superchunks", required=True, help="Path to superchunks JSON")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bias-init", type=float, default=-8.0,
                        help="Initial value for per-superchunk output bias (should be strongly negative)")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--n-eval-gsm8k", type=int, default=100,
                        help="Number of GSM8K test examples to evaluate accuracy on")
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="superchunk_params.pt",
                        help="Path to save trained new_weight and new_bias")
    return parser.parse_args()


def get_leaf_ids(token_id, sc_by_id):
    """Recursively expand merged_from to get the sequence of base (leaf) token IDs."""
    if token_id not in sc_by_id:
        return (token_id,)
    sc = sc_by_id[token_id]
    return (
        get_leaf_ids(sc["merged_from"][0]["token_id"], sc_by_id)
        + get_leaf_ids(sc["merged_from"][1]["token_id"], sc_by_id)
    )


def load_superchunks(path, vocab_size):
    """
    Load superchunks and build lookup structures.

    Superchunk token IDs in the JSON (e.g. 128000+) may overlap with special-token
    IDs in the model's full vocabulary, so we remap each superchunk to a fresh index:
        logit_idx = vocab_size + i   (i = position in the superchunks list)

    Returns:
        superchunks  : list of dicts from JSON
        sc_lookup    : {tuple(leaf_token_ids): logit_idx}
        sc_first_leaf: {logit_idx: first_leaf_token_id}
        max_span     : maximum leaf sequence length across all superchunks
    """
    with open(path) as f:
        superchunks = json.load(f)

    sc_by_id = {sc["token_id"]: sc for sc in superchunks}

    sc_lookup = {}
    sc_first_leaf = {}
    max_span = 1

    for i, sc in enumerate(superchunks):
        leaf_ids = get_leaf_ids(sc["token_id"], sc_by_id)
        logit_idx = vocab_size + i
        sc_lookup[leaf_ids] = logit_idx
        sc_first_leaf[logit_idx] = leaf_ids[0]
        max_span = max(max_span, len(leaf_ids))

    return superchunks, sc_lookup, sc_first_leaf, max_span


def build_targets(token_ids, sc_lookup, max_span):
    """
    Build hot-swapped target IDs for a response token sequence.

    Each position i is evaluated independently: find the longest superchunk whose
    leaf tokens match token_ids[i:i+k]. If found, targets[i] = superchunk logit_idx;
    otherwise targets[i] = token_ids[i] (original base token).
    """
    n = len(token_ids)
    targets = list(token_ids)
    for i in range(n):
        best_id = token_ids[i]
        for length in range(2, min(max_span, n - i) + 1):
            key = tuple(token_ids[i : i + length])
            if key in sc_lookup:
                best_id = sc_lookup[key]
        targets[i] = best_id
    return targets


def build_prompt(question, tokenizer):
    messages = [{"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question: {question}\nAnswer:"


def create_params(model, superchunks, sc_first_leaf, bias_init, vocab_size, device):
    """
    Create trainable new_weight [N_sc, hidden_size] and new_bias [N_sc].

    new_weight[i] is copied from lm_head.weight[first_leaf_token_id_of_sc_i].
    new_bias[i] is initialized to bias_init (strongly negative to suppress superchunk
    predictions unless the model has learned to prefer them).
    """
    lm_weight = model.lm_head.weight  # [vocab_size, hidden_size]

    rows = []
    for i in range(len(superchunks)):
        logit_idx = vocab_size + i
        first_leaf = sc_first_leaf[logit_idx]
        rows.append(lm_weight[first_leaf].detach().clone())

    new_weight = torch.nn.Parameter(torch.stack(rows).to(device))
    new_bias = torch.nn.Parameter(
        torch.full((len(superchunks),), bias_init, dtype=lm_weight.dtype, device=device)
    )
    return new_weight, new_bias


def collate_batch(traces, tokenizer, sc_lookup, max_span, device):
    """Tokenize a batch of traces and build hot-swapped label tensors."""
    all_input_ids = []
    all_labels = []

    for trace in traces:
        prompt = build_prompt(trace["question"], tokenizer)
        response = trace["generated_response"]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        hot_swapped = build_targets(response_ids, sc_lookup, max_span)

        all_input_ids.append(prompt_ids + response_ids)
        all_labels.append([-100] * len(prompt_ids) + hot_swapped)

    max_len = max(len(ids) for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id

    padded_input = [ids + [pad_id] * (max_len - len(ids)) for ids in all_input_ids]
    padded_labels = [lab + [-100] * (max_len - len(lab)) for lab in all_labels]

    input_tensor = torch.tensor(padded_input, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(padded_labels, dtype=torch.long, device=device)
    attn_mask = (input_tensor != pad_id).long()

    return input_tensor, attn_mask, labels_tensor


def forward_loss(model, new_weight, new_bias, input_ids, attn_mask, labels, vocab_size):
    """
    Compute cross-entropy loss over response tokens with hot-swapped targets.

    The model is run under torch.no_grad(); only new_weight and new_bias receive
    gradients (through sc_logits, which is computed outside no_grad).
    """
    n_sc = new_weight.shape[0]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]   # [B, T, H]
        std_logits = outputs.logits          # [B, T, vocab_size]

    # Causal LM shift: logit at position i predicts token at position i+1
    hidden_s = hidden[:, :-1]         # [B, T-1, H]
    std_s = std_logits[:, :-1]        # [B, T-1, vocab_size]
    labels_s = labels[:, 1:]          # [B, T-1]

    # Superchunk logits — gradients flow back to new_weight / new_bias
    sc_logits = F.linear(hidden_s, new_weight) + new_bias   # [B, T-1, N_sc]
    full_logits = torch.cat([std_s, sc_logits], dim=-1)     # [B, T-1, vocab_size+N_sc]

    return F.cross_entropy(
        full_logits.reshape(-1, vocab_size + n_sc),
        labels_s.reshape(-1),
        ignore_index=-100,
    )


def eval_superchunk_hit_rate(
    model, new_weight, new_bias, traces, sc_lookup, max_span,
    tokenizer, eval_batch_size, vocab_size, device
):
    """
    For positions where the hot-swapped target is a superchunk token, count how
    often argmax(full_logits) == superchunk_logit_idx.

    Returns (hit_rate, n_correct, n_total).
    """
    n_sc = new_weight.shape[0]
    n_correct = 0
    n_total = 0

    for batch_start in range(0, len(traces), eval_batch_size):
        batch = traces[batch_start : batch_start + eval_batch_size]
        input_ids, attn_mask, labels = collate_batch(batch, tokenizer, sc_lookup, max_span, device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
            hidden_s = outputs.hidden_states[-1][:, :-1]   # [B, T-1, H]
            std_s = outputs.logits[:, :-1]                 # [B, T-1, vocab_size]
            sc_logits = F.linear(hidden_s, new_weight) + new_bias
            full_logits = torch.cat([std_s, sc_logits], dim=-1)

        labels_s = labels[:, 1:]
        preds = full_logits.argmax(dim=-1)

        is_sc = labels_s >= vocab_size  # positions with a superchunk target
        n_total += is_sc.sum().item()
        n_correct += (preds[is_sc] == labels_s[is_sc]).sum().item()

    if n_total == 0:
        return 0.0, 0, 0
    return n_correct / n_total, n_correct, n_total


def extract_answer(text):
    m = re.search(r"####\s*([\d,.-]+)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def generate_test_traces(model, tokenizer, n_samples, max_new_tokens, device):
    """
    Generate responses for the first n_samples examples of the GSM8K test split.
    Called once before training; results are cached and reused at every log step.
    """
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))
    examples = list(dataset)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    traces = []
    gen_batch = 8

    for i in range(0, len(examples), gen_batch):
        batch = examples[i : i + gen_batch]
        prompts = [build_prompt(ex["question"], tokenizer) for ex in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        for ex, out in zip(batch, out_ids):
            generated = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            traces.append({
                "question": ex["question"],
                "answer": ex["answer"],
                "generated_response": generated,
            })

    tokenizer.padding_side = original_padding_side
    return traces


def eval_gsm8k_accuracy(test_traces):
    """
    Compute GSM8K accuracy from pre-generated test traces (no model call needed).
    Since base model params are frozen, responses don't change during training.
    """
    n_correct = sum(
        1 for t in test_traces
        if (p := extract_answer(t["generated_response"])) is not None
        and (g := extract_answer(t["answer"])) is not None
        and abs(p - g) < 1e-3
    )
    return n_correct / len(test_traces), n_correct, len(test_traces)


def main():
    args = parse_args()

    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    vocab_size = model.lm_head.weight.shape[0]
    print(f"Loaded {args.model} on {device}, vocab_size={vocab_size}")

    with open(args.traces) as f:
        train_traces = json.load(f)
    print(f"Train traces: {len(train_traces)}")

    print(f"Generating {args.n_eval_gsm8k} GSM8K test responses (done once)...")
    test_traces = generate_test_traces(
        model, tokenizer, args.n_eval_gsm8k, args.eval_max_new_tokens, device
    )
    print(f"Test traces generated: {len(test_traces)}")

    superchunks, sc_lookup, sc_first_leaf, max_span = load_superchunks(args.superchunks, vocab_size)
    print(f"Superchunks: {len(superchunks)}, max_span={max_span}")

    new_weight, new_bias = create_params(
        model, superchunks, sc_first_leaf, args.bias_init, vocab_size, device
    )
    print(f"new_weight={tuple(new_weight.shape)}, new_bias={tuple(new_bias.shape)}, bias_init={args.bias_init}")

    optimizer = AdamW([new_weight, new_bias], lr=args.lr)

    step = 0
    recent_traces = []
    for epoch in range(args.epochs):
        random.shuffle(train_traces)

        for batch_start in range(0, len(train_traces), args.batch_size):
            batch = train_traces[batch_start : batch_start + args.batch_size]
            recent_traces.extend(batch)
            input_ids, attn_mask, labels = collate_batch(
                batch, tokenizer, sc_lookup, max_span, device
            )

            optimizer.zero_grad()
            loss = forward_loss(model, new_weight, new_bias, input_ids, attn_mask, labels, vocab_size)
            loss.backward()
            optimizer.step()

            step += 1

            if step % args.log_every_n_steps == 0:
                print(f"\n[Epoch {epoch + 1}, Step {step}] loss={loss.item():.4f}")

                train_hr, train_correct, train_total = eval_superchunk_hit_rate(
                    model, new_weight, new_bias,
                    recent_traces, sc_lookup, max_span,
                    tokenizer, args.eval_batch_size, vocab_size, device,
                )
                print(f"  Train SC hit rate: {train_hr:.4f} ({train_correct}/{train_total})")
                recent_traces = []

                test_hr, test_correct, test_total = eval_superchunk_hit_rate(
                    model, new_weight, new_bias,
                    test_traces, sc_lookup, max_span,
                    tokenizer, args.eval_batch_size, vocab_size, device,
                )
                print(f"  Test SC hit rate:  {test_hr:.4f} ({test_correct}/{test_total})")

                gsm8k_acc, gsm8k_correct, gsm8k_total = eval_gsm8k_accuracy(test_traces)
                print(f"  GSM8K accuracy:    {gsm8k_acc:.4f} ({gsm8k_correct}/{gsm8k_total})")

        print(f"Epoch {epoch + 1}/{args.epochs} complete.")

    torch.save({"new_weight": new_weight.data, "new_bias": new_bias.data}, args.output)
    print(f"\nSaved trained params to {args.output}")


if __name__ == "__main__":
    main()
