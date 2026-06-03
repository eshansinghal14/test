import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from generate_gsm8k_traces import build_prompt
from utils import load_model
from train_superchunk_lm_head import load_superchunks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train superchunk input embeddings via token distillation (MSE on hidden states)"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--traces", required=True, help="Path to train gsm8k_traces.json")
    parser.add_argument("--superchunks", required=True, help="Path to superchunks JSON")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of SC occurrences (windows) per optimizer step")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Accumulate gradients over this many micro-batches before optimizer step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=-1,
                        help="Save a checkpoint every N steps, or -1 to disable")
    parser.add_argument("--output-dir", default="superchunk_embeddings_ckpts",
                        help="Directory to save step_N.pt checkpoints")
    return parser.parse_args()


def create_sc_embeddings(model, superchunks, sc_idx_to_leaves, vocab_size):
    """
    Initialize one float32 input embedding per superchunk as the mean of its
    constituent leaf token embeddings.  Float32 prevents AdamW second-moment
    underflow (same rationale as new_weight in train_superchunk_tokens.py).
    """
    embed_w = model.model.embed_tokens.weight  # [V, H]
    device = embed_w.device
    rows = []
    for i in range(len(superchunks)):
        leaf_ids = sc_idx_to_leaves[vocab_size + i]
        leaf_vecs = embed_w[list(leaf_ids)].detach().clone().float()
        rows.append(leaf_vecs.mean(dim=0))
    return torch.nn.Parameter(torch.stack(rows).to(device))  # [N_sc, H], float32


def collect_occurrences(token_ids, sc_lookup, max_span):
    """Greedy longest-match scan. Returns list of (pos, span, sc_idx)."""
    occurrences = []
    n = len(token_ids)
    i = 0
    while i < n:
        found = False
        for length in range(min(max_span, n - i), 1, -1):
            key = tuple(token_ids[i : i + length])
            if key in sc_lookup:
                occurrences.append((i, length, sc_lookup[key]))
                i += length
                found = True
                break
        if not found:
            i += 1
    return occurrences


def collect_all_sequences(traces, tokenizer, sc_lookup, max_span):
    """Returns list of (token_ids, occurrences) for traces that contain at least one SC."""
    sequences = []
    for trace in traces:
        prompt = build_prompt(trace["question"], tokenizer)
        response = trace["generated_response"]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        full_ids = prompt_ids + response_ids
        occurrences = collect_occurrences(full_ids, sc_lookup, max_span)
        if occurrences:
            sequences.append((full_ids, occurrences))
    return sequences


def distillation_loss(model, sc_embeddings, batch_occurrences, all_sequences, vocab_size, grad_scale=1.0):
    """
    One teacher forward pass per unique sequence in the batch, then one batched
    student forward pass across all occurrences.  The model is loaded with
    device_map="auto" so it is already spread across all available GPUs.

    grad_scale should be 1/grad_accum for gradient accumulation.
    Returns the mean per-occurrence MSE loss for logging.
    """
    if not batch_occurrences:
        return 0.0

    device = next(model.parameters()).device
    embed_module = model.model.embed_tokens
    model_dtype = embed_module.weight.dtype
    H = sc_embeddings.shape[1]

    # One teacher forward pass per unique sequence in the batch (no_grad).
    seq_cache = {}
    for seq_idx, _, _, _ in batch_occurrences:
        if seq_idx in seq_cache:
            continue
        token_ids, _ = all_sequences[seq_idx]
        t_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            t_seq = model.model(input_ids=t_ids).last_hidden_state[0].float()
            base_embeds = embed_module(t_ids[0]).detach()
        seq_cache[seq_idx] = (token_ids, t_seq, base_embeds)

    # Build student embedding sequences and pair each with its teacher target.
    items = []
    for seq_idx, pos, span, sc_idx in batch_occurrences:
        token_ids, t_seq, base_embeds = seq_cache[seq_idx]
        if pos + span >= len(token_ids):
            continue
        sc_emb = sc_embeddings[sc_idx - vocab_size].to(model_dtype)
        parts = []
        if pos > 0:
            parts.append(base_embeds[:pos])
        parts.append(sc_emb.unsqueeze(0))
        parts.append(base_embeds[pos + span:])
        items.append((torch.cat(parts, dim=0), t_seq[pos + span:].detach()))

    if not items:
        return 0.0

    # Batched student forward pass (left-padded; torch.stack preserves grad).
    n = len(items)
    max_s_len = max(s.shape[0] for s, _ in items)
    padded = []
    for s_seq, _ in items:
        pl = max_s_len - s_seq.shape[0]
        if pl > 0:
            pad = torch.zeros(pl, H, dtype=s_seq.dtype, device=device)
            padded.append(torch.cat([pad, s_seq], dim=0))
        else:
            padded.append(s_seq)
    s_batch = torch.stack(padded, dim=0)  # [n, max_s_len, H]
    s_mask = torch.zeros(n, max_s_len, dtype=torch.long, device=device)
    for i, (s_seq, _) in enumerate(items):
        s_mask[i, max_s_len - s_seq.shape[0]:] = 1

    s_all = model.model(inputs_embeds=s_batch, attention_mask=s_mask).last_hidden_state.float()

    loss = sum(
        F.mse_loss(s_all[i, -t_cont.shape[0]:], t_cont)
        for i, (_, t_cont) in enumerate(items)
    ) / n
    (loss * grad_scale).backward()
    return loss.item()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # device_map="auto" spreads the model across all available GPUs, keeping per-GPU
    # weight memory at ~weights/n_gpus and leaving room for activation memory.
    model, tokenizer = load_model(args.model)
    for param in model.parameters():
        param.requires_grad_(False)
    model.train()  # must be train mode for gradient_checkpointing (gated on self.training)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    vocab_size = model.lm_head.weight.shape[0]

    with open(args.traces) as f:
        train_traces = json.load(f)
    print(f"Train traces: {len(train_traces)}")

    superchunks, sc_lookup, _sc_first_leaf, sc_idx_to_leaves, max_span = load_superchunks(
        args.superchunks, vocab_size
    )
    print(f"Superchunks: {len(superchunks)}, max_span={max_span}")

    sc_embeddings = create_sc_embeddings(model, superchunks, sc_idx_to_leaves, vocab_size)
    print(f"sc_embeddings: {tuple(sc_embeddings.shape)}")

    os.makedirs(args.output_dir, exist_ok=True)
    optimizer = AdamW([sc_embeddings], lr=args.lr)

    all_sequences = collect_all_sequences(train_traces, tokenizer, sc_lookup, max_span)

    # Flat list of (seq_idx, pos, span, sc_idx) — one entry per valid occurrence.
    all_occurrences = [
        (seq_idx, pos, span, sc_idx)
        for seq_idx, (token_ids, occs) in enumerate(all_sequences)
        for pos, span, sc_idx in occs
        if pos + span < len(token_ids)
    ]
    print(f"Sequences with SCs: {len(all_sequences)}, total occurrences: {len(all_occurrences)}")

    occ_batches = [all_occurrences[i : i + args.batch_size] for i in range(0, len(all_occurrences), args.batch_size)]
    batch_iter = iter(occ_batches)
    step = 0
    ema_loss = None
    ema_alpha = 0.9
    loss_history = []
    ema_history = []
    plot_path = os.path.join(args.output_dir, "loss.png")
    grad_scale = 1.0 / args.grad_accum

    fig, ax = plt.subplots(figsize=(10, 4))

    while step < args.steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        n_occ = 0

        for _ in range(args.grad_accum):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(occ_batches)
                batch = next(batch_iter)
            accum_loss += distillation_loss(model, sc_embeddings, batch, all_sequences, vocab_size, grad_scale)
            n_occ += len(batch)

        optimizer.step()
        step += 1

        loss = accum_loss / args.grad_accum
        ema_loss = loss if ema_loss is None else ema_alpha * ema_loss + (1 - ema_alpha) * loss
        loss_history.append(loss)
        ema_history.append(ema_loss)
        print(f"[S{step}] loss={loss:.4f}  ema={ema_loss:.4f}  n_occ={n_occ}")

        ax.clear()
        steps_range = range(1, step + 1)
        ax.plot(steps_range, loss_history, color="steelblue", alpha=0.35, linewidth=1, label="loss")
        ax.plot(steps_range, ema_history, color="steelblue", linewidth=1.5, label="ema")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("superchunk embedding distillation loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=100)

        if args.checkpoint_every_n_steps > 0 and step % args.checkpoint_every_n_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f"step_{step}.pt")
            torch.save({"sc_embeddings": sc_embeddings.data, "superchunks": superchunks, "model_name": args.model}, ckpt_path)
            print(f"  checkpoint -> {ckpt_path}")

    plt.close(fig)

    final_path = os.path.join(args.output_dir, f"step_{step}.pt")
    torch.save({"sc_embeddings": sc_embeddings.data, "superchunks": superchunks, "model_name": args.model}, final_path)
    print(f"\nSaved trained sc_embeddings to {final_path}")


if __name__ == "__main__":
    main()
