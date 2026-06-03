import argparse
import concurrent.futures
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW

from generate_gsm8k_traces import build_prompt, generate_traces, load_models_on_all_gpus
from gsm8k_benchmark import eval_gsm8k_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train superchunk token predictions via unembedding extension")
    parser.add_argument("--model", default=None, help="HuggingFace model name (overrides model_name in checkpoint)")
    parser.add_argument("--traces", required=True, help="Path to train gsm8k_traces.json")
    parser.add_argument("--embedding-ckpt", required=True,
                        help="Path to superchunk_embeddings.pt from train_superchunk_embeddings")
    parser.add_argument("--superchunks", default=None,
                        help="Path to superchunks JSON (fallback if embedding-ckpt checkpoint predates superchunk storage)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bias-init", type=float, default=-8.0,
                        help="Initial value for per-superchunk output bias (should be strongly negative)")
    parser.add_argument("--eval-every-n-steps", type=int, default=-1, help="Evaluate every N steps, or -1 to disable")
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--n-eval-gsm8k", type=int, default=100,
                        help="Number of GSM8K test examples to evaluate accuracy on")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Truncate training sequences to this many tokens to limit logit tensor size")
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=-1,
                        help="Save a checkpoint every N steps, or -1 to disable")
    parser.add_argument("--output-dir", default="superchunked_model_ckpts",
                        help="Directory to save step_N.pt checkpoints")
    parser.add_argument("--spec-decoding", action="store_true",
                        help="Use speculative decoding during GSM8K eval: verify each leaf token "
                             "against the base model's argmax and accept only the verified prefix")
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


def _build_superchunk_lookups(superchunks, vocab_size):
    """Build lookup structures from an in-memory superchunks list."""
    sc_by_id = {sc["token_id"]: sc for sc in superchunks}

    sc_lookup = {}
    sc_first_leaf = {}
    sc_idx_to_leaves = {}
    max_span = 1

    for i, sc in enumerate(superchunks):
        leaf_ids = get_leaf_ids(sc["token_id"], sc_by_id)
        logit_idx = vocab_size + i
        sc_lookup[leaf_ids] = logit_idx
        sc_first_leaf[logit_idx] = leaf_ids[0]
        sc_idx_to_leaves[logit_idx] = leaf_ids
        max_span = max(max_span, len(leaf_ids))

    return sc_lookup, sc_first_leaf, sc_idx_to_leaves, max_span


def load_superchunks(path, vocab_size):
    """Load superchunks from JSON and build lookup structures."""
    with open(path) as f:
        superchunks = json.load(f)
    sc_lookup, sc_first_leaf, sc_idx_to_leaves, max_span = _build_superchunk_lookups(superchunks, vocab_size)
    return superchunks, sc_lookup, sc_first_leaf, sc_idx_to_leaves, max_span


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



def create_params(model, superchunks, bias_init, vocab_size, sc_embeddings):
    """
    Create trainable new_weight [N_sc, hidden_size] and new_bias [N_sc].

    new_weight is initialized from sc_embeddings (trained input embeddings from
    train_superchunk_embeddings). new_bias is initialized to bias_init (strongly
    negative to suppress superchunk predictions until learned).

    Placed on lm_head's device so F.linear(hidden, new_weight) is same-device
    regardless of how the model is sharded across GPUs.
    """
    lm_device = model.lm_head.weight.device

    # float32: AdamW momentum buffers inherit param dtype; float16 second-moment
    # underflows to zero (eps=1e-8 < float16 min), causing NaN weights after step 1.
    new_weight = torch.nn.Parameter(sc_embeddings.detach().clone().to(lm_device).float())
    new_bias = torch.nn.Parameter(
        torch.full((len(superchunks),), bias_init, dtype=torch.float32, device=lm_device)
    )
    return new_weight, new_bias


def collate_batch(traces, tokenizer, sc_lookup, max_span, device, max_seq_len=None):
    """Tokenize a batch of traces and build hot-swapped label tensors."""
    all_input_ids = []
    all_labels = []

    for trace in traces:
        prompt = build_prompt(trace["question"], tokenizer)
        response = trace["generated_response"]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        hot_swapped = build_targets(response_ids, sc_lookup, max_span)

        ids = prompt_ids + response_ids
        lab = [-100] * len(prompt_ids) + hot_swapped
        if max_seq_len is not None:
            ids = ids[:max_seq_len]
            lab = lab[:max_seq_len]
        all_input_ids.append(ids)
        all_labels.append(lab)

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

    Standard one-hot cross-entropy: non-SC positions target the base vocab token,
    SC positions target the superchunk logit directly.
    """
    lm_device = new_weight.device
    labels_s = labels[:, 1:].to(lm_device)  # [B, T-1]

    b_idx, t_idx = (labels_s != -100).nonzero(as_tuple=True)
    flat_labels = labels_s[b_idx, t_idx]
    is_sc = flat_labels >= vocab_size
    sc_pos_ids = (flat_labels - vocab_size).clamp(min=0)  # safe index into sc tables

    with torch.no_grad():
        base_out = model.model(input_ids, attention_mask=attn_mask, use_cache=False)
        hidden_s = base_out.last_hidden_state[:, :-1].to(lm_device).float()  # [B, T-1, H]
        del base_out

        # Chunk over vocab to avoid materialising [B, T, vocab_size] in one shot.
        # Peak extra memory per chunk: [B, T, CHUNK] float32.
        lm_w = model.lm_head.weight  # [vocab_size, H]
        lm_b = model.lm_head.bias    # [vocab_size] or None
        B, T = hidden_s.shape[:2]
        base_labels = flat_labels.clamp(0, vocab_size - 1)
        log_Z_std = None
        std_target = torch.zeros(len(b_idx), dtype=torch.float32, device=lm_device)
        CHUNK = 4096
        for v0 in range(0, vocab_size, CHUNK):
            v1 = min(v0 + CHUNK, vocab_size)
            chunk = F.linear(hidden_s, lm_w[v0:v1].float())
            if lm_b is not None:
                chunk = chunk + lm_b[v0:v1].float()
            chunk_lse = torch.logsumexp(chunk, dim=-1)  # [B, T]
            log_Z_std = chunk_lse if log_Z_std is None else torch.logaddexp(log_Z_std, chunk_lse)
            mask = (base_labels >= v0) & (base_labels < v1) & ~is_sc
            if mask.any():
                std_target[mask] = chunk[b_idx[mask], t_idx[mask], base_labels[mask] - v0]
            del chunk

    # Superchunk logits in float32 — gradients flow back to new_weight / new_bias
    sc_logits = F.linear(hidden_s.float(), new_weight.float()) + new_bias.float()  # [B, T-1, N_sc]

    log_Z = torch.logaddexp(log_Z_std, torch.logsumexp(sc_logits, dim=-1))  # [B, T-1]

    sc_raw = sc_logits[b_idx, t_idx, sc_pos_ids]

    # One-hot targets: base token logit for non-SC, superchunk logit for SC positions
    target_logit = std_target.clone()
    target_logit[is_sc] = sc_raw[is_sc]

    return (log_Z[b_idx, t_idx] - target_logit).mean()


def _eval_hr_on_device(model, new_weight, new_bias, traces, sc_lookup, max_span,
                       tokenizer, eval_batch_size, vocab_size, max_seq_len):
    device = next(model.parameters()).device
    lm_device = new_weight.device
    nw = new_weight.to(device)
    nb = new_bias.to(device)
    n_correct = n_total = n_fired = n_positions = n_tp = 0
    for batch_start in range(0, len(traces), eval_batch_size):
        batch = traces[batch_start : batch_start + eval_batch_size]
        input_ids, attn_mask, labels = collate_batch(batch, tokenizer, sc_lookup, max_span, device, max_seq_len)
        with torch.no_grad():
            base_out = model.model(input_ids, attention_mask=attn_mask, use_cache=False)
            hidden_s = base_out.last_hidden_state[:, :-1].to(device)
            del base_out
            B, T, _ = hidden_s.shape
            lm_w = model.lm_head.weight
            std_max_val = torch.full((B, T), float('-inf'), dtype=hidden_s.dtype, device=device)
            std_argmax = torch.zeros(B, T, dtype=torch.long, device=device)
            CHUNK = 4096
            for v0 in range(0, vocab_size, CHUNK):
                chunk = F.linear(hidden_s, lm_w[v0:v0 + CHUNK])
                c_max, c_idx = chunk.max(dim=-1)
                better = c_max > std_max_val
                std_max_val = torch.where(better, c_max, std_max_val)
                std_argmax = torch.where(better, c_idx + v0, std_argmax)
            sc_logits = F.linear(hidden_s.float(), nw.float()) + nb.float()
            sc_max_val, sc_argmax = sc_logits.max(dim=-1)
            sc_wins = sc_max_val > std_max_val.float()
            preds = torch.where(sc_wins, sc_argmax + vocab_size, std_argmax)
        labels_s = labels[:, 1:].to(device)
        valid = labels_s != -100
        is_sc = labels_s >= vocab_size
        n_total += is_sc.sum().item()
        n_correct += (preds[is_sc] == labels_s[is_sc]).sum().item()
        fired = sc_wins & valid
        n_positions += valid.sum().item()
        n_fired += fired.sum().item()
        n_tp += (fired & is_sc & (preds == labels_s)).sum().item()
    return n_correct, n_total, n_fired, n_positions, n_tp


def eval_superchunk_hit_rate(
    models, new_weight, new_bias, traces, sc_lookup, max_span,
    tokenizer, eval_batch_size, vocab_size, max_seq_len=None
):
    """
    Evaluate superchunk prediction quality across all GPUs in parallel.

    Returns (hit_rate, n_correct, n_total, fire_rate, precision).
    """
    num_gpus = len(models)
    sub_size = max(1, (len(traces) + num_gpus - 1) // num_gpus)
    sub_traces = [traces[i : i + sub_size] for i in range(0, len(traces), sub_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(
                _eval_hr_on_device,
                models[i], new_weight, new_bias, sub_traces[i],
                sc_lookup, max_span, tokenizer, eval_batch_size, vocab_size, max_seq_len,
            )
            for i in range(len(sub_traces))
        ]
        results = [f.result() for f in futures]

    n_correct = sum(r[0] for r in results)
    n_total   = sum(r[1] for r in results)
    n_fired   = sum(r[2] for r in results)
    n_positions = sum(r[3] for r in results)
    n_tp      = sum(r[4] for r in results)

    if n_total == 0:
        return 0.0, 0, 0, 0.0, 0.0
    fire_rate = n_fired / n_positions if n_positions > 0 else 0.0
    precision = n_tp / n_fired if n_fired > 0 else 0.0
    return n_correct / n_total, n_correct, n_total, fire_rate, precision


def make_sc_logits_processor(model, new_weight, new_bias, sc_idx_to_leaves, vocab_size,
                              spec_decoding=False):
    """
    Returns (processor, hook_handle).

    Registers a pre-hook on model.lm_head to capture hidden states, then builds
    a LogitsProcessor that augments std logits with sc_logits at each generation step.
    Caller must call hook_handle.remove() when generation is done.
    Each GPU replica needs its own processor+hook pair; use make_sc_logits_processors
    for multi-GPU generation.

    If spec_decoding=True, each SC firing is verified against the base model's greedy
    argmax: the sequence is accepted up to (but not including) the first leaf token
    that the base model would not have chosen. Unverified SC firings fall back to
    standard greedy generation.
    """
    from transformers import LogitsProcessor

    hidden_ref = [None]
    def _capture_hidden(module, args):
        hidden_ref[0] = args[0]
    hook = model.lm_head.register_forward_pre_hook(_capture_hidden)

    class SCLogitsProcessor(LogitsProcessor):
        def __init__(self):
            self.queues = None
            self.n_fired = 0
            self.n_accepted = 0  # SC fired AND at least leaf[0] verified as argmax
            self.n_fired_by_len = {}       # span_len -> firing count (spec_decoding only)
            self.n_full_accepted_by_len = {}  # span_len -> fully-accepted count (spec_decoding only)
            self.n_queue_pops = 0  # tokens emitted via queue (not a model decision step)

        def _spec_verify(self, input_ids_i, scores_i, leaf_ids):
            """Return the accepted prefix of leaf_ids where each token is the base model argmax."""
            if scores_i.argmax().item() != leaf_ids[0]:
                return []
            accepted = [leaf_ids[0]]
            if len(leaf_ids) == 1:
                return accepted

            device = input_ids_i.device
            # Extend input with leaf[0..k-2] so hidden states at T+j-1 predict leaf[j].
            ext = torch.cat([
                input_ids_i,
                torch.tensor(leaf_ids[:-1], dtype=torch.long, device=device),
            ])
            with torch.no_grad():
                hidden_ext = model.model(ext.unsqueeze(0), use_cache=False).last_hidden_state[0]

            T = input_ids_i.shape[0]
            lm_w = model.lm_head.weight.to(hidden_ext.device).float()
            lm_b = model.lm_head.bias
            hidden_check = hidden_ext[T : T + len(leaf_ids) - 1].float()  # [k-1, H]
            logits_check = F.linear(hidden_check, lm_w)
            if lm_b is not None:
                logits_check = logits_check + lm_b.to(hidden_ext.device).float()
            for j, argmax_j in enumerate(logits_check.argmax(dim=-1).tolist()):
                if argmax_j != leaf_ids[j + 1]:
                    break
                accepted.append(leaf_ids[j + 1])
            return accepted

        def __call__(self, input_ids, scores):
            B = scores.shape[0]
            if self.queues is None:
                self.queues = [[] for _ in range(B)]

            last_hidden = hidden_ref[0][:, -1, :].to(new_weight.device).float()
            sc_logits = F.linear(last_hidden, new_weight) + new_bias  # [B, N_sc]
            std_max = scores.max(dim=-1).values.to(new_weight.device)
            sc_max, sc_argmax = sc_logits.max(dim=-1)

            out = scores.clone()
            for i in range(B):
                if self.queues[i]:
                    tok = self.queues[i].pop(0)
                    self.n_queue_pops += 1
                    out[i] = float("-inf")
                    out[i, tok] = float("inf")
                elif sc_max[i] > std_max[i]:
                    leaf_ids = sc_idx_to_leaves[vocab_size + sc_argmax[i].item()]
                    self.n_fired += 1
                    if spec_decoding:
                        k = len(leaf_ids)
                        self.n_fired_by_len[k] = self.n_fired_by_len.get(k, 0) + 1
                        accepted = self._spec_verify(input_ids[i], scores[i], leaf_ids)
                        if accepted:
                            self.n_accepted += 1
                            if len(accepted) == k:
                                self.n_full_accepted_by_len[k] = self.n_full_accepted_by_len.get(k, 0) + 1
                            out[i] = float("-inf")
                            out[i, accepted[0]] = float("inf")
                            self.queues[i].extend(accepted[1:])
                        # else: out[i] stays as scores clone — fall back to standard greedy
                    else:
                        if scores[i].argmax().item() == leaf_ids[0]:
                            self.n_accepted += 1
                        out[i] = float("-inf")
                        out[i, leaf_ids[0]] = float("inf")
                        self.queues[i].extend(leaf_ids[1:])
            return out

    return SCLogitsProcessor(), hook


def make_sc_logits_processors(models, new_weight, new_bias, sc_idx_to_leaves, vocab_size,
                               spec_decoding=False):
    """Create one (processor, hook) pair per GPU replica and return them as parallel lists."""
    pairs = [make_sc_logits_processor(m, new_weight, new_bias, sc_idx_to_leaves, vocab_size,
                                      spec_decoding=spec_decoding)
             for m in models]
    processors, hooks = zip(*pairs)
    return list(processors), list(hooks)


def full_accept_rates_by_len(processors):
    """Aggregate per-span-length full-acceptance counts across processor replicas.

    Returns dict mapping span_len -> (n_full_accepted, n_fired).
    """
    fired = {}
    full = {}
    for p in processors:
        for k, n in p.n_fired_by_len.items():
            fired[k] = fired.get(k, 0) + n
        for k, n in p.n_full_accepted_by_len.items():
            full[k] = full.get(k, 0) + n
    return {k: (full.get(k, 0), n) for k, n in sorted(fired.items())}


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ckpt = torch.load(args.embedding_ckpt, map_location="cpu")
    sc_embeddings = ckpt["sc_embeddings"]
    if "superchunks" in ckpt:
        superchunks = ckpt["superchunks"]
    elif args.superchunks is not None:
        with open(args.superchunks) as f:
            superchunks = json.load(f)
    else:
        raise ValueError("embedding-ckpt checkpoint does not contain superchunks; pass --superchunks")
    if args.model is not None:
        model_name = args.model
    elif "model_name" in ckpt:
        model_name = ckpt["model_name"]
    else:
        raise ValueError("embedding-ckpt checkpoint does not contain model_name; pass --model or re-run train_superchunk_embeddings")

    models, tokenizer = load_models_on_all_gpus(model_name)
    model = models[0]
    device = next(model.parameters()).device

    for m in models:
        for param in m.parameters():
            param.requires_grad_(False)

    vocab_size = model.lm_head.weight.shape[0]
    print(f"Loaded {model_name} on {device}, vocab_size={vocab_size}")

    with open(args.traces) as f:
        train_traces = json.load(f)
    print(f"Train traces: {len(train_traces)}")

    gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_examples = list(gsm8k_dataset.select(range(min(args.n_eval_gsm8k, len(gsm8k_dataset)))))
    print(f"GSM8K eval examples: {len(gsm8k_examples)}")
    sc_lookup, sc_first_leaf, sc_idx_to_leaves, max_span = _build_superchunk_lookups(superchunks, vocab_size)
    print(f"Superchunks: {len(superchunks)}, max_span={max_span}")

    os.makedirs(args.output_dir, exist_ok=True)
    new_weight, new_bias = create_params(model, superchunks, args.bias_init, vocab_size, sc_embeddings)

    plot_path = os.path.join(args.output_dir, "metrics.png")
    loss_steps, loss_hist = [], []
    tr_steps, tr_hr_hist, tr_fire_hist, tr_prec_hist = [], [], [], []
    te_steps, te_hr_hist, te_fire_hist, te_prec_hist, gsm8k_hist = [], [], [], [], []

    fig, axs = plt.subplots(2, 3, figsize=(20, 8))
    ax_loss, ax_hr, ax_fire = axs[0]
    ax_prec, ax_gsm8k, ax_accept = axs[1]

    accept_steps, accept_hist = [], []

    def save_plot():
        for ax in (ax_loss, ax_hr, ax_fire, ax_prec, ax_gsm8k, ax_accept):
            ax.clear()

        if loss_steps:
            ax_loss.plot(loss_steps, loss_hist, color="steelblue", linewidth=1.2)
        ax_loss.set_title("loss"); ax_loss.set_xlabel("step")

        if tr_steps:
            ax_hr.plot(tr_steps, tr_hr_hist, color="seagreen", linewidth=1.2, label="train")
        if te_steps:
            ax_hr.plot(te_steps, te_hr_hist, color="seagreen", linewidth=1.2, linestyle="--", label="test")
        ax_hr.set_title("hit rate"); ax_hr.set_xlabel("step"); ax_hr.legend(fontsize=8)

        if tr_steps:
            ax_fire.plot(tr_steps, tr_fire_hist, color="darkorange", linewidth=1.2, label="train")
        if te_steps:
            ax_fire.plot(te_steps, te_fire_hist, color="darkorange", linewidth=1.2, linestyle="--", label="test")
        ax_fire.set_title("fire rate"); ax_fire.set_xlabel("step"); ax_fire.legend(fontsize=8)

        if tr_steps:
            ax_prec.plot(tr_steps, tr_prec_hist, color="tomato", linewidth=1.2, label="train")
        if te_steps:
            ax_prec.plot(te_steps, te_prec_hist, color="tomato", linewidth=1.2, linestyle="--", label="test")
        ax_prec.set_title("precision"); ax_prec.set_xlabel("step"); ax_prec.legend(fontsize=8)

        if te_steps:
            ax_gsm8k.plot(te_steps, gsm8k_hist, color="mediumpurple", linewidth=1.2)
        ax_gsm8k.set_title("gsm8k accuracy"); ax_gsm8k.set_xlabel("step")

        if accept_steps:
            ax_accept.plot(accept_steps, accept_hist, color="saddlebrown", linewidth=1.2)
        ax_accept.set_title("accept rate (fired SC → model agrees on leaf[0])"); ax_accept.set_xlabel("step")

        fig.tight_layout()
        fig.savefig(plot_path, dpi=100)
    print(f"new_weight={tuple(new_weight.shape)}, new_bias={tuple(new_bias.shape)}, bias_init={args.bias_init}")

    optimizer = AdamW([new_weight, new_bias], lr=args.lr)

    if args.eval_every_n_steps > 0:
        print(f"  generating {len(gsm8k_examples)} test responses...", end="\r", flush=True)
        processors, hooks = make_sc_logits_processors(
            models, new_weight, new_bias, sc_idx_to_leaves, vocab_size,
            spec_decoding=args.spec_decoding,
        )
        try:
            pre_test_traces = generate_traces(
                models, tokenizer, gsm8k_examples, args.max_seq_len,
                batch_size=args.eval_batch_size,
                logits_processor=processors,
                spec_decoding=args.spec_decoding,
            )
        finally:
            for h in hooks:
                h.remove()
        torch.cuda.empty_cache()
        pre_test_hr, pre_correct, pre_total, pre_fire, pre_prec = eval_superchunk_hit_rate(
            models, new_weight, new_bias,
            pre_test_traces, sc_lookup, max_span,
            tokenizer, args.eval_batch_size, vocab_size, args.max_seq_len,
        )
        pre_gsm8k_acc, pre_gsm8k_correct, pre_gsm8k_total = eval_gsm8k_accuracy(pre_test_traces)
        pre_total_base = sum(len(tokenizer.encode(t["generated_response"], add_special_tokens=False)) for t in pre_test_traces)
        pre_queue_pops = sum(p.n_queue_pops for p in processors)
        pre_avg_len = (pre_total_base - pre_queue_pops) / len(pre_test_traces)
        pre_n_fired = sum(p.n_fired for p in processors)
        pre_n_accepted = sum(p.n_accepted for p in processors)
        pre_accept_rate = pre_n_accepted / pre_n_fired if pre_n_fired > 0 else 0.0
        print(f"[pre-train] test_hr={pre_test_hr:.4f} ({pre_correct}/{pre_total})  "
              f"fire={pre_fire:.4f}  prec={pre_prec:.4f}  "
              f"accept={pre_accept_rate:.4f} ({pre_n_accepted}/{pre_n_fired})  "
              f"gsm8k={pre_gsm8k_acc:.4f} ({pre_gsm8k_correct}/{pre_gsm8k_total})  "
              f"avg_len={pre_avg_len:.1f}")
        if args.spec_decoding:
            rates = full_accept_rates_by_len(processors)
            print("  full_accept_by_len: " + "  ".join(f"len{k}={a}/{t}" for k, (a, t) in rates.items()))
        te_steps.append(0)
        te_hr_hist.append(pre_test_hr)
        te_fire_hist.append(pre_fire)
        te_prec_hist.append(pre_prec)
        gsm8k_hist.append(pre_gsm8k_acc)
        accept_steps.append(0)
        accept_hist.append(pre_accept_rate)
        save_plot()
        torch.cuda.empty_cache()

    batches = [train_traces[i:i + args.batch_size] for i in range(0, len(train_traces), args.batch_size)]

    step = 0
    data_iter = iter(batches)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(batches)
            batch = next(data_iter)
        input_ids, attn_mask, labels = collate_batch(
            batch, tokenizer, sc_lookup, max_span, device, args.max_seq_len,
        )

        optimizer.zero_grad()
        loss = forward_loss(model, new_weight, new_bias, input_ids, attn_mask, labels, vocab_size)
        loss.backward()
        optimizer.step()

        step += 1

        train_hr, train_correct, train_total, train_fire, train_prec = eval_superchunk_hit_rate(
            models, new_weight, new_bias,
            batch, sc_lookup, max_span,
            tokenizer, args.eval_batch_size, vocab_size, args.max_seq_len,
        )

        loss_steps.append(step)
        loss_hist.append(loss.item())
        tr_steps.append(step)
        tr_hr_hist.append(train_hr)
        tr_fire_hist.append(train_fire)
        tr_prec_hist.append(train_prec)

        if args.eval_every_n_steps > 0 and step % args.eval_every_n_steps == 0:
            print(f"  generating {len(gsm8k_examples)} test responses...", end="\r", flush=True)
            processors, hooks = make_sc_logits_processors(
                models, new_weight, new_bias, sc_idx_to_leaves, vocab_size,
                spec_decoding=args.spec_decoding,
            )
            try:
                test_traces = generate_traces(
                    models, tokenizer, gsm8k_examples, args.max_seq_len,
                    batch_size=args.eval_batch_size,
                    logits_processor=processors,
                    spec_decoding=args.spec_decoding,
                )
            finally:
                for h in hooks:
                    h.remove()
            torch.cuda.empty_cache()

            test_hr, test_correct, test_total, test_fire, test_prec = eval_superchunk_hit_rate(
                models, new_weight, new_bias,
                test_traces, sc_lookup, max_span,
                tokenizer, args.eval_batch_size, vocab_size, args.max_seq_len,
            )
            gsm8k_acc, gsm8k_correct, gsm8k_total = eval_gsm8k_accuracy(test_traces)
            total_base = sum(len(tokenizer.encode(t["generated_response"], add_special_tokens=False)) for t in test_traces)
            queue_pops = sum(p.n_queue_pops for p in processors)
            avg_len = (total_base - queue_pops) / len(test_traces)
            n_fired_step = sum(p.n_fired for p in processors)
            n_accepted_step = sum(p.n_accepted for p in processors)
            accept_rate = n_accepted_step / n_fired_step if n_fired_step > 0 else 0.0
            te_steps.append(step)
            te_hr_hist.append(test_hr)
            te_fire_hist.append(test_fire)
            te_prec_hist.append(test_prec)
            gsm8k_hist.append(gsm8k_acc)
            accept_steps.append(step)
            accept_hist.append(accept_rate)
            print(f"[S{step}] loss={loss.item():.4f}  bias={new_bias.mean().item():.4f}  "
                  f"train_hr={train_hr:.4f} ({train_correct}/{train_total})  fire={train_fire:.4f}  prec={train_prec:.4f}  "
                  f"test_hr={test_hr:.4f} ({test_correct}/{test_total})  fire={test_fire:.4f}  prec={test_prec:.4f}  "
                  f"accept={accept_rate:.4f} ({n_accepted_step}/{n_fired_step})  "
                  f"gsm8k={gsm8k_acc:.4f} ({gsm8k_correct}/{gsm8k_total})  avg_len={avg_len:.1f}")
            if args.spec_decoding:
                rates = full_accept_rates_by_len(processors)
                print("  full_accept_by_len: " + "  ".join(f"len{k}={a}/{t}" for k, (a, t) in rates.items()))
        else:
            print(f"[S{step}] loss={loss.item():.4f}  bias={new_bias.mean().item():.4f}  "
                  f"train_hr={train_hr:.4f} ({train_correct}/{train_total})  "
                  f"fire={train_fire:.4f}  prec={train_prec:.4f}")

        save_plot()

        if args.checkpoint_every_n_steps > 0 and step % args.checkpoint_every_n_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f"step_{step}.pt")
            torch.save({"new_weight": new_weight.data, "new_bias": new_bias.data}, ckpt_path)
            print(f"  checkpoint -> {ckpt_path}")

        torch.cuda.empty_cache()

    plt.close(fig)

    final_path = os.path.join(args.output_dir, f"step_{step}.pt")
    torch.save({"new_weight": new_weight.data, "new_bias": new_bias.data}, final_path)
    print(f"\nSaved trained params to {final_path}")


if __name__ == "__main__":
    main()
