import argparse
import json
import logging
from collections import defaultdict

import regex
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

# GPT-2 pre-tokenization pattern: splits text into word-like chunks
GPT2_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build superchunk tokens via chunk-constrained BPE on GSM8K traces")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--traces", required=True, help="Path to GSM8K traces JSON")
    parser.add_argument("--num_superchunks", type=int, required=True,
                        help="Number of BPE merges to perform")
    parser.add_argument("--max_chunks_per_token", type=int, required=True,
                        help="Maximum number of GPT-2 chunks a new superchunk token may span")
    parser.add_argument("--output", default="superchunks.json", help="Output JSON file path")
    return parser.parse_args()


def build_corpus(traces, tokenizer):
    """
    Tokenizes each trace's generated_response at the individual HF token level.
    Each symbol is (token_text, num_gpt2_chunks, (hf_token_id,)) where:
      - num_gpt2_chunks = 1  if the token's char span exactly matches a GPT-2 chunk
      - num_gpt2_chunks = 0  if the token is a partial/subword chunk (never merged)
    """
    corpus = []
    for item in traces:
        text = item.get("generated_response", "")
        if not text.strip():
            continue

        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        chunk_spans = {(m.start(), m.end()) for m in GPT2_PATTERN.finditer(text)}

        seq = []
        for tid, (ts, te) in zip(token_ids, offsets):
            if te <= ts:
                continue
            token_text = text[ts:te]
            is_full_chunk = (ts, te) in chunk_spans
            seq.append((token_text, 1 if is_full_chunk else 0, (tid,)))

        if seq:
            corpus.append(seq)

    return corpus


def count_pairs(corpus, max_chunks):
    counts = defaultdict(int)
    for seq in corpus:
        for i in range(len(seq) - 1):
            l_text, l_nc, _ = seq[i]
            r_text, r_nc, _ = seq[i + 1]
            # Both sides must be full chunks (or previously merged superchunks)
            if l_nc >= 1 and r_nc >= 1 and l_nc + r_nc <= max_chunks:
                counts[(l_text, r_text)] += 1
    return counts


def apply_merge(corpus, left_text, right_text):
    new_corpus = []
    for seq in corpus:
        new_seq = []
        i = 0
        while i < len(seq):
            if (
                i < len(seq) - 1
                and seq[i][0] == left_text
                and seq[i + 1][0] == right_text
            ):
                new_seq.append((
                    left_text + right_text,
                    seq[i][1] + seq[i + 1][1],
                    seq[i][2] + seq[i + 1][2],
                ))
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        new_corpus.append(new_seq)
    return new_corpus


def run_bpe(corpus, tokenizer, num_superchunks, max_chunks_per_token):
    base_vocab_size = tokenizer.vocab_size
    symbol_to_id = {}
    next_id = base_vocab_size

    def get_or_assign_id(text, tids):
        nonlocal next_id
        if text in symbol_to_id:
            return symbol_to_id[text]
        # Single HF token: reuse its existing ID; merged superchunk: assign a new one
        if len(tids) == 1:
            symbol_to_id[text] = tids[0]
        else:
            symbol_to_id[text] = next_id
            next_id += 1
        return symbol_to_id[text]

    # Pre-assign IDs for all initial tokens
    for seq in corpus:
        for text, _nc, tids in seq:
            get_or_assign_id(text, tids)

    results = []

    for step in range(num_superchunks):
        pair_counts = count_pairs(corpus, max_chunks_per_token)
        if not pair_counts:
            log.info("step %d/%d: no valid pairs remain", step, num_superchunks)
            break

        best_left, best_right = max(pair_counts, key=pair_counts.get)
        freq = pair_counts[(best_left, best_right)]

        # Retrieve tids for this pair from the corpus
        left_tids = right_tids = None
        for seq in corpus:
            for i in range(len(seq) - 1):
                if seq[i][0] == best_left and seq[i + 1][0] == best_right:
                    left_tids = seq[i][2]
                    right_tids = seq[i + 1][2]
                    break
            if left_tids is not None:
                break

        merged_text = best_left + best_right
        merged_id = next_id
        next_id += 1
        symbol_to_id[merged_text] = merged_id

        results.append({
            "token_string": merged_text,
            "token_id": merged_id,
            "merged_from": [
                {"token_string": best_left, "token_id": get_or_assign_id(best_left, left_tids)},
                {"token_string": best_right, "token_id": get_or_assign_id(best_right, right_tids)},
            ],
        })

        corpus = apply_merge(corpus, best_left, best_right)
        log.info("[%d/%d] %r + %r -> %r (freq=%d)", step + 1, num_superchunks, best_left, best_right, merged_text, freq)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    log.info("loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    log.info("loading traces from %s", args.traces)
    with open(args.traces) as f:
        traces = json.load(f)
    log.info("%d traces loaded", len(traces))

    log.info("building corpus")
    corpus = build_corpus(traces, tokenizer)
    total_tokens = sum(len(seq) for seq in corpus)
    full_chunk_tokens = sum(1 for seq in corpus for _, nc, _ in seq if nc >= 1)
    log.info("%d sequences, %d tokens (%d full-chunk tokens eligible for merging)",
             len(corpus), total_tokens, full_chunk_tokens)

    log.info("running BPE: %d merges, max %d chunks/token", args.num_superchunks, args.max_chunks_per_token)
    results = run_bpe(corpus, tokenizer, args.num_superchunks, args.max_chunks_per_token)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("saved %d superchunks to %s", len(results), args.output)


if __name__ == "__main__":
    main()
