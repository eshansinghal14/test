import argparse
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--num_sequences", type=int, required=True, help="Number of problems to generate traces for")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of problems to process in parallel")
    parser.add_argument("--output", default="gsm8k_traces.json", help="Output JSON file path")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def build_prompt(question, tokenizer):
    messages = [{"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question: {question}\nAnswer:"


def main():
    args = parse_args()

    dataset = load_dataset("gsm8k", "main", split="train")
    dataset = dataset.select(range(args.num_sequences))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results = []
    examples = list(dataset)

    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start: batch_start + args.batch_size]
        prompts = [build_prompt(ex["question"], tokenizer) for ex in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_lengths = inputs["attention_mask"].sum(dim=1)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, (ex, out, prompt_len) in enumerate(zip(batch, output_ids, prompt_lengths)):
            generated_response = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            results.append({
                "problem_index": batch_start + j,
                "question": ex["question"],
                "answer": ex["answer"],
                "generated_response": generated_response,
            })

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[{len(results)}/{args.num_sequences}] saved to {args.output}")

    print(f"Done. {len(results)} traces saved to {args.output}")


if __name__ == "__main__":
    main()
