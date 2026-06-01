import argparse
import json

import torch
from datasets import load_dataset
from utils import load_model


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

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.select(range(args.num_sequences))

    model, tokenizer = load_model(args.model)
    tokenizer.padding_side = "left"
    model.eval()

    results = []
    examples = list(dataset)

    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start: batch_start + args.batch_size]
        prompts = [build_prompt(ex["question"], tokenizer) for ex in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
            )

        input_len = inputs["input_ids"].shape[1]
        for j, (ex, out) in enumerate(zip(batch, output_ids)):
            generated_response = tokenizer.decode(out[input_len:], skip_special_tokens=True)
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
