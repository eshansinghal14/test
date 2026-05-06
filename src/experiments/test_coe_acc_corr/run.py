import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
EXPERIMENT_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure correlation between current-step COE and answer correctness.",
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to graph-cot results JSON with question, prediction, and is_correct fields.",
    )
    parser.add_argument(
        "--dataset-json",
        default=str(EXPERIMENT_DIR / "coe_acc_corr_steps.json"),
        help="Path to write the step-level JSON dataset.",
    )
    parser.add_argument(
        "--plot-path",
        default=str(EXPERIMENT_DIR / "coe_acc_corr.png"),
        help="Path to write the accuracy-vs-COE plot.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name used for COE scoring.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optionally score at most this many step examples.",
    )
    return parser.parse_args()


def _load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")
    return records


def _save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _build_prompt(question: str) -> str:
    return question + " Let's think step by step."


def _split_steps(prediction: str) -> List[str]:
    return [step.strip() for step in prediction.split(". ") if step.strip()]


def _as_correct_label(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes"} else 0
    return 1 if bool(value) else 0


def _build_step_dataset(records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    dataset = []
    skipped_empty_predictions = 0

    for problem_index, record in enumerate(records):
        question = str(record.get("question") or "")
        prediction = str(record.get("prediction") or "")
        steps = _split_steps(prediction)
        if not steps:
            skipped_empty_predictions += 1
            continue

        correct = _as_correct_label(record.get("is_correct"))
        prompt = _build_prompt(question)
        for step_index, current_step in enumerate(steps):
            dataset.append(
                {
                    "problem_index": problem_index,
                    "step_index": step_index,
                    "prompt": prompt,
                    "prior_steps": ". ".join(steps[:step_index]),
                    "current_step": current_step,
                    "correct": correct,
                }
            )

    return dataset, skipped_empty_predictions


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _release_torch_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _prefix_text(prompt: str, prior_steps: str) -> str:
    if prior_steps:
        return f"{prompt} {prior_steps}. "
    return f"{prompt} "


def _calculate_current_step_coe(
    row: Dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    eps: float = 1e-8,
) -> Optional[float]:
    current_step = str(row["current_step"])
    prefix_text = _prefix_text(str(row["prompt"]), str(row["prior_steps"]))
    full_text = prefix_text + current_step

    current_ids = tokenizer(
        current_step,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    if current_ids.numel() == 0:
        return None

    encoded = tokenizer(full_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    current_token_count = int(current_ids.numel())
    output_start = input_ids.size(1) - current_token_count
    if output_start < 0:
        return None

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) < 3:
        return None

    layer_means = torch.stack(
        [
            hidden_state[0, output_start:, :].float().mean(dim=0)
            for hidden_state in hidden_states[1:]
        ],
        dim=0,
    )
    norms = torch.linalg.vector_norm(layer_means, dim=1)
    mag_changes = torch.abs(norms[1:] - norms[:-1])
    dot_products = (layer_means[:-1] * layer_means[1:]).sum(dim=1)
    cosine = dot_products / (norms[:-1] * norms[1:]).clamp_min(eps)
    ang_changes = torch.acos(cosine.clamp(-1.0, 1.0))
    coe_c = torch.abs(torch.complex(mag_changes.mean(), ang_changes.mean()))

    score = float(coe_c.item())
    del outputs, hidden_states, layer_means, norms, mag_changes, dot_products, cosine, ang_changes
    return score


def _score_dataset(
    dataset: List[Dict[str, Any]],
    model_name: str,
    max_steps: Optional[int],
) -> tuple[int, int]:
    from utils import load_model  # type: ignore[import-not-found]

    model, tokenizer = load_model(model_name)
    model.eval()
    device = _model_device(model)

    rows_to_score = dataset if max_steps is None else dataset[:max_steps]
    scored_count = 0
    skipped_count = 0
    for row in rows_to_score:
        coe_c = _calculate_current_step_coe(row, model, tokenizer, device)
        row["coe_c"] = coe_c
        if coe_c is None:
            skipped_count += 1
        else:
            scored_count += 1
        _release_torch_memory(device)

    for row in dataset[len(rows_to_score) :]:
        row["coe_c"] = None

    return scored_count, skipped_count


def _plot_accuracy_vs_coe(dataset: List[Dict[str, Any]], path: Path) -> None:
    scored_rows = [row for row in dataset if row.get("coe_c") is not None]
    if not scored_rows:
        print("No scored rows available for plotting.")
        return

    x_values = [float(row["coe_c"]) for row in scored_rows]
    y_values = [int(row["correct"]) for row in scored_rows]
    y_jitter = [y + (0.04 if index % 2 else -0.04) for index, y in enumerate(y_values)]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.scatter(x_values, y_jitter, alpha=0.65, s=24)
    plt.yticks([0, 1], ["Incorrect", "Correct"])
    plt.xlabel("Current-step COE")
    plt.ylabel("Answer correctness")
    plt.title("Accuracy vs. Current-Step COE")
    plt.ylim(-0.25, 1.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _mean_coe(dataset: List[Dict[str, Any]], correct: int) -> Optional[float]:
    values = [
        float(row["coe_c"])
        for row in dataset
        if row.get("coe_c") is not None and int(row["correct"]) == correct
    ]
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_json)
    dataset_path = Path(args.dataset_json)
    plot_path = Path(args.plot_path)

    records = _load_results(input_path)
    dataset, skipped_empty_predictions = _build_step_dataset(records)
    _save_json(dataset_path, dataset)
    print(f"Wrote {len(dataset)} step rows to {dataset_path}")

    scored_count, skipped_score_count = _score_dataset(dataset, args.model_name, args.max_steps)
    _save_json(dataset_path, dataset)
    _plot_accuracy_vs_coe(dataset, plot_path)

    correct_mean = _mean_coe(dataset, correct=1)
    incorrect_mean = _mean_coe(dataset, correct=0)
    correct_mean_log = "n/a" if correct_mean is None else f"{correct_mean:.4f}"
    incorrect_mean_log = "n/a" if incorrect_mean is None else f"{incorrect_mean:.4f}"

    print(f"Scored steps: {scored_count}")
    print(f"Skipped empty predictions: {skipped_empty_predictions}")
    print(f"Skipped COE scores: {skipped_score_count}")
    print(f"Mean COE for correct steps: {correct_mean_log}")
    print(f"Mean COE for incorrect steps: {incorrect_mean_log}")
    print(f"Updated dataset JSON: {dataset_path}")
    print(f"Plot path: {plot_path}")


if __name__ == "__main__":
    main()
