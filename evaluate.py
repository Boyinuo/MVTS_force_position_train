"""Evaluate a checkpoint on its saved test partition, including after migration."""
import argparse

import torch
from torch.utils.data import DataLoader

from mvts.data import TactileDataset, discover_records, write_json
from mvts.engine import choose_device, load_checkpoint, run_epoch, save_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs/evaluation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    from pathlib import Path
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, checkpoint, config = load_checkpoint(args.checkpoint, device)
    records, report = discover_records(args.data_root, feature_config=config)
    write_json(output / "data_report.json", report)
    if report["rejected"]:
        raise ValueError("Rejected data; inspect evaluation data_report.json")
    by_id = {r["id"]: r for r in records}
    # Resolve paths from current data-root instead of stale absolute checkpoint paths.
    records = [by_id[k] for k in checkpoint["split_ids"]["test"]]
    if [r["target"] for r in records] != checkpoint["split_labels"]["test"]:
        raise ValueError("Test labels differ from training checkpoint")
    if any(abs(r["sample_rate"] - config.sample_rate) > 1 for r in records):
        raise ValueError("Sample rate differs from training checkpoint")
    loader = DataLoader(TactileDataset(records, config, checkpoint["stats"]), batch_size=args.batch_size,
                        pin_memory=(device.type == "cuda"))
    metrics, values = run_epoch(model, loader, device, checkpoint["stats"],
                                position_weight=checkpoint["args"]["position_weight"],
                                wrench_weight=checkpoint["args"]["wrench_weight"],
                                progress_label="Evaluation")
    write_json(output / "metrics.json", metrics)
    save_predictions(output / "predictions.csv", values)
    print(metrics)


if __name__ == "__main__":
    main()
