"""Predict position and wrench for one exported image + NPZ audio window."""
import argparse

import numpy as np
import torch

from mvts.data import TARGET_NAMES, extract_sample_audio, write_json
from mvts.engine import choose_device, load_checkpoint
from mvts.features import LogMel, load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--force-index", type=int,
                        help="Select one zero-based force/audio pair; grouped NPZ defaults to all pairs")
    parser.add_argument("--previous-contact", type=int, choices=(0, 1),
                        help="Previous binary contact state; defaults to NPZ starts_in_contact")
    parser.add_argument("--output", default="runs/prediction.json")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, checkpoint, config = load_checkpoint(args.checkpoint, device)
    stats = checkpoint["stats"]
    with np.load(args.npz, allow_pickle=False) as data:
        adc = torch.from_numpy(extract_sample_audio(data, config, args.force_index).astype(np.float32))
        contact = args.previous_contact
        if contact is None:
            contact = int(data["starts_in_contact"].item()) if model.config.use_contact else 0
    sequence = adc.ndim == 3
    if not sequence:
        adc = adc.unsqueeze(0)
    audio = LogMel(config)(adc)
    audio = (audio - torch.tensor(stats["audio_mean"])[:, None, None]) / torch.tensor(
        stats["audio_std"])[:, None, None]
    model.eval()
    with torch.inference_mode():
        batch_size = audio.shape[0]
        image = load_image(args.image).unsqueeze(0).expand(batch_size, -1, -1, -1)
        result = model(image.to(device), audio.to(device),
                       torch.full((batch_size,), contact, dtype=torch.long, device=device))
        pred = torch.cat((result["position"], result["wrench"]), dim=-1).cpu()
        pred = pred * torch.tensor(stats["target_std"]) + torch.tensor(stats["target_mean"])
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Non-finite prediction")
    result = {"model_name": model.model_name,
              "checkpoint_epoch": checkpoint["epoch"], "sample_rate_hz": config.sample_rate,
              "units": "position: mm; force: N; torque: N*m; acquisition sign/frame retained"}
    if sequence:
        result["predictions"] = [{"force_index": index,
                                  "prediction": dict(zip(TARGET_NAMES, row))}
                                 for index, row in enumerate(pred.tolist())]
    else:
        result["prediction"] = dict(zip(TARGET_NAMES, pred[0].tolist()))
        if args.force_index is not None:
            result["force_index"] = args.force_index
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
