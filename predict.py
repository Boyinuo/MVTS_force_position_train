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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz", help="Original press ADC export")
    source.add_argument("--prepared-window", help="Prepared slide/press events/*.npz (unnormalised log-mel)")
    parser.add_argument("--force-index", type=int,
                        help="Select a zero-based pair; triple-head defaults to current/last, older models to all pairs")
    parser.add_argument("--previous-contact", type=int, choices=(0, 1),
                        help="Legacy contact input; new model accepts previous PREDICTED state only (unknown by default)")
    parser.add_argument("--output", default="runs/prediction.json")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--state-in", help="Triple-head GRU state from preceding prediction")
    parser.add_argument("--state-out", help="Save triple-head GRU state for the next prediction")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, checkpoint, config = load_checkpoint(args.checkpoint, device)
    contact_aware = model.model_name == "model_contact_gated"
    triple = model.model_name in ("model_triple_head", "model_contact_gated")
    stats = checkpoint["stats"]
    with np.load(args.npz or args.prepared_window, allow_pickle=False) as data:
        if args.prepared_window:
            if not triple or args.force_index is not None:
                raise ValueError("Prepared-window requires triple-head; predicts the complete ordered window")
            prepared_audio = torch.from_numpy(data["audio"].copy())
            adc = None
        else:
            force_index = args.force_index
            if triple and force_index is None and data["adc"].ndim == 3:
                force_index = len(data["adc"]) - 1
            adc = torch.from_numpy(extract_sample_audio(data, config, force_index).astype(np.float32))
        contact = args.previous_contact
        if contact is None:
            contact = int(data["starts_in_contact"].item()) if model.config.use_contact else 0
    sequence = args.prepared_window is not None or adc.ndim == 3
    if args.prepared_window:
        audio = prepared_audio
    else:
        if not sequence:
            adc = adc.unsqueeze(0)
        audio = LogMel(config)(adc)
    audio = (audio - torch.tensor(stats["audio_mean"])[:, None, None]) / torch.tensor(
        stats["audio_std"])[:, None, None]
    model.eval()
    with torch.inference_mode():
        if triple:
            image = load_image(args.image).unsqueeze(0).to(device)
            hidden = None
            previous_state = None
            if args.state_in:
                stream = torch.load(args.state_in, map_location=device, weights_only=True)
                hidden = stream["hidden"]
                if contact_aware:
                    if stream.get("model_name") != model.model_name or "contact_state" not in stream:
                        raise ValueError("Contact-aware stream state must include hidden and predicted contact state")
                    previous_state = stream["contact_state"]
            if contact_aware and args.previous_contact is not None:
                if args.state_in:
                    raise ValueError("Use either --state-in or --previous-contact for the predicted prior state")
                previous_state = torch.tensor([args.previous_contact],device=device)
            result = model.forward_sequence(image, audio.unsqueeze(0).to(device), hidden)
            pred = torch.cat([result[k] for k in ("position", "wrench", "slip")], dim=-1)[0].cpu()
            if contact_aware:
                gated = model.gate_outputs(result,previous_state)
                pred = gated["physical_prediction"][0].cpu()
            if args.state_out:
                from pathlib import Path
                Path(args.state_out).parent.mkdir(parents=True, exist_ok=True)
                stream = {"hidden": result["hidden"].cpu(), "model_name": model.model_name}
                if contact_aware:
                    stream["contact_state"] = gated["last_contact"].cpu()
                torch.save(stream, args.state_out)
        else:
            batch_size = audio.shape[0]
            image = load_image(args.image).unsqueeze(0).expand(batch_size, -1, -1, -1)
            result = model(image.to(device), audio.to(device),
                           torch.full((batch_size,), contact, dtype=torch.long, device=device))
            pred = torch.cat((result["position"], result["wrench"]), dim=-1).cpu()
        if not contact_aware:
            pred = pred * torch.tensor(stats["target_std"]) + torch.tensor(stats["target_mean"])
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Non-finite prediction")
    names = stats.get("target_names", TARGET_NAMES)
    result = {"model_name": model.model_name,
              "checkpoint_epoch": checkpoint["epoch"], "sample_rate_hz": config.sample_rate,
              "units": "position/slip: mm; force: N; torque: N*m; acquisition sign/frame retained"}
    if sequence:
        result["predictions"] = [{"force_index": index,
                                  "prediction": dict(zip(names, row))}
                                 for index, row in enumerate(pred.tolist())]
    else:
        result["prediction"] = dict(zip(names, pred[0].tolist()))
        if args.force_index is not None or (triple and args.npz and force_index is not None):
            result["force_index"] = force_index
    if contact_aware:
        states = []
        transition_names = {-1:"unknown",0:"no_contact",1:"onset",2:"release",3:"sustained_contact"}
        for i in range(len(pred)):
            states.append({"contact_probability":float(gated["contact_probability"][0,i].cpu()),
                           "contact_state":int(gated["contact_state"][0,i].cpu()),
                           "transition":transition_names[int(gated["transition"][0,i].cpu())],
                           "position_valid":bool(gated["position_valid"][0,i].cpu()),
                           "slip_valid":bool(gated["slip_valid"][0,i].cpu())})
        if sequence:
            for item,state in zip(result["predictions"],states):
                item.update(state)
        else:
            result.update(states[0])
        result["invalid_outputs"] = "Position/slip use numeric zero placeholders when valid=false; these are not physical location/slip estimates"
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
