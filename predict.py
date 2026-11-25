"""Predict from an image and optional audio (image-only checkpoints need no NPZ)."""
import argparse
from pathlib import Path

import numpy as np
import torch

from mvts.data import TARGET_NAMES, extract_sample_audio, read_json, write_json
from mvts.engine import choose_device, load_checkpoint
from mvts.features import LogMel, load_image
from mvts.architectures import DUAL_RATE_MODELS
from mvts.dual_rate_data import event_timing_seconds


def prepared_image_time(window):
    """Recover the same image-age anchor as training from an adjacent manifest.

    Standalone windows without their manifest fall back to a fresh image anchor;
    normal prepared press/slide cache paths preserve the recorded host timestamp.
    """
    window = Path(window).resolve()
    for root in (window.parent, window.parent.parent):
        manifest = root / "manifest.json"
        if manifest.is_file():
            for record in read_json(manifest).get("records", []):
                if (root / record["file"]).resolve() == window:
                    return record.get("image_pc_ns")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    source = parser.add_mutually_exclusive_group()
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
    triple = model.model_name in DUAL_RATE_MODELS or contact_aware
    slip_guided = getattr(model, "requires_timing", False)
    timing, gate_diagnostics = {}, []
    image_only = model.model_name == "model_triple_head" and model.config.image_only
    if not (args.npz or args.prepared_window) and not image_only:
        parser.error("--npz or --prepared-window is required for checkpoints using audio")
    if args.force_index is not None and not args.npz:
        parser.error("--force-index requires --npz")
    stats = checkpoint["stats"]
    if not (args.npz or args.prepared_window):
        audio, sequence = None, False
    else:
        with np.load(args.npz or args.prepared_window, allow_pickle=False) as data:
            if args.prepared_window:
                if not triple or args.force_index is not None:
                    raise ValueError("Prepared-window requires triple-head; predicts the complete ordered window")
                prepared_audio = torch.from_numpy(data["audio"].copy())
                if slip_guided:
                    timing = event_timing_seconds(
                        data["event_pc_ns"] if "event_pc_ns" in data else None,
                        data["previous_pc_ns"] if "previous_pc_ns" in data else None,
                        prepared_image_time(args.prepared_window), len(prepared_audio),
                        independent="previous_pc_ns" not in data and len(prepared_audio) == 1)
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
                if slip_guided and stream.get("model_name") != model.model_name:
                    raise ValueError("Slip-guided stream state must come from model_slip_gated")
                if contact_aware:
                    if stream.get("model_name") != model.model_name or "contact_state" not in stream:
                        raise ValueError("Contact-aware stream state must include hidden and predicted contact state")
                    previous_state = stream["contact_state"]
            if contact_aware and args.previous_contact is not None:
                if args.state_in:
                    raise ValueError("Use either --state-in or --previous-contact for the predicted prior state")
                previous_state = torch.tensor([args.previous_contact],device=device)
            timing_tensors = {k: torch.as_tensor(v, device=device)[None] for k, v in timing.items()}
            result = model.forward_sequence(image, None if audio is None else audio.unsqueeze(0).to(device), hidden,
                                            **timing_tensors)
            pred = torch.cat([result[k] for k in ("position", "wrench", "slip")], dim=-1)[0].cpu()
            if contact_aware:
                gated = model.gate_outputs(result,previous_state)
                pred = gated["physical_prediction"][0].cpu()
            if slip_guided:
                for i in range(len(pred)):
                    gate_diagnostics.append({
                        "visual_weight": float(result["visual_weight"][0, i].cpu()),
                        "audio_weight": float(result["audio_weight"][0, i].cpu()),
                        "slip_prior_mm": result["slip_prior_mm"][0, i].cpu().tolist(),
                        "prior_speed_mm_s": float(result["prior_speed_mm_s"][0, i].cpu()),
                        "delta_t_s": float(result["delta_t"][0, i].cpu()),
                        "image_age_s": float(result["image_age"][0, i].cpu())})
            if args.state_out:
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
    if slip_guided:
        if sequence:
            for item, diagnostics in zip(result["predictions"], gate_diagnostics):
                item.update(diagnostics)
        else:
            result.update(gate_diagnostics[0])
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
