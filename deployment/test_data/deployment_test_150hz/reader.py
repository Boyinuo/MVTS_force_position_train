"""Portable event inputs: NumPy + Pillow only, no training project imports."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def load_events(root=None, kind="all", normalization=None):
    """Preload before starting a 150 Hz clock. One list element = one call.

    audio_input: contiguous float32 [1,6,64,28], already normalized.
    image_input: contiguous float32 [1,3,224,224] on image updates, else None.
    adc: uint16 [4000,6], unscaled raw ADC, channels in acquisition order.
    target: float64 [11], physical units, for comparison only.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    stats = json.loads(Path(normalization).read_text(encoding="utf-8")) if normalization else None
    events, images = [], {}
    for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if kind != "all" and row["kind"] != kind:
            continue
        with np.load(root / row["data"], allow_pickle=False) as z:
            audio = z["audio_input"].copy() if stats is None else (
                (z["audio"].astype(np.float64) - np.asarray(stats["audio_mean"])[:, None, None])
                / np.asarray(stats["audio_std"])[:, None, None]).astype(np.float32)
            adc, target = z["adc"].copy(), z["target"].copy()
        image = None
        if row["image_update"]:
            if row["image"] not in images:
                with Image.open(root / row["image"]) as im:
                    rgb = np.array(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
                # Match the original PyTorch image preprocessing arithmetic.
                chw = rgb.transpose(2, 0, 1) / np.float32(255)
                chw = (chw - np.asarray([.485, .456, .406], dtype=np.float32)[:, None, None]) / np.asarray([.229, .224, .225], dtype=np.float32)[:, None, None]
                images[row["image"]] = np.ascontiguousarray(chw[None])
            image = images[row["image"]]
        events.append({"meta": row, "image_input": image,
                       "audio_input": np.ascontiguousarray(audio[None]), "adc": adc, "target": target})
    return events


def replay(events, callback, realtime=False):
    """Invoke callback once per event. No events dropped if inference is slow.

    realtime=True targets 150 Hz using absolute deadlines; actual throughput
    depends on inference/runtime/OS. Default runs as fast as callback allows.
    """
    if not events:
        return
    base = events[0]["meta"]["schedule_offset_ns"]
    start = time.perf_counter_ns()
    for event in events:
        if realtime:
            deadline = start + event["meta"]["schedule_offset_ns"] - base
            remaining = deadline - time.perf_counter_ns()
            if remaining > 0:
                time.sleep(remaining / 1e9)
        callback(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--kind", choices=("all", "press", "slide"), default="all")
    parser.add_argument("--normalization", type=Path, help="Override audio normalization for another checkpoint")
    parser.add_argument("--realtime", action="store_true", help="Schedule calls at nominal 150 Hz")
    args = parser.parse_args()
    events = load_events(args.root, args.kind, args.normalization)
    counts = {"calls": 0, "image_updates": 0, "state_resets": 0}

    def check(event):
        assert event["audio_input"].shape == (1, 6, 64, 28)
        assert event["audio_input"].dtype == np.float32
        assert event["adc"].shape == (4000, 6)
        assert event["target"].shape == (11,)
        assert np.isfinite(event["audio_input"]).all()
        if event["image_input"] is not None:
            assert event["image_input"].shape == (1, 3, 224, 224)
        counts["calls"] += 1
        counts["image_updates"] += int(event["meta"]["image_update"])
        counts["state_resets"] += int(event["meta"]["reset_state"])

    start = time.perf_counter()
    replay(events, check, args.realtime)
    print(json.dumps({**counts, "realtime": args.realtime, "elapsed_seconds": time.perf_counter()-start}, indent=2))


if __name__ == "__main__":
    main()
