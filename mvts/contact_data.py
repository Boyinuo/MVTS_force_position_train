"""Contact supervision for the agreed controlled press/slide acquisition.

Background press events are unloaded; target/limit press events and EVERY slide
subevent are contact-positive. starts_in_contact is deliberately never used.
Sparse press events are independent; no synthetic contact transitions are made.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import read_json
from .dual_rate_data import TARGET_NAMES, feature_config
from .features import load_image

CONTACT_CACHE_VERSION = 5
CONTACT_SCHEMA = "contact_gated_v1"
DEFAULT_CONTACT_CACHE = ".cache/data_final_v5_faster_contact"
SUPERVISION_KEYS = ("contact", "contact_valid", "previous_contact", "transition_valid", "target_valid")
CONTACT_POLICY = {
    "press_background": 0, "press_target": 1, "press_limit": 1,
    "slide_all_subevents": 1,
    "starts_in_contact": "not used: acquisition-action start flag, not per-event contact",
    "press_history": "independent events; previous contact unknown",
    "slide_history": "unknown at first retained event/gap; subsequent contact is 1",
    "regression": "background regression targets retained as measurements but masked; zero-wrench loss supervises background",
    "slip": "press contact has known zero tangential motion; slide uses measured within-trajectory XY increments",
}


def supervision_digest(arrays):
    digest = hashlib.sha256()
    for key in SUPERVISION_KEYS:
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def add_contact_supervision(arrays, kind, event_kind=None, previous_available=False):
    n = len(arrays["target"])
    if kind not in ("press", "slide") or (kind == "press" and event_kind not in ("background", "target", "limit")):
        raise ValueError("Unsupported contact label source")
    value = int(kind == "slide" or event_kind != "background")
    contact = np.full(n, value, dtype=np.int8)
    previous = np.full(n, value, dtype=np.int8)
    if not previous_available:
        previous[0] = -1
    arrays = {**arrays, "contact": contact, "contact_valid": np.ones(n, dtype=bool),
              "previous_contact": previous, "transition_valid": previous >= 0,
              "target_valid": np.broadcast_to(contact[:, None] == 1, (n, 11)).copy()}
    source = "slide_sustained" if kind == "slide" else f"press_{event_kind}"
    return arrays, {"contact_source": source, "contact_state": value,
                    "supervision_sha256": supervision_digest(arrays)}


def read_contact_prepared(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if (manifest.get("version") != CONTACT_CACHE_VERSION or manifest.get("schema") != CONTACT_SCHEMA
            or manifest.get("features") != feature_config().to_dict()
            or manifest.get("contact_policy") != CONTACT_POLICY):
        raise ValueError("Contact model requires a v5 contact cache; run preprocess.py --contact-aware")
    records = [{**r, "cache_root": str(root)} for r in manifest["records"]]
    if not records or any(r.get("contact_state") not in (0, 1) or not r.get("supervision_sha256") for r in records):
        raise ValueError("Missing contact supervision in manifest")
    return records, manifest


def load_contact_arrays(record):
    with np.load(Path(record["cache_root"]) / record["file"], allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in ("audio", "target", *SUPERVISION_KEYS)}
    n = len(arrays["target"])
    if (arrays["target"].shape != (n, 11) or arrays["audio"].shape != (n, 6, 64, 28)
            or arrays["target_valid"].shape != (n, 11)
            or any(arrays[k].shape != (n,) for k in SUPERVISION_KEYS[:-1])
            or not np.isin(arrays["contact"], (0, 1)).all()
            or not np.isin(arrays["previous_contact"], (-1, 0, 1)).all()
            or not np.isfinite(arrays["audio"]).all() or not np.isfinite(arrays["target"]).all()):
        raise ValueError(f"Invalid contact arrays: {record['id']}")
    if supervision_digest(arrays) != record["supervision_sha256"]:
        raise ValueError(f"Contact supervision changed: {record['id']}")
    return arrays


def fit_contact_statistics(records):
    count, audio_count = np.zeros(11, dtype=np.int64), 0
    total, square = np.zeros(11), np.zeros(11)
    audio_total, audio_square = np.zeros(6), np.zeros(6)
    classes = np.zeros(2, dtype=np.int64)
    class_weight_counts = np.zeros(2, dtype=np.float64)
    event_count = 0
    for number, r in enumerate(records, 1):
        a = load_contact_arrays(r)
        y, mel = a["target"].astype(np.float64), a["audio"].astype(np.float64)
        valid = a["target_valid"]
        total += (y * valid).sum(0)
        square += (y*y * valid).sum(0)
        count += valid.sum(0)
        audio_total += mel.sum((0, 2, 3))
        audio_square += (mel*mel).sum((0, 2, 3))
        audio_count += mel.shape[0] * mel.shape[2] * mel.shape[3]
        window_classes = np.bincount(a["contact"][a["contact_valid"]], minlength=2)
        classes += window_classes
        # SGD averages each length-homogeneous batch. A ten-step slide window
        # therefore contributes one window mean, not ten press-sized updates.
        # Balance that sampled objective rather than over-weighting backgrounds
        # by the event-count ratio of 1-step press and 10-step slide batches.
        class_weight_counts += window_classes/max(window_classes.sum(),1)
        event_count += len(y)
        if number % 2000 == 0:
            print(f"Contact normalization: {number}/{len(records)} windows", flush=True)
    if (count == 0).any() or (classes == 0).any():
        raise ValueError("Training split needs valid regression labels AND both contact/background classes")
    mean, am = total/count, audio_total/audio_count
    std = np.sqrt(np.maximum(square/count-mean*mean, 1e-8))
    std[9:] = np.maximum(std[9:], .01)
    return {"target_mean": mean.tolist(), "target_std": std.tolist(),
            "audio_mean": am.tolist(),
            "audio_std": np.sqrt(np.maximum(audio_square/audio_count-am*am, 1e-8)).tolist(),
            "target_fit_counts": count.tolist(), "fit_sample_count": event_count,
            "contact_class_counts": classes.tolist(),
            "contact_class_weight_counts": class_weight_counts.tolist(),
            "contact_class_weights": (class_weight_counts.sum()/(2*class_weight_counts)).tolist(),
            "target_names": TARGET_NAMES}


class ContactPreparedDataset(Dataset):
    def __init__(self, records, stats):
        self.records, self.stats = records, stats

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r, s = self.records[index], self.stats
        a = load_contact_arrays(r)
        return {"image": load_image(Path(r["cache_root"]) / r["image"]),
                "audio": torch.tensor((a["audio"]-np.array(s["audio_mean"])[None,:,None,None]) /
                                      np.array(s["audio_std"])[None,:,None,None], dtype=torch.float32),
                "target": torch.tensor((a["target"]-np.array(s["target_mean"]))/np.array(s["target_std"]), dtype=torch.float32),
                **{key: torch.from_numpy(a[key]) for key in SUPERVISION_KEYS},
                "id": r["id"], "sequence": r["sequence"], "order": r["order"]}


def contact_summary(records):
    counts = Counter()
    for r in records:
        counts[str(r["contact_state"])] += r["events"]
    return {"no_contact": counts["0"], "contact": counts["1"]}
