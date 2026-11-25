"""Audit a complete contact cache, optionally comparing an original v4 baseline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mvts.contact_data import (contact_summary, load_contact_arrays, read_contact_prepared)
from mvts.data import write_json
from mvts.dual_rate_data import read_prepared


def verify(root, baseline=None, compare_audio=False):
    if compare_audio and baseline is None:
        raise ValueError("--compare-audio requires --baseline")
    records,manifest = read_contact_prepared(root)
    original = read_prepared(baseline)[0] if baseline is not None else []
    by_id = {r["id"]:r for r in original}
    new_ids = {r["id"] for r in records}
    if len(new_ids) != len(records) or not set(by_id) <= new_ids:
        raise ValueError("Duplicate new IDs or missing baseline records")
    counts,previous,events = Counter(),{},0
    checked_images = set()
    max_audio_difference = 0.
    for number,r in enumerate(records,1):
        a = load_contact_arrays(r)
        contact = r["contact_state"]
        image = Path(root)/r["image"]
        if r["image"] not in checked_images:
            with Image.open(image) as im:
                assert im.size == (224,224) and im.mode == "RGB"
                im.verify()
            checked_images.add(r["image"])
        assert np.all(a["contact"] == contact) and a["contact_valid"].all()
        assert np.all(a["target_valid"] == bool(contact))
        assert np.array_equal(a["target"],np.asarray(r["target"]))
        assert np.array_equal(a["transition_valid"],a["previous_contact"] >= 0)
        if r["kind"] == "slide":
            assert contact == 1 and len(a["contact"]) == 10
            continuous = previous.get(r["sequence"],-2)+1 == r["order"]
            assert a["previous_contact"][0] == (1 if continuous else -1)
            assert np.all(a["previous_contact"][1:] == 1)
            previous[r["sequence"]] = r["order"]
        else:
            assert len(a["contact"]) == 1 and a["previous_contact"].item() == -1
        with np.load(Path(root)/r["file"],allow_pickle=False) as z:
            assert len(z["event_pc_ns"]) == r["events"]
            assert np.all(np.diff(z["event_pc_ns"]) > 0)
            assert np.all(z["audio_last_pc_ns"] < z["event_pc_ns"])
            assert np.all(z["force_pc_ns"] <= z["event_pc_ns"])
            if r["kind"] == "slide":
                assert np.all(z["force_age_ns"] <= 3_000_000)
        if r["id"] in by_id:
            old = by_id[r["id"]]
            assert contact == 1 and old["target"] == r["target"]
            assert old["sequence"] == r["sequence"] and old["order"] == r["order"] and old["group"] == r["group"]
            if compare_audio:
                with np.load(Path(baseline)/old["file"],allow_pickle=False) as z:
                    np.testing.assert_allclose(a["audio"],z["audio"],rtol=1e-6,atol=1e-5)
                    max_audio_difference = max(max_audio_difference,float(np.max(np.abs(a["audio"]-z["audio"]))))
        elif baseline is not None:
            assert r["kind"] == "press" and r["contact_source"] == "press_background" and contact == 0
        if r["kind"] == "press":
            assert r["contact_source"] in ("press_background", "press_target", "press_limit")
            assert contact == int(r["contact_source"] != "press_background")
        else:
            assert r["contact_source"] == "slide_sustained"
        counts[r["kind"]] += 1
        events += r["events"]
        if number % 2000 == 0:
            print(f"Verified {number}/{len(records)} windows",flush=True)
    report = json.loads((Path(root)/"report.json").read_text(encoding="utf-8"))
    assert not report["rejected"] and report["events"] == events and report["windows"] == len(records)
    assert report["contact_counts"] == contact_summary(records)
    result = {"passed":True,"version":manifest["version"],"windows":len(records),"events":events,
              "counts":dict(counts),"contact_counts":contact_summary(records),
              "images_verified":len(checked_images),
              "all_causal_times_and_contact_masks_verified":True,
              "audio_compared":compare_audio,"max_audio_absolute_difference":max_audio_difference if compare_audio else None,
              "transition_note":"No onset/release labels fabricated from sparse press events or slide sequence boundaries"}
    if baseline is not None:
        digest = hashlib.sha256((Path(baseline)/"manifest.json").read_bytes()).hexdigest()
        assert report["selection_reference"]["manifest_sha256"] == digest
        result.update(baseline_windows_preserved=len(original), added_background_windows=len(records)-len(original),
                      baseline_manifest_sha256_unchanged=digest, all_original_regression_targets_preserved=True)
    write_json(Path(root)/"verification.json",result)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root",default=".cache/data_final_v5_faster_contact")
    p.add_argument("--baseline",help="Optional original v4 cache for regression/audio comparison")
    p.add_argument("--compare-audio",action="store_true")
    a=p.parse_args()
    print(json.dumps(verify(a.prepared_root,a.baseline,a.compare_audio),ensure_ascii=False),flush=True)
