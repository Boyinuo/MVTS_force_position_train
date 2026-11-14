import json

import numpy as np
from PIL import Image
import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def torch_threads():
    torch.set_num_threads(2)


@pytest.fixture
def acquisition_data(tmp_path):
    rng = np.random.default_rng(7)
    for session_index in range(2):
        session = tmp_path / f"session_{session_index}"
        (session / "samples").mkdir(parents=True)
        (session / "frames").mkdir()
        (session / "audio").mkdir()
        (session / "audio/session.json").write_text(json.dumps({
            "sample_rate_observed_hz": 50000, "valid_for_time_matching": False,
            "timing_quality_passed": True}), encoding="utf-8")
        (session / "session.json").write_text(json.dumps({"completed": False}), encoding="utf-8")
        rows = []
        for point in range(8):
            image = f"frames/{point}.png"
            npz = f"samples/{point}.npz"
            Image.fromarray(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)).save(session / image)
            xy = [float(point) + 0.001 * session_index, float(-point)]
            wrench = np.arange(6, dtype=np.float64) * 0.01 + point
            marker = 1_000_000_000 + point * 1_000_000
            record = {"valid": True, "kind": "target", "mode": "staircase" if not session_index else "dynamic",
                      "point": point, "sample_id": point, "image": image, "data": npz,
                      "x_mm": point, "y_mm": -point, "actual_xy_mm": xy,
                      "actual_depth_mm": point * 0.1 + 0.5, "starts_in_contact": session_index,
                      "force": {"wrench": wrench.tolist()}, "pc_ns": marker,
                      "camera_age_ms": 20, "force_age_ms": 10}
            np.savez(session / npz, adc=rng.integers(1000, 2000, (2000, 6), dtype=np.uint16),
                     device_us=np.arange(2000, dtype=np.uint64) * 20 + point * 50000,
                     estimated_pc_ns=marker + (np.arange(2000) - 1000) * 20000.0,
                     force_wrench=wrench, actual_xy_mm=xy, starts_in_contact=np.int8(session_index),
                     marker_pc_ns=np.int64(marker))
            rows.append(json.dumps(record))
        (session / "samples.jsonl").write_text("\n".join(rows), encoding="utf-8")
    return tmp_path
