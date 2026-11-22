"""Prepare a reusable press/slide cache without changing raw data."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from mvts.dual_rate_data import prepare_data, subset_prepared
from mvts.data import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/data_final")
    p.add_argument("--output", help="New cache directory; contact default: .cache/data_final_v5_faster_contact")
    p.add_argument("--contact-aware", action="store_true", help="Create v5 contact labels, include press backgrounds and regression-validity masks")
    p.add_argument("--selection-from-cache", help="Reprocess ORIGINAL raw data using the source selection of this v4 cache (does not copy its features)")
    p.add_argument("--from-cache", help="Create a subset by copying an existing prepared cache")
    p.add_argument("--press-source", action="append",
                   help="With --from-cache: press session to retain; repeat for multiple sessions; all slide records retained")
    p.add_argument("--slide-source", action="append",
                   help="Raw preprocessing: keep only this slide dataset folder; repeat for multiple; press remains all sources")
    p.add_argument("--max-units-per-source", type=int, help="Smoke only: press locations / slide trials per source")
    p.add_argument("--skip-unsupported", action="store_true", help="Report and exclude old slide exports without wrench supervision")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    a.output = a.output or (".cache/data_final_v5_faster_contact" if a.contact_aware else ".cache/data_final_v4")
    if a.threads < 1 or (a.max_units_per_source is not None and a.max_units_per_source < 1):
        p.error("threads and max-units-per-source must be positive")
    torch.set_num_threads(a.threads)
    selection_reference, press_sources = None, None
    if a.selection_from_cache:
        if not a.contact_aware or a.from_cache or a.slide_source or a.press_source:
            p.error("--selection-from-cache requires --contact-aware and cannot be combined with source filters or --from-cache")
        baseline = Path(a.selection_from_cache).resolve()
        raw_manifest = (baseline/"manifest.json").read_bytes()
        manifest = json.loads(raw_manifest)
        report = json.loads((baseline/"report.json").read_text(encoding="utf-8"))
        if manifest["version"] != 4:
            p.error("--selection-from-cache expects a v4 baseline cache")
        a.data_root = report["source_root"]
        press_sources = sorted({r["id"].split("/",1)[0] for r in manifest["records"] if r["kind"] == "press"})
        a.slide_source = sorted({r["id"].split("/",1)[0] for r in manifest["records"] if r["kind"] == "slide"})
        selection_reference = {"cache": str(baseline), "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                               "mode": "raw reprocessing; baseline used only to select sources"}
    if a.from_cache:
        if a.contact_aware:
            p.error("Contact labels/backgrounds require raw preprocessing, not --from-cache")
        if a.slide_source:
            p.error("--slide-source is supported only for raw preprocessing, without --from-cache")
        if not a.press_source or a.max_units_per_source is not None:
            p.error("--from-cache requires --press-source and cannot use --max-units-per-source")
        _, report = subset_prepared(a.from_cache, a.output, a.press_source)
    else:
        if a.press_source:
            p.error("--press-source requires --from-cache")
        _, report = prepare_data(a.data_root, a.output, max_units_per_source=a.max_units_per_source,
                                skip_unsupported=a.skip_unsupported, slide_sources=a.slide_source,
                                contact_aware=a.contact_aware, press_sources=press_sources)
        if selection_reference:
            report["selection_reference"] = selection_reference
            write_json(Path(a.output)/"report.json", report)
    print({k: report[k] for k in ("counts", "windows", "events", "smoke_test")})


if __name__ == "__main__":
    main()
