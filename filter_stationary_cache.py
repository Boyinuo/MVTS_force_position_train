"""Copy a v4/v5 cache, trimming each slide at its first near-stationary XY window."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from mvts.contact_data import (contact_summary, load_contact_arrays,
                               read_contact_prepared, supervision_digest)
from mvts.data import _safe_file, write_json
from mvts.dual_rate_data import read_prepared
from mvts.prepared_selection import select_records, xy_diameter_mm


def contact_history(arrays, continuous):
    """Unknown history after a removed window; preserve within-window contact."""
    result = dict(arrays)
    result['previous_contact'] = arrays['previous_contact'].copy()
    result['transition_valid'] = arrays['transition_valid'].copy()
    result['previous_contact'][0] = 1 if continuous else -1
    result['transition_valid'][0] = continuous
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise OSError(f'Copy checksum mismatch: {destination}')


def filter_cache(source, output, threshold_mm=.05, baseline_cache=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output must be a new directory separate from the source cache')
    manifest_path = source / 'manifest.json'
    source_digest = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    version = manifest['version']
    if version == 5:
        read_contact_prepared(source)
    elif version == 4:
        read_prepared(source)
    else:
        raise ValueError('Only v4/v5 prepared caches are supported')
    retained, removed = select_records(manifest['records'], threshold_mm)
    if not retained or len({r['id'] for r in manifest['records']}) != len(manifest['records']):
        raise ValueError('Empty output or duplicate source IDs')
    # Resolve every retained path before creating any output.
    files = {relative: _safe_file(source, relative)
             for r in retained for relative in (r['file'], r['image'])}
    if len({r['file'] for r in retained}) != len(retained):
        raise ValueError('Event files must be unique per record')
    baseline_reference = None
    if baseline_cache is not None:
        if version != 5:
            raise ValueError('--baseline-cache is only valid for a v5 contact cache')
        baseline = Path(baseline_cache).resolve()
        baseline_records, _ = read_prepared(baseline)
        baseline_ids = {r['id'] for r in baseline_records}
        expected_ids = {r['id'] for r in retained if r.get('contact_state') == 1}
        if baseline_ids != expected_ids:
            raise ValueError('Filtered v4 baseline must exactly match contact-positive v5 IDs')
        baseline_reference = {'cache': str(baseline),
                              'manifest_sha256': sha256(baseline / 'manifest.json'),
                              'mode': 'matching filtered v4 selection; features copied from original v5 cache'}
    output.mkdir(parents=True)
    previous, patched = {}, []
    for number, record in enumerate(retained, 1):
        source_file, target_file = files[record['file']], output / record['file']
        target_file.parent.mkdir(parents=True, exist_ok=True)
        changed = False
        if version == 5 and record['kind'] == 'slide':
            continuous = previous.get(record['sequence'], -2) + 1 == record['order']
            with np.load(source_file, allow_pickle=False) as z:
                original = {key: z[key] for key in z.files}
            if supervision_digest(original) != record['supervision_sha256']:
                raise ValueError(f"Invalid source contact digest: {record['id']}")
            arrays = contact_history(original, continuous)
            changed = any(not np.array_equal(arrays[key], original[key])
                          for key in ('previous_contact', 'transition_valid'))
            if changed:
                np.savez(target_file, **arrays)
                with np.load(target_file, allow_pickle=False) as z:
                    assert set(z.files) == set(original)
                    for key in original:
                        assert np.array_equal(z[key], arrays[key]), (record['id'], key)
                record['supervision_sha256'] = supervision_digest(arrays)
                patched.append(record['id'])
            previous[record['sequence']] = record['order']
        if not changed:
            copy_verified(source_file, target_file)
        if number % 2000 == 0 or number == len(retained):
            print(f'{source.name}: copied {number}/{len(retained)} event files', flush=True)
    images = sorted({r['image'] for r in retained})
    for number, relative in enumerate(images, 1):
        copy_verified(files[relative], output / relative)
        if number % 2000 == 0 or number == len(images):
            print(f'{source.name}: copied {number}/{len(images)} images', flush=True)
    with (output / 'filtered_out.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=('id', 'sequence', 'order', 'events',
                                                   'xy_diameter_mm', 'file', 'image',
                                                   'cutoff_order', 'reason'))
        writer.writeheader()
        writer.writerows(removed)
    filtering = {'source_cache': str(source), 'source_manifest_sha256': source_digest,
                 'threshold_mm': threshold_mm, 'comparison': 'strictly less than triggers cutoff; equality alone does not trigger cutoff',
                 'metric': 'maximum pairwise Euclidean XY distance within each slide window; Z ignored',
                 'mode': 'trim each trajectory at first qualifying window, removing that window and ALL later windows regardless of their diameter',
                 'scope': 'slide windows only; all press events retained',
                 'removed_windows': len(removed), 'removed_events': sum(r['events'] for r in removed),
                 'removed_records_file': 'filtered_out.csv',
                 'source_windows': len(manifest['records']),
                 'source_events': sum(r['events'] for r in manifest['records']),
                 'contact_boundary_repairs': len(patched), 'repaired_record_ids': patched,
                 'continuity': 'Original sequence, order, group and timestamps retained; existing loaders reset GRU on order gaps. Contact history is unknown at each new retained segment.',
                 'features': 'Physical copies; original audio, images and regression labels preserved, including original causal audio history and measured slip increments.'}
    report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    if baseline_reference:
        report['source_selection_reference'] = report.get('selection_reference')
        report['selection_reference'] = baseline_reference
    report.update(filtering=filtering, windows=len(retained),
                  events=sum(r['events'] for r in retained),
                  counts=dict(Counter(r['kind'] for r in retained)),
                  copied_files=len(files), copied_bytes=sum((output / p).stat().st_size for p in files))
    if version == 5:
        report['contact_counts'] = contact_summary(retained)
    write_json(output / 'report.json', report)
    write_json(output / 'manifest.json', {**manifest, 'filtering': filtering, 'records': retained})
    # Validate the complete derived dataset with its normal training readers.
    loaded, _ = read_contact_prepared(output) if version == 5 else read_prepared(output)
    for number, r in enumerate(loaded, 1):
        if version == 5:
            arrays = load_contact_arrays(r)
        else:
            with np.load(output / r['file'], allow_pickle=False) as z:
                arrays = {key: z[key] for key in ('audio', 'target')}
        assert np.array_equal(arrays['target'], np.asarray(r['target']))
        assert arrays['audio'].shape == (r['events'], 6, 64, 28)
        assert np.isfinite(arrays['audio']).all()
        assert r['kind'] != 'slide' or xy_diameter_mm(r) >= threshold_mm
        if number % 4000 == 0 or number == len(loaded):
            print(f'{source.name}: validated {number}/{len(loaded)} output event files', flush=True)
    assert sha256(manifest_path) == source_digest
    source_press = {r['id'] for r in manifest['records'] if r['kind'] == 'press'}
    assert source_press == {r['id'] for r in retained if r['kind'] == 'press'}
    source_slides, retained_slides = {}, {}
    for r in manifest['records']:
        if r['kind'] == 'slide':
            source_slides.setdefault(r['sequence'], []).append(r)
    for r in retained:
        if r['kind'] == 'slide':
            retained_slides.setdefault(r['sequence'], []).append(r)
    for sequence, original in source_slides.items():
        original = sorted(original, key=lambda r: r['order'])
        selected = sorted(retained_slides.get(sequence, []), key=lambda r: r['order'])
        assert [r['id'] for r in selected] == [r['id'] for r in original[:len(selected)]]
        assert all(b['order'] == a['order'] + 1 for a, b in zip(selected, selected[1:]))
    verification = {'passed': True, 'version': version, 'output': str(output),
                    'windows': len(retained), 'events': report['events'], 'counts': report['counts'],
                    'removed_windows': len(removed), 'removed_events': filtering['removed_events'],
                    'all_press_preserved': True, 'all_retained_slides_meet_threshold': True,
                    'all_retained_slide_sequences_are_contiguous_prefixes': True,
                    'source_slide_trajectories': len(source_slides),
                    'retained_slide_trajectories': len(retained_slides),
                    'all_unchanged_files_checksum_verified': True,
                    'all_output_event_files_loaded_and_validated': True,
                    'source_manifest_unchanged': True, 'contact_boundary_repairs': len(patched)}
    write_json(output / 'filter_verification.json', verification)
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold-mm', type=float, default=.05)
    parser.add_argument('--baseline-cache', help='Matching filtered v4 cache for a v5 output')
    args = parser.parse_args()
    print(json.dumps(filter_cache(args.source, args.output, args.threshold_mm, args.baseline_cache),
                     ensure_ascii=False, indent=2), flush=True)
