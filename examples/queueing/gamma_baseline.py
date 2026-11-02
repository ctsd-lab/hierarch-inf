"""Short deterministic aggregate baseline; no queueing inference or certificates."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from hierarch_inf.queueing.gamma import load_gamma_pair, read_phase_log, phase_labels, file_sha256


def fit_affine(loads, means_seconds, counts):
    """Nonnegative WLS: m(lambda) = a + b lambda/1000, using training rows only."""
    x, y, w = (np.asarray(v, dtype=float) for v in (loads, means_seconds, counts))
    if (x.ndim != 1 or x.shape != y.shape or x.shape != w.shape or len(x) < 2
            or np.any(~np.isfinite(x + y + w)) or np.any(x < 0) or np.any(y < 0)
            or np.any(w <= 0) or len(np.unique(x)) < 2):
        raise ValueError('Need finite paired training observations at two distinct loads and positive counts')
    design = np.column_stack([np.ones(len(x)), x / 1000])
    fit = lsq_linear(design * np.sqrt(w[:, None]), y * np.sqrt(w), bounds=(0, np.inf))
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def predict_affine(parameters, loads):
    return parameters[0] + parameters[1] * np.asarray(loads) / 1000


def write_rows(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def prepare(data_dir, output_dir, manifest_path):
    """Create compact derived observations from the pinned actual GAMMA subset."""
    manifest = json.loads(Path(manifest_path).read_text())
    root, out = Path(data_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    verified = set()
    for entry in manifest['files']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Manifest path escape')
        if path.stat().st_size != entry['uncompressed_bytes'] or file_sha256(path) != entry['sha256']:
            raise ValueError(f'Source mismatch: {path}')
        verified.add(path)
    def source(path):
        if path.resolve() not in verified:
            raise ValueError(f'Unverified source: {path}')
        return path
    observations, distributions, graphs, audit = [], [], [], []
    edges_ms = np.geomspace(.01, 10000, 161)
    for spec in manifest['runs']:
        directory = root / spec['directory']
        args = json.loads(source(directory / 'args.txt').read_text())
        load = args['rps'][0]
        phases = [p for host in args['bottlenecked_nodes']
                  for p in read_phase_log(source(directory / (host + '_phases')), host=host)]
        request, graph = spec['request_type'], spec['graph_id']
        batch = load_gamma_pair(source(directory / f'{request}_rps_start_time_{graph}.csv'),
                                source(directory / f'{request}_service_time_{graph}.csv'),
                                source(directory / f'{request}_ids_{graph}'),
                                source(directory / f'{request}_graph_paths_{graph}'),
                                timestamp_unit=manifest['timestamp_unit'], duration_unit=manifest['duration_unit'])
        first = min(p.start for p in phases)
        end, start = first - 5, first - 245
        if start < batch.root_start_seconds.min() or end > batch.root_start_seconds.max():
            raise ValueError('Requested 240-second baseline not fully observed')
        retained_count, nonempty_count = 0, 0
        for k in range(24):
            left, right = start + k * 10, start + (k + 1) * 10
            keep = ((batch.root_start_seconds >= left)
                    & (batch.root_start_seconds + batch.root_duration_seconds < right))
            y = batch.root_duration_seconds[keep]
            if not len(y):
                continue
            retained_count += len(y)
            nonempty_count += 1
            observations.append(dict(run_id=spec['run_id'], offered_rps=load, block=k,
                relative_start_seconds=left-first, requests=len(y), mean_response_seconds=float(y.mean()),
                std_response_seconds=float(y.std(ddof=1)) if len(y)>1 else 0,
                split='heldout_workload' if load == 400 else ('train' if k < 12 else 'heldout_time')))
        labels = phase_labels(batch.root_start_seconds, batch.root_duration_seconds, phases, guard_seconds=5)
        labels = np.asarray(labels)
        for regime, keep in [('no_logged_interference', labels == 'no_logged_interference'),
                             ('logged_interference', np.char.startswith(labels, 'interference:'))]:
            values = batch.root_duration_seconds[keep] * 1000
            hist, _ = np.histogram(values, bins=edges_ms)
            for i, count in enumerate(hist):
                distributions.append(dict(offered_rps=load, regime=regime, left_ms=edges_ms[i],
                    right_ms=edges_ms[i+1], count=int(count), total_requests=len(values),
                    below_range=int((values < edges_ms[0]).sum()), above_range=int((values > edges_ms[-1]).sum())))
        graphs.append(dict(run_id=spec['run_id'], nodes=batch.node_ids, edges=batch.edges))
        audit.append(dict(run_id=spec['run_id'], quality=batch.quality,
                          first_injection_epoch_seconds=first, fitting_window_seconds=[-245,-5],
                          nonempty_windows=nonempty_count, empty_windows=24-nonempty_count,
                          retained_requests=retained_count,
                          edge_crossing_requests=int(((batch.root_start_seconds >= start) & (batch.root_start_seconds < end)).sum())-retained_count))
    write_rows(out/'baseline_windows.csv', observations)
    write_rows(out/'response_histograms.csv', distributions)
    (out/'graphs.json').write_text(json.dumps(graphs, indent=2)+'\n')
    (out/'baseline_provenance.json').write_text(json.dumps(dict(
        source_manifest_sha256=file_sha256(manifest_path), runs=audit,
        fit_rule='200 and800 RPS: first120 seconds train; last120 seconds temporal holdout;400 RPS fully held out',
        observation='Ten-second request-weighted root-span means; edge-crossing requests excluded',
        histogram_scope='All observed run requests by logged regime; descriptive only, not fitting input'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--manifest', default='examples/queueing/gamma/sample_manifest.json')
    args = parser.parse_args()
    prepare(args.data_dir, args.output_dir, args.manifest)
