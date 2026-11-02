"""Download a bounded GAMMA subset and prepare phase-aligned response curves.

Run from the repository root with ``python -m examples.queueing.gamma_study``.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile

from hierarch_inf.queueing.gamma import (
    file_sha256, load_gamma_pair, load_run_arguments, read_phase_log, summarize_batch,
)


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("manifest paths must stay inside the data directory")
    return path


def download_files(manifest, root, *, max_total_bytes):
    files = manifest["files"]
    if sum(f["uncompressed_bytes"] for f in files) > max_total_bytes:
        raise ValueError("manifest exceeds the selected total download budget")
    for entry in files:
        target = safe_path(root, entry["path"])
        if target.exists():
            if file_sha256(target) != entry["sha256"]:
                raise ValueError(f"existing file checksum mismatch: {target}")
            continue
        parsed = urlparse(entry["source_url"])
        if (parsed.scheme != "https" or parsed.netloc != "www.kaggle.com"
                or not parsed.path.startswith("/api/v1/datasets/download/gagansomashekar/"
                                             "microservices-bottleneck-detection-dataset/")):
            raise ValueError("sample manifest must use the official GAMMA download endpoint")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            downloaded = Path(temporary) / "download"
            limit = max(entry["uncompressed_bytes"] * 2, 65536)
            with urlopen(entry["source_url"], timeout=90) as response, downloaded.open("wb") as stream:
                count = 0
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > limit:
                        raise ValueError("download exceeds per-file byte budget")
                    stream.write(block)
            decoded = Path(temporary) / "decoded"
            if zipfile.is_zipfile(downloaded):
                with zipfile.ZipFile(downloaded) as archive:
                    members = archive.infolist()
                    if len(members) != 1 or Path(members[0].filename).name != target.name:
                        raise ValueError("expected a single-file GAMMA download archive")
                    if members[0].file_size != entry["uncompressed_bytes"]:
                        raise ValueError("archive member size does not match manifest")
                    decoded.write_bytes(archive.read(members[0]))
            else:
                downloaded.rename(decoded)
            if decoded.stat().st_size != entry["uncompressed_bytes"] or file_sha256(decoded) != entry["sha256"]:
                raise ValueError(f"source checksum or length mismatch: {entry['path']}")
            decoded.replace(target)


def write_csv(path, rows):
    if not rows:
        raise ValueError("no rows to write")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(manifest, root, output, *, guard_seconds, window_seconds):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        path = safe_path(root, entry["path"])
        if path.stat().st_size != entry["uncompressed_bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"source checksum mismatch: {path}")
    windows, audits = [], []
    for spec in manifest["runs"]:
        directory = safe_path(root, spec["directory"])
        request, graph = spec["request_type"], spec["graph_id"]
        args = load_run_arguments(directory / "args.txt")
        hosts = args.get("bottlenecked_nodes")
        if not isinstance(hosts, list) or not hosts:
            raise ValueError("selected interference runs must list their injected hosts")
        phases = [p for host in hosts for p in read_phase_log(directory / (host + "_phases"), host=host)]
        if any(not any(p.host == host for p in phases) for host in hosts):
            raise ValueError("every injected host requires complete phase entries")
        batch = load_gamma_pair(
            directory / f"{request}_rps_start_time_{graph}.csv",
            directory / f"{request}_service_time_{graph}.csv",
            directory / f"{request}_ids_{graph}", directory / f"{request}_graph_paths_{graph}",
            timestamp_unit=manifest["timestamp_unit"], duration_unit=manifest["duration_unit"],
        )
        topology = hashlib.sha256(json.dumps([batch.node_ids, batch.edges], sort_keys=True).encode()).hexdigest()
        rows = summarize_batch(batch, run_id=spec["run_id"], request_type=request, graph_id=graph,
                               offered_rps=args["rps"][0], phases=phases,
                               guard_seconds=guard_seconds, window_seconds=window_seconds)
        for row in rows:
            row["topology_sha256"] = topology
        windows.extend(rows)
        audits.append({"run_id": spec["run_id"], "offered_rps": args["rps"][0],
                       "arguments": args, "phase_intervals": [vars(p) for p in phases],
                       "topology_sha256": topology, "quality": batch.quality})
    grouped = defaultdict(list)
    for row in windows:
        if (row["phase"] in ("mixed_phase_window", "transition_guard")
                or row["first_or_last_observed_window"]):
            continue
        grouped[(row["run_id"], row["request_type"], row["graph_id"], row["offered_rps"],
                 row["phase"], row["topology_sha256"])].append(row)
    curves = []
    for key, rows in sorted(grouped.items()):
        requests = sum(r["requests"] for r in rows)
        total = sum(r["sum_response_seconds"] for r in rows)
        curves.append(dict(zip(
            ("run_id", "request_type", "graph_id", "offered_rps", "phase", "topology_sha256"), key),
            windows=len(rows), requests=requests, mean_response_seconds=total / requests,
            mean_response_raw=sum(r["mean_response_raw"] * r["requests"] for r in rows) / requests))
    write_csv(output / "windows.csv", windows)
    write_csv(output / "curves.csv", curves)
    report = {"dataset_doi": manifest["dataset_doi"], "dataset_version": manifest["dataset_version"],
              "request_scope": manifest["scope"], "window_seconds": window_seconds,
              "transition_guard_seconds": guard_seconds, "runs": audits,
              "total_request_rows": sum(a["quality"]["request_rows"] for a in audits),
              "result_type": "observed descriptive means; no inference speedup experiment",
              "phase_scope": "outside all logged injections versus exact logged treatment; transition and edge windows excluded from curves",
              "units_source": manifest["units_source"], "extraction_source": manifest["extraction_source"]}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "prepare"])
    parser.add_argument("--manifest", default="examples/queueing/gamma/sample_manifest.json")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-total-bytes", type=int, default=100_000_000)
    parser.add_argument("--guard-seconds", type=float)
    parser.add_argument("--window-seconds", type=float)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if args.action == "download":
        download_files(manifest, args.data_dir, max_total_bytes=args.max_total_bytes)
    else:
        if args.output_dir is None or args.guard_seconds is None or args.window_seconds is None:
            parser.error("prepare requires --output-dir, --guard-seconds and --window-seconds")
        report = prepare(manifest, args.data_dir, args.output_dir,
                         guard_seconds=args.guard_seconds, window_seconds=args.window_seconds)
        print(json.dumps({"request_rows": report["total_request_rows"], "runs": len(report["runs"])}))


if __name__ == "__main__":
    main()
