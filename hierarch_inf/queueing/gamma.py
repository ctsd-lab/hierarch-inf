"""Validated readers for GAMMA's paired RPC start/latency matrices.

These are measured spans. The release's ``service_time`` filename is retained
as provenance; its values are treated as inclusive RPC response durations,
not identified queue service requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


_TIME_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix(path):
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"empty matrix: {path}") from error
        if header != [str(i) for i in range(len(header))] or not header:
            raise ValueError("matrix columns must be consecutive RPC node indices from zero")
        rows = []
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"row width mismatch at {path}:{line}")
            try:
                values = [float(x) for x in row]
            except ValueError as error:
                raise ValueError(f"nonnumeric entry at {path}:{line}") from error
            if not np.all(np.isfinite(values)) or min(values) < 0:
                raise ValueError(f"nonfinite or negative entry at {path}:{line}")
            rows.append(values)
    return header, np.asarray(rows, dtype=float).reshape(-1, len(header))


def read_node_ids(path):
    """Read positional node/RPC identities; these IDs are not request IDs."""
    result = {}
    with Path(path).open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 2:
                raise ValueError("node ID rows require index and RPC name")
            index = int(row[0])
            if index in result:
                raise ValueError("duplicate node index")
            result[index] = row[1].strip()
    if sorted(result) != list(range(len(result))):
        raise ValueError("node IDs must be consecutive from zero")
    return result


def read_graph_paths(path, node_count):
    edges = set()
    covered = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        nodes = [int(x) for x in line.strip().split("->")]
        if not nodes or nodes[0] != 0 or len(set(nodes)) != len(nodes):
            raise ValueError("each graph path must be acyclic and begin at root node zero")
        if any(x < 0 or x >= node_count for x in nodes):
            raise ValueError("graph path contains an unknown node")
        covered.update(nodes)
        edges.update(zip(nodes, nodes[1:]))
    if covered != set(range(node_count)):
        raise ValueError("graph paths do not cover every matrix node")
    # Check the union, since separately acyclic paths can create a cycle.
    indegree = [0] * node_count
    successors = [[] for _ in range(node_count)]
    for a, b in edges:
        indegree[b] += 1
        successors[a].append(b)
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in successors[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != node_count:
        raise ValueError("combined graph paths contain a cycle")
    return sorted(edges)


@dataclass
class GammaTraceBatch:
    root_start_seconds: np.ndarray
    root_duration_seconds: np.ndarray
    root_duration_raw: np.ndarray
    original_row: np.ndarray
    node_ids: dict
    edges: list
    quality: dict


def load_gamma_pair(start_path, duration_path, node_ids_path, graph_paths_path, *,
                    timestamp_unit, duration_unit):
    """Load one run/request-type/graph pair, sorting rows chronologically.

    Explicit units are required. CSV row correspondence is supplied by the
    source release; shape checks cannot establish correspondence independently.
    All nodes are validated, while root column zero defines request latency.
    """
    if timestamp_unit not in _TIME_SCALE or duration_unit not in _TIME_SCALE:
        raise ValueError("time units must be s, ms, us, or ns")
    start_columns, starts = _matrix(start_path)
    duration_columns, durations = _matrix(duration_path)
    if start_columns != duration_columns or starts.shape != durations.shape:
        raise ValueError("paired matrices must have identical columns and row counts")
    if not len(starts):
        raise ValueError("trace pair contains no requests")
    nodes = read_node_ids(node_ids_path)
    if len(nodes) != starts.shape[1]:
        raise ValueError("node ID count does not match matrix columns")
    edges = read_graph_paths(graph_paths_path, len(nodes))
    start_seconds = starts * _TIME_SCALE[timestamp_unit]
    duration_seconds = durations * _TIME_SCALE[duration_unit]
    order = np.argsort(start_seconds[:, 0], kind="stable")
    # Clock-skew diagnostics only: inclusive span durations are never
    # converted into intrinsic services by parent-child subtraction.
    negative_offsets = start_seconds[:, 1:] < start_seconds[:, :1]
    quality = {
        "request_rows": len(starts), "rpc_nodes": starts.shape[1],
        "rows_with_child_start_before_root": int(np.any(negative_offsets, axis=1).sum()),
        "zero_root_duration_rows": int((durations[:, 0] == 0).sum()),
        "input_rows_chronological": bool(np.all(np.diff(starts[:, 0]) >= 0)),
        "timestamp_unit": timestamp_unit, "duration_unit": duration_unit,
        "latency_semantics": "root inclusive response duration",
        "request_identity": "source row index; release does not supply trace IDs in these files",
        "source_sha256": {str(Path(p).name): file_sha256(p)
                          for p in (start_path, duration_path, node_ids_path, graph_paths_path)},
    }
    return GammaTraceBatch(start_seconds[order, 0], duration_seconds[order, 0],
                           durations[order, 0], order, nodes, edges, quality)


@dataclass(frozen=True)
class InterferencePhase:
    host: str
    kind: str
    measure: float
    start: float
    end: float


_PHASE = re.compile(r"^Bottleneck of type (\S+) with measure ([0-9.eE+-]+) "
                    r"(starts|ends) at ([0-9.eE+-]+)$")


def read_phase_log(path, *, host):
    phases, pending = [], None
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        match = _PHASE.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"unrecognized phase log line: {line}")
        kind, measure, action, timestamp = match.groups()
        measure, timestamp = float(measure), float(timestamp)
        if not np.isfinite(measure) or not np.isfinite(timestamp):
            raise ValueError("nonfinite phase entry")
        if action == "starts":
            if pending is not None or (phases and timestamp < phases[-1].end):
                raise ValueError("phase started before previous phase ended")
            pending = (kind.lower(), measure, timestamp)
        else:
            if pending is None or (kind.lower(), measure) != pending[:2] or timestamp < pending[2]:
                raise ValueError("phase end does not match its start")
            phases.append(InterferencePhase(host, kind.lower(), measure, pending[2], timestamp))
            pending = None
    if pending is not None:
        raise ValueError("phase log has an unmatched start")
    return phases


def phase_labels(starts, durations, phases, *, guard_seconds):
    """Label by overlap with complete injection logs, keeping boundaries separate."""
    starts, durations = np.asarray(starts, dtype=float), np.asarray(durations, dtype=float)
    if (starts.ndim != 1 or starts.shape != durations.shape
            or not np.all(np.isfinite(starts)) or not np.all(np.isfinite(durations))
            or np.any(durations < 0)):
        raise ValueError("start and duration vectors must be paired and finite with nonnegative durations")
    if not np.isfinite(guard_seconds) or guard_seconds < 0:
        raise ValueError("guard_seconds must be finite and nonnegative")
    labels = []
    for start, duration in zip(starts, durations):
        end = start + duration
        near = any(start <= boundary + guard_seconds and end >= boundary - guard_seconds
                   for p in phases for boundary in (p.start, p.end))
        active = [p for p in phases if start < p.end and end >= p.start]
        if near:
            labels.append("transition_guard")
        elif active:
            labels.append("interference:" + ";".join(sorted(
                f"{p.host}/{p.kind}/{p.measure:g}" for p in active)))
        else:
            labels.append("no_logged_interference")
    return np.asarray(labels)


def summarize_batch(batch, *, run_id, request_type, graph_id, offered_rps,
                    phases, guard_seconds, window_seconds):
    """Per-window request-weighted means, conditional on the supplied graph.

    Offered load comes from run arguments; observed graph-specific counts
    per window are reported separately and never substituted for total load.
    """
    if not np.isfinite(offered_rps) or offered_rps <= 0:
        raise ValueError("offered_rps must be positive")
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    labels = phase_labels(batch.root_start_seconds, batch.root_duration_seconds,
                          phases, guard_seconds=guard_seconds)
    origin = np.floor(batch.root_start_seconds.min() / window_seconds) * window_seconds
    bins = np.floor((batch.root_start_seconds - origin) / window_seconds).astype(int)
    edge_bins = (min(bins), max(bins))
    rows = []
    for index in sorted(set(bins)):
        mask = bins == index
        window_labels = set(labels[mask])
        label = next(iter(window_labels)) if len(window_labels) == 1 else "mixed_phase_window"
        values = batch.root_duration_seconds[mask]
        rows.append({
            "run_id": run_id, "request_type": request_type, "graph_id": graph_id,
            "offered_rps": float(offered_rps), "window_start_seconds": float(origin + index * window_seconds),
            "window_seconds": float(window_seconds), "phase": label, "requests": len(values),
            "observed_graph_rps": len(values) / window_seconds,
            "mean_response_seconds": float(values.mean()),
            "sum_response_seconds": float(values.sum()),
            "mean_response_raw": float(batch.root_duration_raw[mask].mean()),
            "response_std_seconds": float(values.std(ddof=1)) if len(values) > 1 else None,
            "first_or_last_observed_window": bool(index in edge_bins),
        })
    return rows


def load_run_arguments(path):
    arguments = json.loads(Path(path).read_text())
    if len(arguments.get("rps", [])) != 1:
        raise ValueError("each selected run must specify exactly one offered RPS")
    value = float(arguments["rps"][0])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("offered RPS must be positive and finite")
    return arguments


def phase_aligned_baselines(batch, phases, *, durations_seconds, guard_seconds):
    """Descriptive equal-duration windows ending before the first injection.

    Only windows fully covered by the observed root-start range are returned.
    Requests crossing either window edge are excluded. No stationarity or
    warm-up completion is inferred from this alignment.
    """
    if not phases or not np.isfinite(guard_seconds) or guard_seconds < 0:
        raise ValueError("complete phases and a finite nonnegative guard are required")
    durations = np.asarray(durations_seconds, dtype=float)
    if durations.ndim != 1 or np.any(~np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("baseline durations must be positive and finite")
    end = min(p.start for p in phases) - guard_seconds
    rows = []
    for duration in durations:
        start = end - duration
        if start < batch.root_start_seconds.min() or end > batch.root_start_seconds.max():
            continue
        keep = ((batch.root_start_seconds >= start)
                & (batch.root_start_seconds + batch.root_duration_seconds <= end))
        values = batch.root_duration_seconds[keep]
        if values.size:
            rows.append({"duration_seconds": float(duration), "start_epoch_seconds": start,
                         "end_epoch_seconds": end, "requests": int(values.size),
                         "mean_response_seconds": float(values.mean())})
    return rows
