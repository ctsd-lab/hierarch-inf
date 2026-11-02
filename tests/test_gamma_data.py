"""Handwritten synthetic schema fixtures; no GAMMA observations are bundled."""

import json

import numpy as np
import pytest

from hierarch_inf.queueing.gamma import (
    InterferencePhase, load_gamma_pair, load_run_arguments, phase_labels,
    read_graph_paths, read_phase_log, summarize_batch,
)


def fixture_paths(tmp_path):
    paths = [tmp_path / n for n in ("starts.csv", "durations.csv", "ids", "paths")]
    paths[0].write_text("0,1\n3000000,3000010\n1000000,999990\n2000000,2000010\n")
    paths[1].write_text("0,1\n3000,1000\n1000,500\n2000,1000\n")
    paths[2].write_text("0,root\n1,child\n")
    paths[3].write_text("0->1\n")
    return paths


def test_pair_sort_and_root_duration_preserve_row_identity(tmp_path):
    paths = fixture_paths(tmp_path)
    b = load_gamma_pair(*paths, timestamp_unit="us", duration_unit="us")
    np.testing.assert_allclose(b.root_start_seconds, [1, 2, 3])
    np.testing.assert_allclose(b.root_duration_seconds, [.001, .002, .003])
    np.testing.assert_array_equal(b.original_row, [1, 2, 0])
    assert b.quality["rows_with_child_start_before_root"] == 1
    assert b.edges == [(0, 1)]


@pytest.mark.parametrize("contents", ["0,2\n1,2\n", "0,1\n1\n", "0,1\nNaN,1\n", "0,1\n-1,2\n"])
def test_bad_matrix_rejected(tmp_path, contents):
    paths = fixture_paths(tmp_path)
    paths[0].write_text(contents)
    with pytest.raises(ValueError):
        load_gamma_pair(*paths, timestamp_unit="us", duration_unit="us")


def test_pair_row_mismatch_rejected(tmp_path):
    paths = fixture_paths(tmp_path)
    paths[1].write_text("0,1\n100,200\n")
    with pytest.raises(ValueError, match="identical columns and row counts"):
        load_gamma_pair(*paths, timestamp_unit="us", duration_unit="us")


def test_phase_logs_and_transition_guard(tmp_path):
    p = tmp_path / "phases"
    p.write_text("Bottleneck of type cpu with measure 0.75 starts at 10\n"
                 "Bottleneck of type CPU with measure 0.75 ends at 20\n")
    phases = read_phase_log(p, host="host1")
    assert phases == [InterferencePhase("host1", "cpu", .75, 10., 20.)]
    labels = phase_labels([1, 9, 15, 21, 30], [0.1] * 5, phases, guard_seconds=2)
    assert labels.tolist() == ["no_logged_interference", "transition_guard",
                              "interference:host1/cpu/0.75", "transition_guard",
                              "no_logged_interference"]
    assert phase_labels([9], [7], phases, guard_seconds=0)[0] == "transition_guard"


@pytest.mark.parametrize("contents", [
    "Bottleneck of type cpu with measure 1 starts at 10\n",
    "Bottleneck of type cpu with measure 1 ends at 10\n",
    "Bottleneck of type cpu with measure 1 starts at 10\nBottleneck of type cpu with measure 2 ends at 20\n",
    "unknown log row\n",
])
def test_incomplete_or_invalid_phase_log_rejected(tmp_path, contents):
    p = tmp_path / "phases"
    p.write_text(contents)
    with pytest.raises(ValueError):
        read_phase_log(p, host="h")


def test_graph_checks_cycle_and_node_coverage(tmp_path):
    p = tmp_path / "paths"
    p.write_text("0->1->2\n0->2->1\n")
    with pytest.raises(ValueError, match="cycle"):
        read_graph_paths(p, 3)
    p.write_text("0->1\n")
    with pytest.raises(ValueError, match="cover"):
        read_graph_paths(p, 3)


def test_means_use_requests_and_keep_offered_load_distinct(tmp_path):
    b = load_gamma_pair(*fixture_paths(tmp_path), timestamp_unit="us", duration_unit="us")
    rows = summarize_batch(b, run_id="synthetic", request_type="fixture", graph_id=1,
                           offered_rps=200, phases=[], guard_seconds=0, window_seconds=10)
    assert len(rows) == 1
    assert rows[0]["requests"] == 3
    assert rows[0]["mean_response_seconds"] == pytest.approx(.002)
    assert rows[0]["observed_graph_rps"] == pytest.approx(.3)
    assert rows[0]["offered_rps"] == 200
    assert rows[0]["first_or_last_observed_window"]


def test_arguments_require_single_positive_load(tmp_path):
    p = tmp_path / "args.txt"
    p.write_text(json.dumps({"rps": [200]}))
    assert load_run_arguments(p)["rps"] == [200]
    p.write_text(json.dumps({"rps": [100, 200]}))
    with pytest.raises(ValueError):
        load_run_arguments(p)


def test_phase_intervals_cannot_overlap_or_reverse_time(tmp_path):
    p = tmp_path / "phases"
    p.write_text("Bottleneck of type cpu with measure 0.75 starts at 10\n"
                 "Bottleneck of type cpu with measure 0.75 ends at 20\n"
                 "Bottleneck of type cpu with measure 0.75 starts at 15\n"
                 "Bottleneck of type cpu with measure 0.75 ends at 25\n")
    with pytest.raises(ValueError, match="previous phase"):
        read_phase_log(p, host="h")
