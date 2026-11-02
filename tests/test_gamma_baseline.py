"""Baseline fixtures are synthetic; notebook CSVs are documented actual observations."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest
from examples.queueing.gamma_baseline import fit_affine, predict_affine


def test_recovers_known_curve_with_unequal_weights():
    x=np.array([200,200,800,800])
    y=.02+.01*x/1000
    p=fit_affine(x,y,[1,3,2,8])
    np.testing.assert_allclose(p,[.02,.01],atol=1e-12)
    np.testing.assert_allclose(predict_affine(p,[400]),[.024])


def test_decreasing_data_gives_nonnegative_slope():
    p=fit_affine([200,800],[.03,.02],[1,1])
    assert p[1]>=0
    assert p[1]<1e-5


@pytest.mark.parametrize('x,y,w', [([1],[2],[1]),([1,1],[2,3],[1,1]),
    ([1,2],[2,float('nan')],[1,1]),([1,2],[2,3],[0,1]),([1,2],[-1,2],[1,1])])
def test_invalid_fit_input(x,y,w):
    with pytest.raises(ValueError): fit_affine(x,y,w)


def test_committed_split_is_temporal_and_holds_out_workload():
    path=Path(__file__).parents[1]/'examples/queueing/gamma/notebook_data/baseline_windows.csv'
    rows=list(csv.DictReader(path.open()))
    keys=[(r['run_id'],r['block']) for r in rows]
    assert len(keys)==len(set(keys))
    for r in rows:
        load=float(r['offered_rps']);block=int(r['block'])
        expected='heldout_workload' if load==400 else ('train' if block<12 else 'heldout_time')
        assert r['split']==expected
        assert float(r['requests'])>0
    assert {float(r['offered_rps']) for r in rows if r['split']=='train'}=={200,800}


def test_executed_notebook_has_real_outputs_and_no_errors():
    path=Path(__file__).parents[1]/'examples/queueing/gamma_baseline.ipynb'
    nb=json.loads(path.read_text()); cells=[c for c in nb['cells'] if c['cell_type']=='code']
    assert all(c['execution_count'] is not None for c in cells)
    outputs=[o for c in cells for o in c['outputs']]
    assert not any(o['output_type']=='error' for o in outputs)
    assert sum('image/png' in o.get('data',{}) for o in outputs)>=7
