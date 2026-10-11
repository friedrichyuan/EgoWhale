"""Quality levels that need no scene: run erosion, action statistics, and parsing the VLM's reply."""

from __future__ import annotations

import numpy as np

from egowhale.quality import actions, checks, vlm


def test_erode_drops_short_runs_between_failures_only():
    fps = 30.0
    short = int(checks.MIN_RUN_SECONDS * fps) - 1
    valid = np.array([True] * 3 + [False] + [True] * short + [False] + [True] * 20)
    out = checks.erode(valid, fps)
    assert out[:3].all(), "a short run at the start is not sandwiched"
    assert not out[4 : 4 + short].any()
    assert out[-20:].all()


def test_trajectory_shapes_and_rotation_deltas():
    frames = 6
    position = np.zeros((frames, 2, 3))
    position[:, 0, 0] = np.arange(frames) * 0.01
    rotation = np.repeat(np.eye(3)[None, None], frames, axis=0).repeat(2, axis=1)
    width = np.full((frames, 2), 0.05)
    state, action = actions.trajectory(position, rotation, width)
    assert state.shape == (frames, 8) and action.shape == (frames - 1, 14)
    np.testing.assert_allclose(action[:, 0], 0.01)
    np.testing.assert_allclose(action[:, 6:12], 0.0, atol=1e-12)


def test_outliers_and_sudden_flag_a_single_jump():
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(0, 0.001, size=(200, 3)), axis=0)
    series[120:] += 0.5
    valid = np.ones(len(series), dtype=bool)
    flagged = actions.sudden(series, valid)
    assert flagged[120] and flagged.sum() <= 6
    action = np.diff(series, axis=0)
    out = actions.outliers(action, valid)
    assert out[120] and out.sum() <= 3


def test_verdict_reads_json_inside_text_and_tolerates_garbage():
    reply = 'Sure. {"is consistent": false, "confidence": 0.8, "reasoning": "wrong object"} done'
    assert vlm._verdict(reply) == {"status": "ok", "is_consistent": False, "confidence": 0.8, "reasoning": "wrong object"}
    assert vlm._verdict("no json here")["status"] == "error"
    assert vlm._verdict("{not: json}")["status"] == "error"


def test_audit_without_a_key_is_skipped(monkeypatch, tmp_path):
    monkeypatch.delenv("EGOWHALE_VLM_API_KEY", raising=False)
    assert vlm.audit(tmp_path / "none.mp4", 0, 10, 30.0, "task")["status"] == "skipped"
