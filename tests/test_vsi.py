import numpy as np

from vsi.core import (Subtitle, VSIConfig, fuse_scores, select_keyframes,
                      soft_threshold, subtitle_frame_scores)
from vsi.ocr import merge_ocr_observations, normalize_ocr_text


def test_paper_soft_threshold_equation():
    np.testing.assert_allclose(soft_threshold([0.2, 0.6, 0.9]), [0.2, 0.8, 1.0])


def test_gaussian_subtitle_peak_and_window():
    scores = subtitle_frame_scores([Subtitle(4, 6, "event")], [0.8], 100, 10)
    assert scores[50] == np.max(scores)
    assert scores[50] == 0.8
    assert scores[19] == 0 and scores[81] == 0


def test_constant_missing_branch_does_not_add_bias():
    np.testing.assert_allclose(fuse_scores([0, 0, 0], [0, 1, 0], 0.3),
                               0.3 * np.array([-0.70710528, 1.41421056, -0.70710528]), rtol=1e-5)


def test_iterative_search_focuses_and_is_deterministic():
    truth = np.exp(-((np.arange(101) - 70) ** 2) / (2 * 5**2))
    cfg = VSIConfig(top_k=4, samples_per_round=8, detection_budget=40, text_weight=0.5, seed=7)
    result = select_keyframes(101, 10, lambda ids: truth[ids], text_scores=truth, config=cfg)
    assert result.rounds == 5
    assert len(result.visited_indices) == 40
    assert any(abs(i - 70) <= 5 for i in result.frame_indices)
    assert result.frame_indices == sorted(result.frame_indices)


def test_ocr_observations_become_timed_subtitle_segments():
    observations = [(0.0, "白 蛇"), (0.5, "白蛇"), (1.0, ""), (2.0, "许仙"), (2.5, "许 仙")]
    segments = merge_ocr_observations(observations, sample_period=0.5)
    assert [(x.start, x.end, x.text) for x in segments] == [
        (0.0, 1.0, "白蛇"), (2.0, 3.0, "许仙")
    ]
    assert normalize_ocr_text(" <白 蛇>！ ") == "白蛇！"
