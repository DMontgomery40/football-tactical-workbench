from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.benchmark_eval import coco_detection


def _install_fake_cocoeval(monkeypatch, precision_matrix: np.ndarray) -> None:
    class FakeCOCOeval:
        def __init__(self, coco_gt, coco_dt, iou_type):  # noqa: ANN001
            self.params = SimpleNamespace(imgIds=[], catIds=[])
            self.eval = {"precision": precision_matrix.copy()}

        def evaluate(self) -> None:
            return None

        def accumulate(self) -> None:
            return None

    monkeypatch.setattr(coco_detection, "COCOeval", FakeCOCOeval)


def test_category_ap_uses_coco_precision_tensor_without_index_error(monkeypatch) -> None:
    precision = np.full((10, 5, 1, 1, 1), -1.0, dtype=float)
    precision[0, 0, 0, 0, 0] = 0.50
    precision[1, 0, 0, 0, 0] = 0.25
    _install_fake_cocoeval(monkeypatch, precision)

    result = coco_detection._category_ap(object(), object(), [1], 1)

    assert result == 0.375


def test_category_ap_returns_none_when_coco_precision_has_no_valid_entries(monkeypatch) -> None:
    precision = np.full((10, 5, 1, 1, 1), -1.0, dtype=float)
    _install_fake_cocoeval(monkeypatch, precision)

    result = coco_detection._category_ap(object(), object(), [1], 1)

    assert result is None
