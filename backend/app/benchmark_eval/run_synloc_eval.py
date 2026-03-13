from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SynLoc evaluation and emit JSON.")
    parser.add_argument("--ground-truth-json", required=True, type=Path)
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict | list:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"Expected a JSON object or array at {path}.")
    return payload


def _normalized_area(annotation: dict) -> float:
    area = annotation.get("area")
    if area is not None:
        try:
            return float(area)
        except (TypeError, ValueError):
            pass
    bbox = annotation.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return float(bbox[2]) * float(bbox[3])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _normalize_ground_truth(payload: dict) -> dict:
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("SynLoc ground truth must contain an `annotations` array.")
    normalized_payload = dict(payload)
    normalized_annotations = []
    for index, annotation in enumerate(annotations, start=1):
        if not isinstance(annotation, dict):
            continue
        normalized = dict(annotation)
        normalized.setdefault("id", index)
        normalized.setdefault("iscrowd", 0)
        normalized.setdefault("ignore", 0)
        normalized["area"] = _normalized_area(normalized)
        normalized_annotations.append(normalized)
    normalized_payload["annotations"] = normalized_annotations
    return normalized_payload


def _build_coco(payload: dict) -> COCO:
    coco = COCO()
    coco.dataset = payload
    coco.createIndex()
    return coco


class LocSimCOCOeval(COCOeval):
    locsim_tau = 1

    def get_img_pos(self, dt):
        return [np.array(det["keypoints"]).reshape(-1, 3)[self.params.position_from_keypoint_index, :2] for det in dt]

    def computeIoU(self, imgId, catId):
        from sskit import image_to_ground

        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 or len(dt) == 0:
            return []
        inds = np.argsort([-float(d.get("score", 0.0)) for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        img = self.cocoGt.loadImgs(int(imgId))[0]
        if hasattr(self.params, "position_from_keypoint_index") and self.params.position_from_keypoint_index is not None:
            img_pos_dt = np.array(self.get_img_pos(dt))
            w, h = np.float32(img["width"]), np.float32(img["height"])
            nimg_pos_dt = ((img_pos_dt - ((w - 1) / 2, (h - 1) / 2)) / w).astype(np.float32)
            bev_dt = image_to_ground(img["camera_matrix"], img["undist_poly"], nimg_pos_dt)[:, :2]
        else:
            bev_dt = np.array([det["position_on_pitch"] for det in dt])
        bev_gt = np.array([det["position_on_pitch"] for det in gt])

        aa, bb = np.meshgrid(bev_gt[:, 0], bev_dt[:, 0])
        dist2 = (aa - bb) ** 2
        aa, bb = np.meshgrid(bev_gt[:, 1], bev_dt[:, 1])
        dist2 += (aa - bb) ** 2

        locsim = np.exp(np.log(0.05) * dist2 / self.locsim_tau**2)
        return locsim

    def accumulate(self, p=None):
        if p is None:
            p = self.params
        super().accumulate(p)

        iou = p.iouThrs == 0.5
        area = p.areaRngLbl.index("all")
        dets = np.argmax(p.maxDets)

        precision = np.squeeze(self.eval["precision"][iou, :, 0, area, dets])
        scores = np.squeeze(self.eval["scores"][iou, :, 0, area, dets])
        recall = p.recThrs
        f1 = 2 * precision * recall / (precision + recall)

        self.eval["precision_50"] = precision
        self.eval["recall_50"] = recall
        self.eval["f1_50"] = f1
        self.eval["scores_50"] = scores

    def frame_accuracy(self, threshold):
        rng = self.params.areaRng[self.params.areaRngLbl.index("all")]
        iou = self.params.iouThrs == 0.5

        ok = bad = 0
        for e in self.evalImgs:
            if e is None:
                continue
            if e["aRng"] == rng:
                matches = (e["dtMatches"][iou] > -1)[0]
                if (np.array(e["dtScores"])[matches] > threshold).sum() == len(e["gtIds"]):
                    ok += 1
                else:
                    bad += 1
        return ok / (ok + bad) if (ok + bad) > 0 else 0.0

    def summarize(self):
        super().summarize()
        if hasattr(self.params, "score_threshold") and self.params.score_threshold is not None:
            threshold = self.params.score_threshold
        else:
            if len(self.eval["scores_50"]) == 0:
                threshold = 0.0
            else:
                i = int(self.eval["f1_50"].argmax())
                next_index = min(i + 1, len(self.eval["scores_50"]) - 1)
                if next_index == i:
                    threshold = float(self.eval["scores_50"][i])
                else:
                    threshold = float(self.eval["scores_50"][i] + self.eval["scores_50"][next_index]) / 2.0
        i = np.searchsorted(-self.eval["scores_50"], -threshold, "right") - 1
        i = max(min(int(i), len(self.eval["scores_50"]) - 1), 0)
        stats = [self.eval["precision_50"][i], self.eval["recall_50"][i], self.eval["f1_50"][i], threshold, self.frame_accuracy(threshold)]
        self.stats = np.concatenate([self.stats, stats])

        print()
        print(f"  Precision      @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[0]:5.3f}")
        print(f"  Recall         @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[1]:5.3f}")
        print(f"  F1             @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[2]:5.3f}")
        print(f"  Frame Accuracy @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[4]:5.3f}")
        print(f"  mAP-LocSim     @[ LocSim=0.50:0.95 | ScoreTh={threshold:5.3f} ] = {self.stats[0]:5.3f}")


def main() -> int:
    args = _parse_args()
    repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "sskit"
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    ground_truth_json = args.ground_truth_json.expanduser().resolve()
    predictions_json = args.predictions_json.expanduser().resolve()
    metadata_json = args.metadata_json.expanduser().resolve()

    metadata = {}
    if metadata_json.exists():
        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            metadata = {}

    start = perf_counter()
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        ground_truth_payload = _normalize_ground_truth(_load_json(ground_truth_json))
        prediction_payload = _load_json(predictions_json)
        if not isinstance(prediction_payload, list):
            raise ValueError("SynLoc predictions must be a JSON array of detection records.")

        coco = _build_coco(ground_truth_payload)
        coco_det = coco.loadRes(prediction_payload)
        coco_eval = LocSimCOCOeval(coco, coco_det, "bbox")
        coco_eval.params.useSegm = None
        if metadata.get("score_threshold") is not None:
            coco_eval.params.score_threshold = float(metadata["score_threshold"])
        if metadata.get("position_from_keypoint_index") is not None:
            coco_eval.params.position_from_keypoint_index = int(metadata["position_from_keypoint_index"])

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    elapsed_seconds = perf_counter() - start

    images = coco.dataset.get("images") or []
    frames_per_second = (len(images) / elapsed_seconds) if elapsed_seconds > 0 and images else None
    avg_frame_latency_ms = ((elapsed_seconds / len(images)) * 1000.0) if elapsed_seconds > 0 and images else None
    precision, recall, f1, score_threshold, frame_accuracy = [float(value) for value in coco_eval.stats[12:17]]
    payload = {
        "map_locsim": float(coco_eval.stats[0]),
        "precision_at_05": precision,
        "recall_at_05": recall,
        "f1_at_05": f1,
        "score_threshold": score_threshold,
        "frame_accuracy": frame_accuracy,
        "frames_per_second": frames_per_second,
        "avg_frame_latency_ms": avg_frame_latency_ms,
        "images_evaluated": len(images),
        "elapsed_seconds": elapsed_seconds,
        "vendor_stdout": captured_stdout.getvalue(),
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
