from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

try:
    import yaml
except Exception:  # pragma: no cover - environment dependent
    yaml = None  # type: ignore[assignment]

from app.wide_angle import choose_device

from .common import BenchmarkEvaluationUnavailable, metric_value, na_metric, normalize_label

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_ALIASES = {
    "ball": {"ball", "football", "soccer ball", "sports ball"},
    "goalkeeper": {"goalkeeper", "goalie", "keeper"},
    "player": {"player", "footballer", "athlete", "person"},
    "referee": {"referee", "ref", "official", "linesman"},
}


def _find_dataset_yaml(dataset_root: Path) -> Path:
    for candidate_name in ("data.yaml", "dataset.yaml"):
        candidate = dataset_root / candidate_name
        if candidate.exists():
            return candidate
    raise BenchmarkEvaluationUnavailable(f"No dataset YAML found under {dataset_root}")


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise BenchmarkEvaluationUnavailable("PyYAML is required to evaluate YOLO datasets.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise BenchmarkEvaluationUnavailable(f"Dataset YAML is not an object: {path}")
    return payload


def _resolve_list_or_dir(base_dir: Path, raw_value: Any) -> list[Path]:
    if isinstance(raw_value, list):
        resolved: list[Path] = []
        for item in raw_value:
            resolved.extend(_resolve_list_or_dir(base_dir, item))
        return resolved
    candidate = Path(str(raw_value or "").strip())
    if not candidate:
        return []
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    if candidate.is_dir():
        return sorted(path for path in candidate.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if candidate.is_file() and candidate.suffix.lower() == ".txt":
        items: list[Path] = []
        for line in candidate.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_path = Path(line.strip()).expanduser()
            if not image_path.is_absolute():
                image_path = (candidate.parent / image_path).resolve()
            items.append(image_path)
        return items
    if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
        return [candidate]
    return []


def _resolve_split_images(dataset_root: Path, dataset_yaml: dict[str, Any], split_key: str) -> list[Path]:
    base_dir = dataset_root
    if str(dataset_yaml.get("path") or "").strip():
        path_root = Path(str(dataset_yaml["path"]).strip()).expanduser()
        base_dir = path_root if path_root.is_absolute() else (dataset_root / path_root).resolve()
    images = _resolve_list_or_dir(base_dir, dataset_yaml.get(split_key))
    if not images:
        raise BenchmarkEvaluationUnavailable(f"No images found for split '{split_key}' under {dataset_root}")
    return images


def _normalize_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(item).strip() for item in raw_names]
    if isinstance(raw_names, dict):
        ordered: list[tuple[int, str]] = []
        for key, value in raw_names.items():
            try:
                sort_key = int(key)
            except Exception:
                sort_key = len(ordered)
            ordered.append((sort_key, str(value).strip()))
        return [value for _index, value in sorted(ordered, key=lambda item: item[0])]
    return []


def _label_path_for_image(dataset_root: Path, image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        image_index = parts.index("images")
        return Path(*parts[:image_index], "labels", *parts[image_index + 1 :]).with_suffix(".txt")
    candidate = dataset_root / "labels" / image_path.with_suffix(".txt").name
    return candidate


def _load_image_shape(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise BenchmarkEvaluationUnavailable(f"Could not read benchmark image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def _class_aliases(label: str) -> set[str]:
    normalized = normalize_label(label)
    for canonical, aliases in LABEL_ALIASES.items():
        if normalized in aliases or normalized == canonical:
            return {canonical, *aliases}
    return {normalized}


def _selected_category_ids(names: list[str], suite_id: str) -> list[int]:
    if suite_id == "det.ball_quick_v1":
        return [
            index
            for index, label in enumerate(names)
            if normalize_label(label) in LABEL_ALIASES["ball"]
        ]
    return [index for index, label in enumerate(names) if normalize_label(label)]


def _build_ground_truth(
    *,
    dataset_root: Path,
    images: list[Path],
    names: list[str],
    suite_id: str,
) -> tuple[dict[str, Any], dict[int, int], dict[str, int]]:
    selected_ids = _selected_category_ids(names, suite_id)
    if not selected_ids:
        raise BenchmarkEvaluationUnavailable(f"No benchmark categories could be resolved for suite {suite_id}")
    category_id_map = {original_id: index + 1 for index, original_id in enumerate(selected_ids)}
    category_name_map = {
        normalize_label(names[original_id]): category_id_map[original_id]
        for original_id in selected_ids
    }
    categories = [
        {"id": category_id_map[original_id], "name": names[original_id]}
        for original_id in selected_ids
    ]
    image_rows: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_index, image_path in enumerate(images, start=1):
        width, height = _load_image_shape(image_path)
        image_rows.append({
            "id": image_index,
            "file_name": str(image_path.relative_to(dataset_root)),
            "width": width,
            "height": height,
        })
        label_path = _label_path_for_image(dataset_root, image_path)
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])
            except Exception:
                continue
            if class_id not in category_id_map:
                continue
            abs_width = box_width * width
            abs_height = box_height * height
            x_min = (x_center * width) - (abs_width / 2.0)
            y_min = (y_center * height) - (abs_height / 2.0)
            annotations.append({
                "id": annotation_id,
                "image_id": image_index,
                "category_id": category_id_map[class_id],
                "bbox": [x_min, y_min, abs_width, abs_height],
                "area": abs_width * abs_height,
                "iscrowd": 0,
            })
            annotation_id += 1
    gt = {
        "images": image_rows,
        "annotations": annotations,
        "categories": categories,
    }
    return gt, category_id_map, category_name_map


def _prediction_category_map(recipe: dict[str, Any], category_name_map: dict[str, int]) -> dict[int, int]:
    class_names = {
        int(key): str(value)
        for key, value in (recipe.get("class_mapping", {}).get("class_names") or {}).items()
    }
    if not class_names:
        class_names = {
            int(key): str(value)
            for key, value in (resolve_detector_spec(str(recipe.get("artifact_path") or "")).get("class_names") or {}).items()
        }
    mapping: dict[int, int] = {}
    for class_id, label in class_names.items():
        aliases = _class_aliases(label)
        target = next((category_id for category_name, category_id in category_name_map.items() if category_name in aliases), None)
        if target is not None:
            mapping[int(class_id)] = int(target)
    return mapping


def _safe_load_results(coco_gt: COCO, predictions: list[dict[str, Any]]) -> Any:
    return coco_gt.loadRes(predictions if predictions else [])


def _category_ap(coco_gt: COCO, coco_dt: Any, image_ids: list[int], category_id: int) -> float | None:
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = [category_id]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval.get("precision") if evaluator.eval is not None else None
    if precision is None or getattr(precision, "size", 0) == 0:
        return None
    category_precision = precision[:, :, 0, 0, -1]
    valid = category_precision[category_precision > -1]
    if getattr(valid, "size", 0) == 0:
        return None
    return float(valid.mean())


def evaluate_coco_detection(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    if not dataset_root_path.exists():
        raise BenchmarkEvaluationUnavailable(f"Dataset root is missing: {dataset_root_path}")
    dataset_yaml_path = _find_dataset_yaml(dataset_root_path)
    dataset_yaml = _load_yaml(dataset_yaml_path)
    names = _normalize_names(dataset_yaml.get("names"))
    split_key = str(suite.get("dataset_split") or "test")
    images = _resolve_split_images(dataset_root_path, dataset_yaml, split_key)

    ground_truth, _category_id_map, category_name_map = _build_ground_truth(
        dataset_root=dataset_root_path,
        images=images,
        names=names,
        suite_id=str(suite.get("id") or ""),
    )
    ground_truth_path = Path(artifacts_dir).expanduser().resolve() / "ground_truth.json"
    predictions_path = Path(artifacts_dir).expanduser().resolve() / "predictions.json"
    ground_truth_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    model_path = str(recipe.get("artifact_path") or "")
    model = YOLO(model_path)
    device = choose_device()
    prediction_category_map = _prediction_category_map(recipe, category_name_map)
    predictions: list[dict[str, Any]] = []
    total_inference_seconds = 0.0
    image_id_by_path = {Path(image["file_name"]): int(image["id"]) for image in ground_truth["images"]}

    for image_path in images:
        relative_path = image_path.relative_to(dataset_root_path)
        image_id = image_id_by_path.get(relative_path)
        if image_id is None:
            continue
        start = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            conf=0.001,
            iou=0.7,
            device=device,
            verbose=False,
        )
        total_inference_seconds += time.perf_counter() - start
        if not results:
            continue
        boxes = results[0].boxes
        if boxes is None or boxes.cls is None or boxes.conf is None or boxes.xyxy is None:
            continue
        classes = boxes.cls.cpu().numpy().astype(int).tolist()
        confidences = boxes.conf.cpu().numpy().astype(float).tolist()
        coordinates = boxes.xyxy.cpu().numpy().astype(float).tolist()
        for class_id, score, (x1, y1, x2, y2) in zip(classes, confidences, coordinates, strict=False):
            category_id = prediction_category_map.get(int(class_id))
            if category_id is None:
                continue
            predictions.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score),
            })

    predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    coco_dt = _safe_load_results(coco_gt, predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = [int(image["id"]) for image in ground_truth["images"]]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats if evaluator.stats is not None else []
    ap_50_95 = float(stats[0]) if len(stats) > 0 else None
    ap_50 = float(stats[1]) if len(stats) > 1 else None
    ap_75 = float(stats[2]) if len(stats) > 2 else None
    avg_precision = float(stats[0]) if len(stats) > 0 else None
    avg_recall = float(stats[8]) if len(stats) > 8 else None

    category_metrics: dict[str, dict[str, Any]] = {}
    for category in ground_truth["categories"]:
        category_name = normalize_label(category["name"])
        category_ap = _category_ap(coco_gt, coco_dt, evaluator.params.imgIds, int(category["id"]))
        category_metrics[f"ap_{category_name}_50_95"] = metric_value(category_ap, label=f"AP {category['name']}")

    images_per_second = (len(images) / total_inference_seconds) if total_inference_seconds > 0 else None
    avg_image_latency_ms = ((total_inference_seconds / len(images)) * 1000.0) if images else None

    metrics = {
        "ap_50_95": metric_value(ap_50_95, label="AP@[.50:.95]"),
        "ap_50": metric_value(ap_50, label="AP50"),
        "ap_75": metric_value(ap_75, label="AP75"),
        "precision": metric_value(avg_precision, label="Precision"),
        "recall": metric_value(avg_recall, label="Recall"),
        "images_per_second": metric_value(images_per_second, label="Images/s", precision=2),
        "avg_image_latency_ms": metric_value(avg_image_latency_ms, label="Latency", unit=" ms", precision=2),
    }
    metrics.update(category_metrics)
    if str(suite.get("id") or "") == "det.ball_quick_v1":
        metrics["ap_ball_50_95"] = metrics.get("ap_ball_50_95") or metrics.get("ap_ball")
        if metrics["ap_ball_50_95"] is None:
            metrics["ap_ball_50_95"] = category_metrics.get("ap_ball_50_95", na_metric(label="AP ball"))

    return {
        "metrics": metrics,
        "artifacts": {
            "ground_truth_json": str(ground_truth_path),
            "predictions_json": str(predictions_path),
        },
        "raw_result": {
            "total_images": len(images),
            "total_annotations": len(ground_truth["annotations"]),
            "device": device,
            "prediction_class_map": prediction_category_map,
        },
    }
