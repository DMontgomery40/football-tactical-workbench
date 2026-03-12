from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.training_provenance import PROVENANCE_FILENAME
from app.training_ai_analysis import ARTIFACT_FILENAME as TRAINING_ANALYSIS_ARTIFACT_FILENAME
from app.wide_angle import (
    BALL_CLASS_LABEL_HINTS,
    PLAYER_CLASS_LABEL_HINTS,
    REFEREE_CLASS_LABEL_HINTS,
)

try:
    import ultralytics
except Exception:  # pragma: no cover - handled through config surface
    ultralytics = None  # type: ignore[assignment]

try:
    import yaml
except Exception:  # pragma: no cover - handled through scan errors
    yaml = None  # type: ignore[assignment]

TRAINING_BACKEND = "ultralytics"
TRAINING_BACKEND_LABEL = "Ultralytics YOLO"
TRAINING_BASE_WEIGHT_OPTIONS = [
    {"id": "soccana", "label": "soccana (football-pretrained)"},
]
TRAINING_EXPECTED_CLASSES = {"player", "goalkeeper", "referee", "ball"}
TRAINING_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAINING_DEVICE_OPTIONS = ["auto", "mps", "cpu", "cuda"]
TRAINING_LICENSE_NOTE = (
    "Detector fine-tuning currently uses the local Ultralytics stack already present in this repo. "
    "Ultralytics is AGPL by default unless you have their enterprise license, so keep that caveat in mind "
    "for commercial redistribution."
)
SUMMARY_FILENAME = "summary.json"
SCAN_FILENAME = "dataset_scan.json"
RUNTIME_DATASET_FILENAME = "dataset_runtime.yaml"


def get_training_backend_version() -> str | None:
    version = getattr(ultralytics, "__version__", None)
    return str(version) if version else None


def build_training_backend_config() -> dict[str, Any]:
    return {
        "backend": TRAINING_BACKEND,
        "backend_label": TRAINING_BACKEND_LABEL,
        "backend_version": get_training_backend_version(),
        "license_note": TRAINING_LICENSE_NOTE,
        "default_base_weights": "soccana",
        "available_base_weights": TRAINING_BASE_WEIGHT_OPTIONS,
        "device_options": TRAINING_DEVICE_OPTIONS,
        "default_hyperparameters": {
            "epochs": 50,
            "imgsz": 640,
            "batch": 16,
            "device": "auto",
            "workers": 4,
            "patience": 20,
            "freeze": None,
            "cache": False,
        },
    }


def _normalize_class_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _normalize_yaml_class_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(item).strip() for item in raw_names if str(item).strip()]
    if isinstance(raw_names, dict):
        normalized: list[tuple[int, str]] = []
        for key, value in raw_names.items():
            try:
                sort_key = int(key)
            except Exception:
                sort_key = len(normalized)
            label = str(value).strip()
            if label:
                normalized.append((sort_key, label))
        return [label for _index, label in sorted(normalized, key=lambda item: item[0])]
    return []


def _path_is_hidden_from_root(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return any(part.startswith(".") for part in relative.parts)


def _find_dataset_yaml_candidate(dataset_path: Path) -> Path | None:
    candidates: list[Path] = [
        dataset_path / "dataset.yaml",
        dataset_path / "data.yaml",
        dataset_path / "data" / "dataset.yaml",
        dataset_path / "data" / "data.yaml",
    ]
    candidates.extend(sorted(dataset_path.glob("*.yaml")))
    candidates.extend(sorted(dataset_path.glob("*.yml")))
    candidates.extend(sorted(dataset_path.glob("*/*.yaml")))
    candidates.extend(sorted(dataset_path.glob("*/*.yml")))
    candidates.extend(
        sorted(
            candidate
            for candidate in dataset_path.rglob("*.yaml")
            if candidate.is_file() and not _path_is_hidden_from_root(candidate, dataset_path)
        )
    )
    candidates.extend(
        sorted(
            candidate
            for candidate in dataset_path.rglob("*.yml")
            if candidate.is_file() and not _path_is_hidden_from_root(candidate, dataset_path)
        )
    )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen or not resolved.exists() or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            text = resolved.read_text(encoding="utf-8")
        except Exception:
            continue
        lowered = text.lower()
        if any(token in lowered for token in ("train:", "val:", "names:")):
            return resolved
    return None


def _safe_yaml_load(path: Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_json_load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _detect_non_yolo_dataset_hint(dataset_root: Path) -> str | None:
    metadata_path = dataset_root / "metadata.json"
    samples_path = dataset_root / "samples.json"
    fiftyone_path = dataset_root / "fiftyone.yml"
    present_files = [path.name for path in (fiftyone_path, metadata_path, samples_path) if path.exists()]
    if not present_files:
        return None

    metadata_payload = _safe_json_load(metadata_path) if metadata_path.exists() else {}
    sample_fields = metadata_payload.get("sample_fields")
    has_fiftyone_detection_field = False
    if isinstance(sample_fields, list):
        for field in sample_fields:
            if not isinstance(field, dict):
                continue
            name = str(field.get("name") or "").strip().lower()
            embedded_doc_type = str(field.get("embedded_doc_type") or "").strip()
            if name in {"ground_truth", "detections"} and embedded_doc_type.endswith("Detections"):
                has_fiftyone_detection_field = True
                break

    if not has_fiftyone_detection_field and not (samples_path.exists() and fiftyone_path.exists()):
        return None

    files_text = ", ".join(present_files)
    return (
        "Detected a FiftyOne-style dataset export "
        f"({files_text}) with JSON detection annotations. Training Studio expects a YOLO detector dataset "
        "with `dataset.yaml` or `data.yaml` plus per-image `.txt` labels, so convert or export this dataset "
        "to YOLO before fine-tuning."
    )


def _matching_class_ids(names: list[str], hints: tuple[str, ...]) -> list[int]:
    hint_set = {_normalize_class_name(hint) for hint in hints}
    matched_ids: list[int] = []
    for class_id, label in enumerate(names):
        normalized = _normalize_class_name(label)
        if not normalized:
            continue
        tokens = set(normalized.split())
        if normalized in hint_set or tokens.intersection(hint_set):
            matched_ids.append(class_id)
    return sorted(set(matched_ids))


def _resolve_reference_path(raw_value: str, base_dir: Path) -> Path:
    candidate = Path(str(raw_value).strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _read_image_list(list_path: Path) -> list[Path]:
    image_paths: list[Path] = []
    try:
        lines = list_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return image_paths
    for raw_line in lines:
        value = raw_line.strip()
        if not value:
            continue
        candidate = _resolve_reference_path(value, list_path.parent)
        if candidate.exists() and candidate.is_file():
            image_paths.append(candidate)
    return image_paths


def _collect_images_from_reference(value: Any, base_dir: Path) -> tuple[list[Path], list[str], list[str], list[str]]:
    image_paths: list[Path] = []
    refs: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    values = value if isinstance(value, list) else [value]
    for raw_value in values:
        if raw_value in {None, ""}:
            continue
        resolved = _resolve_reference_path(str(raw_value), base_dir)
        refs.append(str(resolved))
        if not resolved.exists():
            errors.append(f"Declared dataset path does not exist: {resolved}")
            continue
        if resolved.is_file():
            if resolved.suffix.lower() == ".txt":
                image_paths.extend(_read_image_list(resolved))
            elif resolved.suffix.lower() in TRAINING_IMAGE_SUFFIXES:
                image_paths.append(resolved)
            else:
                warnings.append(f"Unsupported dataset reference file was ignored: {resolved}")
            continue
        image_paths.extend(
            sorted(
                item.resolve()
                for item in resolved.rglob("*")
                if item.is_file() and item.suffix.lower() in TRAINING_IMAGE_SUFFIXES
            )
        )
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in image_paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped, refs, warnings, errors


def _derive_label_candidates(image_path: Path, dataset_root: Path) -> list[Path]:
    candidates: list[Path] = []
    same_dir = image_path.with_suffix(".txt")
    candidates.append(same_dir)

    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        replaced = Path(*parts[:idx], "labels", *parts[idx + 1 :]).with_suffix(".txt")
        candidates.append(replaced)

    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError:
        relative = None
    if relative is not None:
        rel_parts = list(relative.parts)
        if rel_parts and rel_parts[0] == "images":
            candidates.append((dataset_root / "labels" / Path(*rel_parts[1:])).with_suffix(".txt"))
        candidates.append((dataset_root / relative).with_suffix(".txt"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        deduped.append(resolved)
    return deduped


def _parse_label_file(label_path: Path, class_count: int) -> tuple[dict[str, int], list[str]]:
    text = label_path.read_text(encoding="utf-8", errors="replace")
    stripped = text.strip()
    if not stripped:
        return {"instances": 0, "empty": 1, "invalid_rows": 0}, []

    instances = 0
    invalid_rows = 0
    issues: list[str] = []
    for line_number, raw_line in enumerate(stripped.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            invalid_rows += 1
            issues.append(f"{label_path}: line {line_number} has {len(parts)} columns; detector labels need 5.")
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except Exception:
            invalid_rows += 1
            issues.append(f"{label_path}: line {line_number} is not numeric.")
            continue
        if class_id < 0 or (class_count > 0 and class_id >= class_count):
            invalid_rows += 1
            issues.append(f"{label_path}: line {line_number} uses class id {class_id} outside 0..{max(class_count - 1, 0)}.")
            continue
        x_center, y_center, width, height = values
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            invalid_rows += 1
            issues.append(f"{label_path}: line {line_number} has bbox values outside normalized YOLO ranges.")
            continue
        instances += 1
    return {"instances": instances, "empty": 0, "invalid_rows": invalid_rows}, issues


@dataclass
class SplitInspection:
    name: str
    source: str
    path: str | None
    refs: list[str]
    image_paths: list[Path] = field(default_factory=list)
    images: int = 0
    label_files: int = 0
    labeled_images: int = 0
    empty_label_files: int = 0
    missing_labels: int = 0
    invalid_label_files: int = 0
    invalid_label_rows: int = 0
    instances: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "path": self.path,
            "refs": self.refs,
            "images": self.images,
            "label_files": self.label_files,
            "labeled_images": self.labeled_images,
            "empty_label_files": self.empty_label_files,
            "missing_labels": self.missing_labels,
            "invalid_label_files": self.invalid_label_files,
            "invalid_label_rows": self.invalid_label_rows,
            "instances": self.instances,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class DatasetInspection:
    dataset_path: Path
    yaml_path: Path | None
    yaml_payload: dict[str, Any]
    classes: list[str]
    classes_source: str
    splits: dict[str, SplitInspection]
    warnings: list[str]
    errors: list[str]
    suggested_validation_strategy: str
    can_start: bool
    class_mapping: dict[str, Any]

    @property
    def tier(self) -> str:
        return "invalid" if self.errors else ("valid" if not self.warnings else "usable_with_warnings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.dataset_path),
            "tier": self.tier,
            "has_yaml": self.yaml_path is not None,
            "yaml_path": str(self.yaml_path) if self.yaml_path else None,
            "classes": self.classes,
            "class_count": len(self.classes),
            "classes_source": self.classes_source,
            "class_mapping": self.class_mapping,
            "split_paths": {name: split.path for name, split in self.splits.items()},
            "splits": {name: split.to_dict() for name, split in self.splits.items()},
            "warnings": self.warnings,
            "errors": self.errors,
            "can_start": self.can_start,
            "suggested_validation_strategy": self.suggested_validation_strategy,
            "detected_format": "yolo_detection",
        }


def _inspect_split(
    *,
    dataset_root: Path,
    split_name: str,
    source: str,
    refs: list[str],
    image_paths: list[Path],
    class_count: int,
) -> SplitInspection:
    path = refs[0] if refs else None
    split = SplitInspection(name=split_name, source=source, path=path, refs=refs)
    split.image_paths = image_paths
    split.images = len(image_paths)
    issue_samples: list[str] = []
    for image_path in image_paths:
        label_path = next((candidate for candidate in _derive_label_candidates(image_path, dataset_root) if candidate.exists()), None)
        if label_path is None:
            split.missing_labels += 1
            continue
        split.label_files += 1
        try:
            label_stats, issues = _parse_label_file(label_path, class_count)
        except Exception:
            split.invalid_label_files += 1
            issue_samples.append(f"{label_path}: could not be parsed.")
            continue
        split.instances += int(label_stats["instances"])
        split.empty_label_files += int(label_stats["empty"])
        split.invalid_label_rows += int(label_stats["invalid_rows"])
        if label_stats["instances"] > 0:
            split.labeled_images += 1
        if label_stats["empty"] == 0 and label_stats["instances"] == 0:
            split.invalid_label_files += 1
        if issues:
            issue_samples.extend(issues[:3])

    if split.invalid_label_rows > 0:
        split.errors.append(
            f"{split_name} has {split.invalid_label_rows} malformed label rows across {split.invalid_label_files or split.label_files} files."
        )
    if split.images > 0 and split.missing_labels > 0:
        split.warnings.append(
            f"{split_name} is missing label files for {split.missing_labels} of {split.images} images."
        )
    if split.empty_label_files > 0:
        split.warnings.append(
            f"{split_name} contains {split.empty_label_files} empty label files."
        )
    if issue_samples:
        split.errors.extend(issue_samples[:5])
    return split


def _discover_dataset_split(dataset_root: Path, split_name: str, yaml_payload: dict[str, Any], yaml_path: Path | None) -> tuple[str, list[str], list[Path], list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    yaml_base_dir = dataset_root
    raw_base = yaml_payload.get("path")
    if raw_base:
        yaml_base_dir = _resolve_reference_path(str(raw_base), yaml_path.parent if yaml_path else dataset_root)
    elif yaml_path:
        yaml_base_dir = yaml_path.parent
    if split_name in yaml_payload:
        images, refs, ref_warnings, ref_errors = _collect_images_from_reference(
            yaml_payload.get(split_name),
            yaml_base_dir,
        )
        warnings.extend(ref_warnings)
        errors.extend(ref_errors)
        return "dataset_yaml", refs, images, warnings, errors

    images_root = dataset_root / "images"
    split_dir = images_root / split_name
    if split_dir.exists() and split_dir.is_dir():
        images, refs, ref_warnings, ref_errors = _collect_images_from_reference(str(split_dir), dataset_root)
        warnings.extend(ref_warnings)
        errors.extend(ref_errors)
        return "images_folder", refs, images, warnings, errors

    if split_name == "train":
        fallback_roots = [images_root] if images_root.exists() else [dataset_root]
        for root in fallback_roots:
            images, refs, ref_warnings, ref_errors = _collect_images_from_reference(str(root), dataset_root)
            if images:
                warnings.extend(ref_warnings)
                errors.extend(ref_errors)
                return "folder_fallback", refs, images, warnings, errors
    return "missing", [], [], warnings, errors


def inspect_training_dataset(dataset_path: Path) -> DatasetInspection:
    folder = dataset_path.expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="Dataset folder does not exist")

    warnings: list[str] = []
    errors: list[str] = []
    non_yolo_dataset_hint = _detect_non_yolo_dataset_hint(folder)
    yaml_path = _find_dataset_yaml_candidate(folder)
    yaml_payload = _safe_yaml_load(yaml_path) if yaml_path else {}
    classes = _normalize_yaml_class_names(yaml_payload.get("names"))
    classes_source = str(yaml_path) if yaml_path and classes else ("missing" if not yaml_path else "unparsed")

    if yaml_path is None:
        errors.append(
            non_yolo_dataset_hint
            or "No dataset YAML with class names was found. V1 detector fine-tuning needs `dataset.yaml` or `data.yaml` so the app can map classes back into analysis."
        )
    elif yaml is None:
        errors.append("PyYAML is unavailable, so dataset YAML files cannot be parsed in this backend build.")
    elif not classes:
        errors.append(
            "A dataset YAML was found but no class names were parsed from `names`. Add explicit class names before starting detector fine-tuning."
        )

    class_mapping = {
        "player_class_ids": _matching_class_ids(classes, PLAYER_CLASS_LABEL_HINTS),
        "ball_class_ids": _matching_class_ids(classes, BALL_CLASS_LABEL_HINTS),
        "referee_class_ids": _matching_class_ids(classes, REFEREE_CLASS_LABEL_HINTS),
    }
    class_mapping["player_class_id"] = class_mapping["player_class_ids"][0] if class_mapping["player_class_ids"] else -1
    class_mapping["ball_class_id"] = class_mapping["ball_class_ids"][0] if class_mapping["ball_class_ids"] else -1
    class_mapping["referee_class_id"] = class_mapping["referee_class_ids"][0] if class_mapping["referee_class_ids"] else -1

    if classes:
        if not class_mapping["player_class_ids"]:
            errors.append("Class names do not include a player or goalkeeper label that the analysis pipeline can map.")
        if not class_mapping["ball_class_ids"]:
            errors.append("Class names do not include a ball label that the analysis pipeline can map.")
        if not class_mapping["referee_class_ids"]:
            warnings.append("No referee class was detected in the class list; analysis will continue without referee-specific filtering.")

    unknown_classes = sorted({label for label in classes if _normalize_class_name(label) not in TRAINING_EXPECTED_CLASSES})
    if unknown_classes:
        warnings.append(f"Additional classes detected: {', '.join(unknown_classes)}.")

    splits: dict[str, SplitInspection] = {}
    for split_name in ("train", "val", "test"):
        source, refs, image_paths, split_warnings, split_errors = _discover_dataset_split(folder, split_name, yaml_payload, yaml_path)
        split = _inspect_split(
            dataset_root=folder,
            split_name=split_name,
            source=source,
            refs=refs,
            image_paths=image_paths,
            class_count=len(classes),
        )
        split.warnings = split_warnings + split.warnings
        split.errors = split_errors + split.errors
        splits[split_name] = split

    train_split = splits["train"]
    val_split = splits["val"]
    all_images = sum(split.images for split in splits.values())
    if all_images == 0:
        errors.append("No images were found in a recognizable YOLO dataset structure.")
    if train_split.images == 0:
        errors.append("No training images were found.")
    if train_split.labeled_images == 0 or train_split.instances == 0:
        errors.append(
            non_yolo_dataset_hint
            if non_yolo_dataset_hint and non_yolo_dataset_hint not in errors
            else "Training split has no usable labeled detections."
        )
    if train_split.errors:
        errors.extend(train_split.errors[:5])
    if val_split.errors:
        errors.extend(val_split.errors[:5])
    if splits["test"].errors:
        errors.extend(splits["test"].errors[:5])

    if train_split.warnings:
        warnings.extend(train_split.warnings)
    if val_split.warnings:
        warnings.extend(val_split.warnings)
    if splits["test"].warnings:
        warnings.extend(splits["test"].warnings)

    if val_split.images == 0 and train_split.images > 1:
        warnings.append("Validation split is missing; the app will generate a deterministic validation file list inside the training run.")
        validation_strategy = "generate_from_train"
    elif val_split.images == 0 and train_split.images == 1:
        warnings.append("Validation split is missing and train has one image; the app will reuse that image for validation in this run.")
        validation_strategy = "reuse_train_single_image"
    else:
        validation_strategy = "existing_split"

    can_start = not errors
    return DatasetInspection(
        dataset_path=folder,
        yaml_path=yaml_path,
        yaml_payload=yaml_payload,
        classes=classes,
        classes_source=classes_source,
        splits=splits,
        warnings=warnings,
        errors=errors,
        suggested_validation_strategy=validation_strategy,
        can_start=can_start,
        class_mapping=class_mapping,
    )


def scan_training_dataset_path(dataset_path: Path) -> dict[str, Any]:
    return inspect_training_dataset(dataset_path).to_dict()


def _write_image_list(path: Path, image_paths: list[Path]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(item.resolve()) for item in image_paths]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return str(path.resolve())


def prepare_training_run_inputs(dataset_path: Path, run_dir: Path) -> dict[str, Any]:
    inspection = inspect_training_dataset(dataset_path)
    if inspection.errors:
        raise HTTPException(status_code=400, detail=inspection.errors[0])

    run_dir.mkdir(parents=True, exist_ok=True)
    scan_path = run_dir / SCAN_FILENAME
    scan_payload = inspection.to_dict()
    scan_path.write_text(json.dumps(scan_payload, indent=2), encoding="utf-8")

    train_images = list(inspection.splits["train"].image_paths)
    val_images = list(inspection.splits["val"].image_paths)
    test_images = list(inspection.splits["test"].image_paths)

    generated_lists: dict[str, str] = {}
    validation_strategy = inspection.suggested_validation_strategy
    if not val_images:
        deduped_train = sorted({str(path.resolve()): path.resolve() for path in train_images}.values(), key=lambda item: str(item))
        if len(deduped_train) <= 1:
            train_images = deduped_train
            val_images = deduped_train
        else:
            val_count = max(1, int(round(len(deduped_train) * 0.2)))
            if val_count >= len(deduped_train):
                val_count = 1
            val_images = deduped_train[-val_count:]
            train_images = deduped_train[:-val_count] or deduped_train[:1]

    generated_lists["train"] = _write_image_list(run_dir / "splits" / "train.txt", train_images)
    generated_lists["val"] = _write_image_list(run_dir / "splits" / "val.txt", val_images)
    if test_images:
        generated_lists["test"] = _write_image_list(run_dir / "splits" / "test.txt", test_images)

    runtime_yaml_path = run_dir / RUNTIME_DATASET_FILENAME
    runtime_yaml = {
        "path": str(dataset_path.expanduser().resolve()),
        "train": generated_lists["train"],
        "val": generated_lists["val"],
        "names": {index: name for index, name in enumerate(inspection.classes)},
    }
    if "test" in generated_lists:
        runtime_yaml["test"] = generated_lists["test"]
    if yaml is None:
        raise HTTPException(status_code=500, detail="PyYAML is required to generate the runtime dataset manifest.")
    runtime_yaml_path.write_text(yaml.safe_dump(runtime_yaml, sort_keys=False), encoding="utf-8")

    return {
        "scan": scan_payload,
        "dataset_scan_path": str(scan_path.resolve()),
        "generated_dataset_yaml": str(runtime_yaml_path.resolve()),
        "generated_split_lists": generated_lists,
        "validation_strategy": validation_strategy,
    }


def collect_training_artifacts(run_dir: Path) -> dict[str, Any]:
    weights_dir = run_dir / "weights"
    yolo_train_dir = run_dir / "yolo_output" / "train"
    plots = sorted(
        str(path.resolve())
        for path in yolo_train_dir.glob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}
    )
    return {
        "config": str((run_dir / "config.json").resolve()),
        "dataset_scan": str((run_dir / SCAN_FILENAME).resolve()) if (run_dir / SCAN_FILENAME).exists() else None,
        "generated_dataset_yaml": str((run_dir / RUNTIME_DATASET_FILENAME).resolve()) if (run_dir / RUNTIME_DATASET_FILENAME).exists() else None,
        "training_provenance": str((run_dir / PROVENANCE_FILENAME).resolve()) if (run_dir / PROVENANCE_FILENAME).exists() else None,
        "training_analysis": str((run_dir / TRAINING_ANALYSIS_ARTIFACT_FILENAME).resolve()) if (run_dir / TRAINING_ANALYSIS_ARTIFACT_FILENAME).exists() else None,
        "train_log": str((run_dir / "train.log").resolve()) if (run_dir / "train.log").exists() else None,
        "progress": str((run_dir / "progress.json").resolve()) if (run_dir / "progress.json").exists() else None,
        "summary": str((run_dir / SUMMARY_FILENAME).resolve()) if (run_dir / SUMMARY_FILENAME).exists() else None,
        "weights_dir": str(weights_dir.resolve()) if weights_dir.exists() else None,
        "best_checkpoint": str((weights_dir / "best.pt").resolve()) if (weights_dir / "best.pt").exists() else None,
        "results_csv": str((yolo_train_dir / "results.csv").resolve()) if (yolo_train_dir / "results.csv").exists() else None,
        "args_yaml": str((yolo_train_dir / "args.yaml").resolve()) if (yolo_train_dir / "args.yaml").exists() else None,
        "plots": plots,
    }
