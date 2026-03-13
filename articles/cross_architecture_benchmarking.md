# Cross-Architecture Benchmarking in Sports Computer Vision: Comparing the Incomparable

## How We Built a Unified Evaluation Framework for Fundamentally Different Model Architectures — and Why Nobody Has Done This Before

---

*A technical deep-dive into the challenges of building an apples-to-apples benchmarking system across CNN-based detectors, transformer-based pipelines, and hybrid architectures for football video analysis — and the surprising possibility that enough cross-model data could let us evaluate the evaluation systems themselves.*

---

### The Problem Nobody Talks About

If you've ever tried to compare a YOLO model's output to an RT-DETR pipeline's output, you already know the pain. The numbers look similar — both report mAP, both produce bounding boxes, both claim real-time performance. But underneath, these architectures inhabit entirely different universes.

YOLO (You Only Look Once) is a single-stage, anchor-free, CNN-based detector that produces dense predictions and relies on Non-Maximum Suppression (NMS) to collapse redundant detections into final outputs. RT-DETR (Real-Time Detection Transformer) is an end-to-end transformer-based detector that uses learned object queries, bipartite matching, and produces a fixed set of predictions with no NMS required. SoccerNet's TrackLab/sn-gamestate baseline is an entirely different beast: a modular pipeline that bundles detection, tracking, re-identification, camera calibration, team identification, role classification, and jersey OCR into a single hydra-headed system.

These are not three flavors of the same thing. They are three fundamentally different computational paradigms that happen to accept video frames as input and produce structured scene descriptions as output. Comparing their results is not a matter of running them on the same dataset and sorting by AP. It is an unsolved research problem.

We are building a system that attempts to solve it. And after extensive research, we believe that what we are attempting — and the second-order insight it enables — has not been done before.

---

### What We're Actually Building

The [Football Tactical Workbench](https://github.com/DMontgomery40/football-tactical-workbench) is an open-source platform for football (soccer) video analysis. At its core is **Benchmark Lab**, a multi-suite, capability-aware benchmarking system that evaluates models across ten distinct benchmark suites spanning seven task families:

| Suite ID | Task Family | Primary Metric | Protocol |
|---|---|---|---|
| `det.roles_quick_v1` | Detection | AP@[.50:.95] | COCO Detection |
| `det.ball_quick_v1` | Detection | AP (ball) @[.50:.95] | COCO Detection |
| `loc.synloc_quick_v1` | Localization | mAP-LocSim | SynLoc |
| `spot.team_bas_quick_v1` | Event Spotting | Team mAP@1 | Team Spotting |
| `calib.sn_calib_medium_v1` | Calibration | Completeness × JaC@5 | SN-Calibration |
| `track.sn_tracking_medium_v1` | Tracking | HOTA | SN-Tracking |
| `spot.pcbas_medium_v1` | Spotting | F1@15 | PCBAS |
| `gsr.medium_v1` | Game State | GS-HOTA | Gamestate |
| `gsr.long_v1` | Game State | GS-HOTA | Gamestate |
| `ops.clip_review_v1` | Operational | FPS | Operational |

Each suite enforces its own evaluation protocol, its own metric family, and its own dataset contract. A single benchmark run produces a **suite × recipe matrix** — every model configuration ("recipe") is evaluated against every compatible suite, producing a dense grid of results.

The recipes themselves represent radically different computational strategies:

- **Detector recipes** (`detector:soccana`, `detector:<custom>`) — swap a single YOLO-family detector into the classic pipeline, keeping ByteTrack or Hybrid ReID for tracking and a keypoint model for calibration.
- **Tracking recipes** (`tracker:soccana+bytetrack+soccana_keypoint`) — compose a detector, tracker, and keypoint model into a full tracking pipeline with separable components.
- **Pipeline recipes** (`pipeline:soccermaster`) — use a monolithic, bundled architecture (SoccerMaster's unified backbone) that performs detection, tracking, calibration, and team identification as a single forward pass.
- **External pipeline recipes** (`pipeline:sn-gamestate-tracklab`) — invoke an entirely separate framework (TrackLab + sn-gamestate) with its own Python environment, its own dependency tree (`mmocr==1.0.1`, `sn-trackeval`), and its own runtime profile.

This is where the comparison problem becomes genuinely hard.

---

### Challenge 1: Confidence Score Incommensurability

The most insidious problem in cross-architecture benchmarking is that **confidence scores do not mean the same thing across architectures**.

A YOLO model's confidence score is the product of an objectness score and a class probability, filtered through NMS. An RT-DETR model's confidence is a learned query-level score that emerges from the decoder's cross-attention with the encoder's multi-scale feature representation — there is no objectness/classification decomposition, and no NMS. A two-stage detector like Faster R-CNN produces confidences from a Region Proposal Network followed by a classification head.

Research confirms this is not an abstract concern. Popordanoska et al. (WACV 2024) showed that **detection calibration varies significantly by architecture**: RetinaNet and FCOS (anchor-free, focal-loss-based) exhibit lower calibration error than Faster R-CNN. Gilg et al. (WACV 2024) demonstrated that IoU-aware calibration — adjusting confidence as a function of both overlap and initial score — can replace NMS entirely, but the calibration curve shape varies from Gaussian (high confidence) to linear decay (low confidence).

What this means in practice: when our benchmark system sets `player_conf=0.25` and `ball_conf=0.20` as detection thresholds (as we do in `BENCHMARK_RUNTIME_PROFILE`), those thresholds have **different semantic meanings** depending on which architecture produced them. A 0.25 confidence from YOLO is a fundamentally different statement about detection certainty than a 0.25 from RT-DETR. Simply applying the same threshold and computing AP produces numbers that are technically comparable but epistemologically suspect.

Our current mitigation approach is a combination of:

1. **Per-recipe class mapping** — each recipe carries a `class_mapping` that translates the model's internal class vocabulary to the benchmark's canonical categories using fuzzy alias matching (`ball` ↔ `football` ↔ `soccer ball` ↔ `sports ball`).
2. **Low-threshold evaluation** — for COCO detection suites, we run inference at `conf=0.001` with `iou=0.7` to capture the full precision-recall curve rather than a single operating point, letting the COCO evaluation protocol (which sweeps across recall thresholds) handle the comparison.
3. **Protocol-specific normalization** — each evaluation protocol (COCO, HOTA, GS-HOTA, SynLoc, etc.) defines its own matching and scoring logic, so the raw confidence scores are consumed differently by each evaluator.

But this is a band-aid. The deeper problem remains: confidence distributions are architecture-dependent, and no post-hoc normalization can fully decouple a model's detection quality from its confidence calibration.

---

### Challenge 2: Output Space Heterogeneity

Different architectures don't just produce different confidence scores — they produce **structurally different outputs**.

A separable detector recipe produces a flat list of bounding boxes with class labels and confidences. A tracking recipe produces that list plus temporal identity assignments (track IDs). A pipeline recipe like SoccerMaster produces detections, tracks, calibration homographies, and team assignments as a single atomic output. The sn-gamestate/TrackLab pipeline produces all of the above plus re-identification embeddings, role classifications, and jersey number OCR results.

Our system handles this through a **capability model**. Each recipe declares its capabilities:

```python
capabilities = {
    "detection": True,
    "tracking": True,
    "reid": True,
    "calibration": True,
    "team_id": True,
    "role_id": True,
    "jersey_ocr": True,
    "event_spotting": False,
}
```

Each benchmark suite declares its **required capabilities**:

```json
{
    "id": "gsr.medium_v1",
    "required_capabilities": ["detection", "tracking", "calibration"],
    "primary_metric": "gs_hota"
}
```

The orchestrator only evaluates recipe × suite pairs where the recipe satisfies the suite's capability requirements. A detection-only recipe can run the COCO detection suites but not the HOTA tracking suite. A bundled pipeline recipe can run everything it declares support for.

This is the right architectural decision, but it creates a different problem: **the comparison matrix is sparse**. Detection-only recipes can't be compared on tracking metrics. Bundled pipelines can't have their detection component isolated for a clean detection-only comparison. You end up comparing apples to apple orchards.

---

### Challenge 3: Runtime Binding and Environment Isolation

Perhaps the most practically challenging aspect is that different architectures require **different runtime environments**.

Our system defines multiple runtime profiles:

| Runtime Key | Description |
|---|---|
| `backend_default` | Standard Python 3.11+ with Ultralytics, PyTorch, pycocotools |
| `sn_calibration_legacy` | SoccerNet calibration evaluator with specific NumPy/OpenCV versions |
| `tracklab_gamestate_py39_np1` | TrackLab/sn-gamestate requiring Python 3.9, NumPy <2, mmocr==1.0.1 |
| `modern_action_spotting` | Action spotting with team-aware evaluation |
| `footpass_eval` | FOOTPASS play-by-play evaluation |

The sn-gamestate/TrackLab pipeline literally cannot run in the same Python environment as the main application. It requires Python 3.9, NumPy <2, and `mmocr==1.0.1` — a pinned dependency from 2023 that conflicts with virtually every modern ML library. Our system handles this through `external_cli.py`, which spawns evaluators as subprocesses with their own environments and communicates results via JSON.

This is not merely an engineering inconvenience. It means that **timing comparisons are inherently unfair**. A YOLO model running in-process with GPU tensor operations gets different latency characteristics than a TrackLab pipeline spawned as a subprocess, loading its own model weights, and writing results to disk. Our `frames_per_second` and `avg_image_latency_ms` metrics capture wall-clock time, but the overhead structure is categorically different.

---

### Challenge 4: Prediction Export Translation

Each evaluation protocol expects predictions in a specific format. COCO wants `[image_id, category_id, bbox, score]` tuples. The SoccerNet tracking evaluator expects a ZIP archive with MOT-format text files. The calibration evaluator expects per-frame homography JSON files. The gamestate evaluator expects TrackLab state files.

Our `prediction_exports.py` module (over 800 lines) is essentially a universal prediction translator. It takes the raw output of any recipe and converts it into the format expected by each evaluator:

- **COCO detection**: YOLO `xyxy` coordinates → COCO `[x, y, w, h]` with class ID remapping through the alias system
- **SynLoc localization**: detection + calibration → world-coordinate player positions using homography projection
- **Calibration**: keypoint detections → per-frame extremity annotations in the SoccerNet camera calibration format
- **Tracking**: detection sequences → MOT-format track files packaged into evaluator-compatible ZIP archives
- **Gamestate**: full pipeline outputs → TrackLab tracker state files compatible with GS-HOTA evaluation

Each of these translations involves assumptions. When we project detections through a homography to get world coordinates for SynLoc evaluation, the quality of the homography affects the localization score — but the homography came from the same pipeline. When we convert YOLO detections to MOT tracks for the tracking evaluator, the track quality depends on both the detector and the tracker, and there's no way to decompose their individual contributions from the HOTA score.

---

### Challenge 5: Metric Incommensurability Across Task Families

Even when we successfully run all suite × recipe combinations, we face the fundamental question: **how do you compare a model that scores 0.72 AP@[.50:.95] on detection, 0.45 HOTA on tracking, and 0.83 Completeness × JaC@5 on calibration against a model that scores 0.68 AP, 0.52 HOTA, and 0.79 on calibration?**

These metrics live in different mathematical spaces:

- **AP@[.50:.95]** (COCO Average Precision) — the area under the precision-recall curve, averaged across IoU thresholds from 0.50 to 0.95 in steps of 0.05. Higher is better. Range: [0, 1].
- **HOTA** (Higher Order Tracking Accuracy) — the geometric mean of detection accuracy (DetA) and association accuracy (AssA), computed across localization thresholds. It decomposes into `HOTA = sqrt(DetA × AssA)`, allowing separate evaluation of detection and identity association.
- **GS-HOTA** (Game State HOTA) — extends HOTA to the game-state reconstruction task by incorporating calibration-projected world coordinates and team/role assignments.
- **mAP-LocSim** (SynLoc) — mean average precision using localization similarity in world coordinates, measured in meters rather than pixel-space IoU.
- **Completeness × JaC@5** — the product of annotation completeness (fraction of pitch lines successfully detected) and the Jaccard index at a 5-pixel threshold for camera calibration accuracy.
- **F1@15** — F1 score at a 15-second temporal tolerance for action spotting.
- **Team mAP@1** — mean average precision at 1-second temporal tolerance with team-side attribution.

There is no principled way to reduce these to a single scalar. Any weighted combination is arbitrary. Any ranking is metric-dependent. This is not a solvable problem in the traditional sense — it's a fundamental limitation of multi-task evaluation.

Our approach is to **preserve the full matrix** and let the user explore it through the Benchmark Lab UI, rather than collapsing it into a single leaderboard. We provide per-suite rankings, cross-suite comparison views, and raw metric access. But this transparency comes at the cost of simple answers.

---

### The Second-Order Insight: Evaluating the Evaluations

Here is where this work enters genuinely novel territory.

If you run enough models across enough evaluation suites — say, our current set plus 10 additional tracker architectures (DeepSORT, OC-SORT, BoT-SORT, StrongSORT, etc.), several dedicated ball trackers (TrackNet, MonoTrack), multiple calibration approaches (NBJW, PnLCalib, Broadtrack), and various team identification methods (CLIP-based, OSNet-based, color histogram) — you accumulate a dense matrix of results. Each row is a model configuration. Each column is a metric from a specific evaluation protocol.

This matrix contains information not just about the models, but about the **evaluation systems themselves**.

Consider: if two evaluation suites that claim to measure "detection quality" produce rankings that are highly correlated across a diverse set of architectures, that's evidence that both suites are measuring something real and consistent. If they produce rankings that diverge — say, a DETR-based model ranks highly on one suite but poorly on another, while CNN-based models show the opposite pattern — that divergence tells us something about the biases embedded in the evaluation protocols.

This is **meta-evaluation**: using cross-architecture benchmarking data to assess the reliability, consistency, and biases of the evaluation datasets and metrics themselves.

#### What Meta-Evaluation Could Reveal

1. **Annotation bias detection**: If transformer-based models consistently score higher on one dataset but lower on another for the same task, the discrepancy may reveal systematic differences in annotation style. COCO-ReM (Raffel et al., 2024) demonstrated this: they found that models producing visually sharper masks scored higher on cleaned annotations, suggesting that the original COCO annotations had systematic imprecision that penalized certain architectures.

2. **Metric sensitivity analysis**: By comparing how different metrics rank the same set of models, we can identify which metrics are most sensitive to architectural differences versus actual performance differences. If GS-HOTA and HOTA produce identical rankings for detection-focused models but divergent rankings when calibration-dependent models are included, that tells us GS-HOTA is capturing calibration quality that HOTA ignores (by design) — but it also means GS-HOTA rankings are confounded by calibration quality for models that don't control their own calibration.

3. **Dataset difficulty characterization**: Different architectures have different failure modes. CNNs struggle with occlusion; transformers struggle with small objects at low resolution; hybrid models may have blind spots at the seams of their component integration. By profiling how each architecture fails on each dataset, we can characterize dataset difficulty not as a single scalar but as a per-architecture difficulty vector.

4. **Evaluation protocol sensitivity**: The choice of IoU threshold, temporal tolerance, spatial resolution, and matching algorithm can change not just the absolute scores but the relative rankings of models. By sweeping these parameters across a large model population, we can identify the regions of protocol space where rankings are stable versus where they flip.

#### Why This Hasn't Been Done

After extensive literature review, we found no published work that:

1. Builds a unified benchmarking framework comparing CNN-based detectors, transformer-based detectors, and hybrid pipelines **specifically in sports video analysis** across detection, tracking, calibration, and game-state reconstruction simultaneously.

2. Uses the resulting cross-architecture evaluation data to **meta-evaluate the evaluation protocols themselves**, assessing the consistency, bias, and discriminative power of different benchmark suites and metrics.

The closest existing work falls into several categories:

- **Single-architecture comparison studies** — papers like "DETRs Beat YOLOs on Real-time Object Detection" (Zhao et al., CVPR 2024) compare architectures on standard benchmarks (COCO) but don't address the evaluation methodology question.
- **Benchmark cleaning efforts** — "Benchmarking Object Detectors with COCO: A New Path Forward" (arXiv 2024) examines annotation quality but doesn't use cross-architecture data to detect annotation biases.
- **Automated model evaluation** — PCR (arXiv 2508.12082) proposes label-free model evaluation but doesn't compare evaluation protocols against each other.
- **SoccerNet challenge results** — the annual SoccerNet challenges compare submitted models within each task (tracking, calibration, etc.) but don't analyze cross-task correlations or use the data for meta-evaluation.
- **Universal calibration benchmarking** — ProCC (arXiv 2404.09807) proposes a camera-model-agnostic calibration benchmark protocol for sports but doesn't extend to cross-task meta-evaluation.
- **Metric critiques** — OVDEval (arXiv 2308.13177) identified problems with the AP metric for open-vocabulary detection and proposed NMS-AP as an alternative, while ARCADE (arXiv 2508.04102) showed that small protocol decisions can shift scores by 30% and reorder rankings. But neither used cross-architecture data to systematically compare evaluation systems.

The gap exists because it requires solving the first problem (unified cross-architecture benchmarking) before you can even begin the second (meta-evaluation). And the first problem is hard enough that most researchers stop there.

---

### Is This Even Possible Without Hand-Waving?

Let's be honest about where the technical rigor breaks down.

**What we can do rigorously:**

- Compare models within the same evaluation protocol on the same dataset using the same metric. AP@[.50:.95] on `det.roles_quick_v1` is a well-defined comparison, even across architectures, as long as we use low-threshold full-curve evaluation.
- Measure rank correlations between different suites. If Suite A and Suite B produce Kendall's τ > 0.8 across 20+ model configurations, that's statistically meaningful evidence of evaluation consistency.
- Identify systematic architecture-metric interactions using mixed-effects models or stratified analysis. If transformer models consistently gain +X on Suite A relative to Suite B compared to CNN models, that interaction effect is measurable.

**What requires careful qualification:**

- Any comparison of absolute metric values across different task families. HOTA = 0.45 and AP = 0.72 are incommensurable. We can compare how models rank, but not how the numbers relate.
- Latency comparisons across runtime environments. In-process GPU inference versus subprocess invocation versus external pipeline execution have fundamentally different overhead structures.
- Attribution of performance to architectural components in bundled pipelines. When SoccerMaster produces a GS-HOTA score, we cannot decompose how much came from its detector versus its tracker versus its calibrator.

**What would be hand-waving:**

- Claiming a single "best model" across all tasks without specifying the weighting of task importance.
- Treating meta-evaluation results as ground truth about dataset quality without independent validation.
- Extrapolating calibration-metric relationships observed on football data to other sports or domains.

We are committed to staying on the rigorous side of these boundaries.

---

### The Technical Architecture

For those interested in the engineering, here's how the system works:

#### The Orchestrator Pattern

`BenchmarkOrchestrator` manages the full lifecycle:

```
create_benchmark(suite_ids, recipe_ids)
    → spawn background thread
    → for each suite:
        → build_suite_dataset_state()
        → for each recipe:
            → check capability compatibility
            → check recipe availability
            → check dataset readiness
            → prepare_prediction_exports()  # translate predictions
            → run_suite_evaluation()        # protocol-specific eval
            → persist result to suite_results/{suite_id}/{recipe_id}/
    → persist final benchmark.json
```

#### The Capability Gate

Before any evaluation, the system checks:

```python
required = suite.get("required_capabilities")   # e.g., ["detection", "tracking"]
capabilities = recipe.get("capabilities")        # e.g., {"detection": True, "tracking": True, ...}
if not all(capabilities.get(key) for key in required):
    return "not_supported"
```

This prevents meaningless comparisons (you can't evaluate a detection-only recipe on tracking metrics) while allowing maximal coverage (a pipeline that supports everything gets evaluated on everything).

#### The Protocol Dispatch

Each evaluation protocol has its own runner:

```python
PROTOCOL_RUNNERS = {
    "coco_detection": evaluate_coco_detection,    # pycocotools
    "synloc": evaluate_synloc,                     # Spiideo sskit
    "team_spotting": evaluate_team_spotting,        # SoccerNet BAS
    "calibration": evaluate_calibration,            # sn-calibration
    "tracking": evaluate_tracking,                  # sn-tracking HOTA
    "pcbas": evaluate_pcbas,                        # FOOTPASS
    "gamestate": evaluate_gamestate,                # TrackLab GS-HOTA
    "operational": evaluate_operational,            # wall-clock review
}
```

Each runner handles its own prediction format translation, evaluator invocation, and metric extraction. The runners are deliberately not abstracted into a common interface beyond `(suite, recipe, dataset_root, artifacts_dir, benchmark_id) → {metrics, artifacts, raw_result}`. The implementation details are too different to benefit from a shared abstraction — trying to force them into one would either lose critical protocol-specific information or result in a lowest-common-denominator interface that helps no one.

#### The Metric Schema

Every metric value is wrapped in a standard envelope:

```python
{
    "label": "AP@[.50:.95]",
    "value": 0.7234,
    "display_value": "0.7234",
    "unit": "",
    "sort_value": 0.7234,
    "is_na": False,
}
```

This allows the UI to display, sort, and filter metrics uniformly even though their mathematical meanings are entirely different. The `is_na` field is critical: when a recipe doesn't support a suite, every metric cell is filled with `N/A` rather than left empty, preserving the matrix structure.

---

### What Comes Next

We are currently working toward three milestones:

1. **Expanding the recipe catalog** — adding RT-DETR, RF-DETR, YOLOv12, and custom fine-tuned variants to enable genuine cross-architecture comparison within the existing suite framework.

2. **Dense matrix population** — running enough recipe × suite combinations to produce statistically meaningful meta-evaluation data. Our target is 20+ distinct model configurations evaluated across all compatible suites.

3. **Meta-evaluation analysis toolkit** — building the statistical analysis layer that computes rank correlations, architecture-metric interaction effects, and dataset difficulty profiles from the accumulated benchmark data.

If we succeed, the result won't just be a better benchmarking system for football video analysis. It will be a template for how to think about evaluation in any domain where fundamentally different model architectures compete on the same tasks — and a demonstration that the evaluation systems themselves are legitimate objects of scientific inquiry.

---

### References and Related Work

- Zhao, Y. et al. "DETRs Beat YOLOs on Real-time Object Detection." CVPR 2024. [(paper)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf)
- "Benchmarking Object Detectors with COCO: A New Path Forward." arXiv 2403.18819, 2024. [(paper)](https://arxiv.org/abs/2403.18819)
- "Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability." arXiv 2508.12082, 2025. [(paper)](https://arxiv.org/abs/2508.12082)
- "A Universal Protocol to Benchmark Camera Calibration for Sports." arXiv 2404.09807, 2024. [(paper)](https://arxiv.org/html/2404.09807v1)
- "SoccerNet 2025 Challenges Results." arXiv 2508.19182, 2025. [(paper)](https://arxiv.org/html/2508.19182v1)
- Popordanoska, T. et al. "Beyond Classification: Definition and Density-Based Estimation of Calibration in Object Detection." WACV 2024. [(paper)](https://openaccess.thecvf.com/content/WACV2024/papers/Popordanoska_Beyond_Classification_Definition_and_Density-Based_Estimation_of_Calibration_in_Object_WACV_2024_paper.pdf)
- Gilg, J. et al. "Do We Still Need Non-Maximum Suppression? Accurate Confidence Estimates and Implications for Object Detection." WACV 2024. [(paper)](https://openaccess.thecvf.com/content/WACV2024/papers/Gilg_Do_We_Still_Need_Non-Maximum_Suppression_Accurate_Confidence_Estimates_and_WACV_2024_paper.pdf)
- "How to Evaluate the Generalization of Detection? A Benchmark for Comprehensive Open-Vocabulary Detection." arXiv 2308.13177, 2023. [(paper)](https://arxiv.org/html/2308.13177v2)
- "AR as an Evaluation Playground: Bridging Metric and Visual Perception of CV Models." arXiv 2508.04102, 2026. [(paper)](https://arxiv.org/html/2508.04102)
- "Replication Study and Benchmarking of Real-Time Object Detection Models." arXiv 2405.06911, 2024. [(paper)](https://arxiv.org/abs/2405.06911)
- "Object Detection Based on CNN and Vision-Transformer: A Survey." IET Computer Vision, 2025. [(paper)](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.70028)
- "Object Recognition Datasets and Challenges: A Review." arXiv 2507.22361, 2025. [(paper)](https://arxiv.org/html/2507.22361v1)
- "SDQM: Synthetic Data Quality Metric for Object Detection Dataset Evaluation." arXiv 2510.06596, 2025. [(paper)](https://arxiv.org/abs/2510.06596)
- SoccerNet Calibration Repository. [(GitHub)](https://github.com/SoccerNet/sn-calibration)
- SoccerNet Tracking Repository. [(GitHub)](https://github.com/SoccerNet/sn-tracking)

---

*David Montgomery — Football Tactical Workbench — March 2026*

*This article describes work in progress on the [football-tactical-workbench](https://github.com/DMontgomery40/football-tactical-workbench) project. The benchmarking system, meta-evaluation framework, and all evaluation code are open source.*
