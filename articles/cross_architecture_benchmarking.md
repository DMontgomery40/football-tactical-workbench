# Cross-Architecture Benchmarking in Sports Computer Vision: Comparing the Incomparable

## Why Fairly Comparing a CNN Detector to a Vision Transformer Is an Unsolved Problem — and What Happens When You Try to Do It Across Detection, Tracking, Calibration, and Game-State Reconstruction Simultaneously

---

*A technical deep-dive into the eleven known confounds that make cross-architecture model comparison unreliable, how we're building a system that confronts them head-on for football video analysis, and the second-order insight that emerges when you accumulate enough cross-model data to evaluate the evaluation systems themselves.*

---

### Comparing a CNN to a CLIP Transformer: A Parable

Consider a seemingly simple question: is YOLO better than CLIP at detecting football players?

The question is malformed. YOLO is a single-stage, anchor-free, CNN-based object detector that directly predicts bounding boxes in a single forward pass through convolutional layers with strong locality bias. CLIP is a contrastive vision-language model that learns a shared embedding space between images and text — it doesn't natively produce bounding boxes at all. YOLO was trained on labeled bounding box annotations. CLIP was trained on 400 million image-text pairs from the internet.

You cannot compare these systems on mAP without first deciding: do you wrap CLIP in a detection head (OWL-ViT, Grounding DINO)? Do you fine-tune it on your domain, or evaluate zero-shot? Do you account for the fact that CLIP brings encyclopedic pre-training knowledge while YOLO uses only your labeled dataset? Do you measure at the confidence threshold where DETR-family models have flat score distributions while CNN detectors drop steeply (Wenkel et al., [Sensors 2021](https://www.mdpi.com/1424-8220/21/13/4350))?

This is not an exotic edge case. It is the **central methodological challenge** of modern computer vision evaluation, and it manifests every time you try to compare any two architectures built on fundamentally different computational paradigms.

Recent work has shown this challenge cuts deeper than most researchers acknowledge. The [Battle of the Backbones](https://encord.com/blog/top-computer-vision-models/) benchmark attempted to normalize across vision-language models, self-supervised models, and supervised models across 1,500+ training runs — and found that "convolutional neural networks pretrained in a supervised fashion on large training sets still perform best on most tasks." But this finding itself depends on which tasks, which metrics, and which evaluation protocols you choose. As one [multi-model comparison study](https://aimultiple.com/large-vision-models) put it: GPT-4o, YOLOv8n, and DETR "showed different levels of accuracy and speed because they are built for different purposes and process visual information in distinct ways."

We are building a system that does not shy away from this. We compare YOLO-family detectors against transformer-based unified pipelines against modular external frameworks — across detection, tracking, calibration, and full game-state reconstruction — and we do it knowing that no comparison is perfectly fair. What we've learned is that the unfairness itself is informative.

---

### The Eleven Confounds

The computer vision literature has identified at least eleven distinct confounds that undermine cross-architecture comparison. We encountered all of them. To our knowledge, no prior system has confronted all eleven simultaneously in a single benchmarking framework.

#### 1. Training Recipe Confounds

Zhang et al.'s landmark ATSS paper ([CVPR 2020](https://arxiv.org/abs/1912.02424)) demonstrated that the "essential difference" between anchor-based and anchor-free detectors is actually the definition of positive and negative training samples — not the architecture itself. When using the same sample selection strategy, the performance gap vanishes.

This finding is devastating for architectural comparison. It means that what looks like an architectural advantage may be a training recipe advantage. Zoph et al. ([ECCV 2020](https://arxiv.org/abs/1906.11172)) showed that learned data augmentation on ResNet-50 improved RetinaNet by +2.3 mAP — exceeding the +2.1 mAP gain from switching to a larger ResNet-101 backbone. Wightman et al.'s "ResNet Strikes Back" ([arXiv 2021](https://arxiv.org/pdf/2110.00476)) showed a vanilla ResNet-50 reaching 80.4% ImageNet accuracy with training recipe improvements alone, closing most of the gap with "newer" architectures.

In our system, we cannot control training recipes: models arrive pre-trained from different sources with different augmentation pipelines, learning rate schedules, and training set compositions. Our `import_local_checkpoint()` and `import_hf_checkpoint()` functions accept any Ultralytics-compatible checkpoint, but the training history is opaque.

#### 2. Confidence Score Incommensurability

A YOLO model's confidence score is the product of an objectness score and a class probability, filtered through NMS. RT-DETR's confidence is a learned query-level score that emerges from the decoder's cross-attention — no objectness decomposition, no NMS. A two-stage detector produces confidences from an RPN followed by a classification head. These numbers are not on the same scale.

Kuppers et al. ([CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.pdf)) showed that calibration error depends on architecture, object location, and object size. Munir et al.'s BPC loss ([CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Munir_Bridging_Precision_and_Confidence_A_Train-Time_Loss_for_Calibrating_Object_CVPR_2023_paper.pdf)) found that post-hoc calibration methods designed for classification are sub-optimal for detection. Wenkel et al. ([Sensors 2021](https://www.mdpi.com/1424-8220/21/13/4350)) demonstrated that DETR shows a remarkably flat mAP curve across confidence thresholds while CNN detectors drop sharply — meaning mAP rankings at the standard low-threshold evaluation may not reflect deployment-relevant rankings at operationally useful confidence levels.

When our benchmark system sets `player_conf=0.25` and `ball_conf=0.20`, those thresholds have **different semantic meanings** depending on which architecture produced them. Our mitigation — running COCO evaluation at `conf=0.001` to capture the full precision-recall curve — helps but does not eliminate the problem.

#### 3. NMS vs. End-to-End Detection

YOLO produces thousands of overlapping boxes filtered by NMS. DETR/RT-DETR directly predict a one-to-one object set via bipartite matching. This is not a minor implementation detail — it changes the fundamental output distribution.

Gilg et al. ([WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Gilg_Do_We_Still_Need_Non-Maximum_Suppression_Accurate_Confidence_Estimates_and_WACV_2024_paper.pdf)) proposed IoU-aware calibration as a principled bridge: conditional Beta calibration that implicitly models the probability of each detection being a duplicate, replacing NMS with a calibration-based alternative. This was tested across anchor-free, NMS-based, NMS-free, transformer, and CNN models. Their NMS-Bench framework ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/dcc0ac74ac8b95dc1939804acce0317d-Paper-Conference.pdf)) provides a unified framework for fair comparison of different NMS algorithms — but applying it across architectures that fundamentally disagree about whether NMS should exist at all remains an open question.

#### 4. Convergence Epoch Asymmetry

DETR requires 500 training epochs to converge versus 12-36 for Faster R-CNN (Sun et al., [ICCV 2021](https://arxiv.org/abs/2011.10881)). Conditional DETR ([ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Meng_Conditional_DETR_for_Fast_Training_Convergence_ICCV_2021_paper.pdf)) achieves 6.7x faster convergence, and DN-DETR ([arXiv 2022](https://arxiv.org/pdf/2203.01305)) further accelerates via query denoising — but the asymmetry persists. Compute-normalized comparison is nearly impossible when one architecture needs 10x the training compute to reach its potential.

#### 5. Pre-Training Data Asymmetry

CLIP was trained on 400M+ image-text pairs. YOLO typically starts from ImageNet or scratch. SoccerMaster uses a fine-tuned SiglIP2-large backbone. TrackLab/sn-gamestate uses pre-trained foundation components for re-identification, role classification, and jersey OCR — including CLIP- and OSNet-based features, and even Vision-Language Models like LLaMA-Vision and Qwen2 VL (as seen in [SoccerNet 2025 challenge submissions](https://arxiv.org/html/2508.19182v1)).

This creates massive asymmetry. Fine-tuning a foundation model is almost always more data-efficient than training from scratch, with studies showing up to 15% accuracy improvement and 40% training time reduction. Comparing a foundation-model pipeline's accuracy to a task-specific pipeline's accuracy without acknowledging that the former brings billions of implicit training examples is methodologically unsound.

He et al. ([ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.pdf)) added a wrinkle: training from scratch achieves competitive detection accuracy on COCO without ImageNet pretraining, and Zoph et al. ([NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/27e9661e033a73a6ad8cefcde965c54d-Paper.pdf)) showed that with strong augmentation, pretraining actually *hurts* accuracy by -1.0 AP. The benefit of pretraining depends on augmentation strength, meaning architecture comparisons using different augmentation pipelines are fundamentally confounded.

#### 6. Inductive Bias Mismatch

CNNs embed strong inductive biases: locality (nearby pixels are related) and translation equivariance (patterns are position-independent). Vision Transformers use self-attention for global context but lack built-in spatial priors ([Pattern Recognition 2024](https://www.sciencedirect.com/science/article/abs/pii/S0031320324002619)). The ViTAE paper ([NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/efb76cff97aaf057654ef2f38cd77d73-Paper.pdf)) showed that injecting CNN-like inductive biases into transformers improves performance — confirming that the bias itself, not just the architecture, drives results.

As ConViT ([arXiv 2021](https://arxiv.org/pdf/2103.10697)) put it: "hard inductive biases can greatly improve sample-efficiency but become constraining when dataset size is not an issue." This means any benchmark inherently favors one architecture depending on evaluation distribution: if the benchmark has many small objects, CNNs win; if it requires contextual reasoning, transformers win.

In sports CV, this maps directly: YOLO's locality bias is good for player detection (small, spatially-concentrated objects). Foundation-model pipelines' global attention is good for understanding game context (who has the ball, which team is which, where on the pitch). No single-metric benchmark is neutral.

#### 7. Spatial Bias in Evaluation

Zheng et al. ([IEEE TPAMI 2024](https://arxiv.org/abs/2310.13215)) discovered that object detectors perform unevenly across image zones. Sparse detectors (DETR series, Sparse R-CNN) have Zone Precision variance of 12.9-26.9 — performing well centrally but poorly at borders. Dense one-stage detectors have lower spatial bias. Standard mAP **hides** this architecture-specific spatial weakness. The 96% border zone of the image does not even reach the AP value of the central zone.

This has direct implications for football analysis, where players frequently appear at frame edges during broadcast footage.

#### 8. Annotation Quality Bias

Singh et al.'s COCO-ReM ([ECCV 2024](https://arxiv.org/abs/2403.18819)) found that COCO-2017 contains systematic annotation errors — imprecise boundaries, non-exhaustive annotations, mislabeled masks — and that cleaning these annotations **changes the model ranking**. Query-based models (Mask2Former, OneFormer) score much higher on COCO-ReM than region-based models (ViTDet), meaning COCO-2017's annotation noise was systematically disadvantaging architectures that produce sharper outputs.

If this happens on COCO — the most scrutinized dataset in object detection — it almost certainly happens on sports-specific datasets, where annotation budgets are smaller and domain expertise requirements are higher.

#### 9. Distribution Shift Sensitivity

Mao et al.'s COCO-O ([ICCV 2023](https://arxiv.org/abs/2307.12730)) evaluated 100+ detectors under 6 types of distribution shift and found a 55.7% relative performance drop for Faster R-CNN. Their critical architectural finding: the backbone is the most important component for robustness, and end-to-end transformer design (DETR-style) brings **no robustness enhancement** and may even reduce it. Large-scale foundation models show the greatest leap — but only because of their pre-training, not their architecture.

A [robustness evaluation of open-vocabulary detectors](https://arxiv.org/html/2405.14874v3) comparing OWL-ViT, YOLO-World, and Grounding DINO found that YOLO-World had the highest baseline mAP (39.30) but dropped to 23.42 on out-of-distribution data, while Grounding DINO maintained 34.7 on adversarial subsets. The foundation model's robustness advantage came from pre-training breadth, not architectural superiority.

#### 10. Metric Implementation Inconsistency

Padilla et al. ([IWSSIP 2020](https://ieeexplore.ieee.org/document/9145130/); [MDPI Electronics 2021](https://www.mdpi.com/2079-9292/10/3/279)) documented that there is no consensus on AP computation across the community. Different interpolation methods (11-point vs. all-point) produce different results. Six AP variants exist, and different implementations yield different evaluation outcomes. Their [open-source toolkit](https://github.com/rafaelpadilla/review_object_detection_metrics) supporting 15 metrics and multiple bounding box formats was built specifically to address this chaos.

The OVDEval benchmark ([AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28485)) went further, showing that traditional AP metrics yield "inflated" and "deceptive" results for open-vocabulary detection models, and proposing NMS-AP as a fairer metric.

#### 11. Cross-Framework Numerical Drift

Even with identical models, CPU vs. GPU vs. compiled backends produce different pre-NMS orderings — within 10⁻⁵ tensor-level differences that NMS amplifies into different final detections ([arXiv 2025](https://arxiv.org/pdf/2509.06977)). TENSORSCOPE ([USENIX Security 2023](https://www.usenix.org/system/files/usenixsecurity23-deng-zizhuang.pdf)) found 34 bugs in TensorFlow and 30 in PyTorch through differential testing: equivalent APIs across frameworks behave differently.

In our system, the sn-gamestate/TrackLab pipeline runs in a separate Python 3.9 environment, while the main pipeline runs on Python 3.11+. These are not just different frameworks — they're different runtime universes communicating via JSON over subprocess boundaries.

---

### The CNN-to-CLIP-Transformer Analogy: Does It Hold?

The question "is comparing a CNN detector to a CLIP transformer analogous to what we're doing?" has a nuanced answer, because the analogy holds in most dimensions but breaks in important ways.

**Where it holds strongly:**

| Dimension | CLIP vs. YOLO | Our Pipeline Comparison |
|---|---|---|
| Paradigm mismatch | Representation learning vs. end-to-end detection | Modular pipeline vs. end-to-end pipeline |
| Metric ambiguity | mAP favors YOLO; zero-shot generalization favors CLIP | Detection mAP favors YOLO; GS-HOTA favors TrackLab |
| Pre-training asymmetry | 400M image-text pairs vs. domain-specific labels | Foundation components vs. sport-specific training |
| Speed-accuracy trade-off | YOLO faster; foundation models more robust | YOLO pipeline faster; TrackLab more complete |
| Complementarity | Field moving to YOLO-World (CLIP text encoder + YOLO architecture) | Field moving to composable pipelines that mix approaches |

CLIP is already being used in sports CV: [CLIP-ReIdent](https://arxiv.org/abs/2303.11855) achieved 98.44% mAP on the MMSports 2022 Player Re-Identification challenge. ActionCLIP extends CLIP for zero-shot sports action recognition. [SoccerNet 2025 challenge participants](https://arxiv.org/html/2508.19182v1) used CLIP-based features, OSNet, and Vision Language Models (LLaMA-Vision, Qwen2 VL) for player feature extraction. The CLIP-vs-YOLO comparison is not hypothetical in sports — it's happening.

**Where the analogy partially breaks:**

Comparing CLIP to YOLO is a **model-to-model** comparison. Comparing a YOLO sports pipeline to TrackLab/sn-gamestate is a **system-to-system** comparison involving detection + tracking + re-ID + calibration + jersey OCR. Each pipeline stage introduces its own architectural choices, creating more confounding variables. The sports CV comparison is strictly harder.

**Where it does not hold:**

CLIP and YOLO can both be evaluated on the same images with the same ground truth (bounding boxes on COCO). Sports pipelines may define the output task itself differently — "detect players in frame" vs. "reconstruct full game state on minimap" are **different tasks**, not just different approaches to the same task. The comparison requires first agreeing on what the task *is*, which is a higher-order problem that CLIP-vs-YOLO does not face to the same degree.

This is precisely why our system decomposes evaluation into capability-gated suites: rather than forcing a single comparison, we let each suite define its own task, and compare only where capabilities overlap.

---

### What We're Actually Building

The [Football Tactical Workbench](https://github.com/DMontgomery40/football-tactical-workbench) confronts all eleven confounds through a multi-suite, capability-aware benchmarking system called **Benchmark Lab**. It evaluates models across ten benchmark suites spanning seven task families:

| Suite | Task | Primary Metric | What It Measures |
|---|---|---|---|
| `det.roles_quick_v1` | Detection | AP@[.50:.95] | Four-class player/goalkeeper/ball/referee detection |
| `det.ball_quick_v1` | Detection | AP (ball) | Small-object detection on the hardest sports object |
| `loc.synloc_quick_v1` | Localization | mAP-LocSim | World-coordinate player positioning via homography |
| `spot.team_bas_quick_v1` | Spotting | Team mAP@1 | Ball action spotting with team attribution |
| `calib.sn_calib_medium_v1` | Calibration | Completeness × JaC@5 | Camera calibration accuracy vs. SoccerNet ground truth |
| `track.sn_tracking_medium_v1` | Tracking | HOTA | Multi-object tracking identity preservation |
| `spot.pcbas_medium_v1` | Spotting | F1@15 | Person-centric ball action spotting |
| `gsr.medium_v1` | Game State | GS-HOTA | Full game-state reconstruction |
| `gsr.long_v1` | Game State | GS-HOTA | Full validation-scale game-state reconstruction |
| `ops.clip_review_v1` | Operational | FPS | Non-ground-truth pipeline review |

The recipes being compared represent four fundamentally different computational strategies:

1. **Separable detector recipes** — a single YOLO-family detector swapped into the classic pipeline. CNN-based. End-to-end detection with NMS. Strong locality bias.
2. **Composed tracking recipes** — a detector + tracker + keypoint model assembled from separable components. Hybrid CNN architecture. Detection + association + calibration as three separate inference steps.
3. **Bundled pipeline recipes** (SoccerMaster) — a unified backbone (fine-tuned SiglIP2-large vision transformer) that performs detection, tracking, calibration, and team identification as a single forward pass. Transformer-based foundation model.
4. **External pipeline recipes** (TrackLab/sn-gamestate) — an entirely separate modular framework with its own Python 3.9 environment, CLIP/OSNet-based re-identification, mmocr for jersey OCR, and its own evaluation pipeline. Foundation-model components composed into a modular system.

These span the full spectrum from pure CNN to pure transformer to hybrid to modular-foundation — exactly the comparison space where all eleven confounds are active.

---

### How We Confront the Confounds

We do not claim to solve all eleven. But we address each one explicitly:

**Training recipe (Confound 1)**: We cannot control it, so we document it. Each recipe carries full provenance metadata — checkpoint origin, training run ID, DVC tracking — so that consumers of benchmark results can assess training recipe differences themselves.

**Confidence calibration (Confounds 2-3)**: For COCO suites, we evaluate at `conf=0.001` with `iou=0.7` to capture the full precision-recall curve. The COCO evaluation protocol sweeps across recall thresholds, making confidence calibration less impactful than at a single operating point. For future work, we plan to integrate post-hoc calibration using the [net:cal framework](https://github.com/EFS-OpenSource/calibration-framework), which supports Dependent Beta Calibration for object detection across architectures.

**Output space heterogeneity (Confounds 6-7)**: The capability model ensures we only compare where comparison is meaningful. Each recipe declares what it can do; each suite declares what it needs. The resulting matrix is sparse but honest.

**Runtime isolation (Confound 11)**: Different architectures run in different environments via subprocess boundaries. We measure wall-clock time but flag it as structurally different across runtime profiles. The `external_cli.py` module handles the TrackLab subprocess with its Python 3.9 + NumPy <2 + mmocr==1.0.1 dependency stack.

**Metric heterogeneity (Confound 10)**: Every metric value is wrapped in a standard envelope with `value`, `display_value`, `unit`, `sort_value`, and `is_na`. We preserve the full suite × recipe matrix rather than collapsing it into a single leaderboard. Rankings are per-suite, not global.

**Spatial and annotation bias (Confounds 7-8)**: We use multiple evaluation suites on different datasets — HuggingFace community datasets, official SoccerNet benchmarks, Spiideo SynLoc, FOOTPASS — rather than relying on a single dataset. If an architecture scores well on one but not another, that divergence is information.

---

### The Second-Order Insight: Evaluating the Evaluations

This is where the work enters genuinely novel territory, and where the new research fundamentally changes the framing.

The computer vision community has begun to recognize that **evaluation systems themselves are objects of scientific inquiry**. COCO-ReM showed that cleaning COCO annotations changes model rankings. COCO-O showed that distribution shift affects architectures asymmetrically. Zone Evaluation showed that mAP hides spatial bias. OVDEval showed that traditional AP produces deceptive results for open-vocabulary models. ProCC ([arXiv 2404.09807](https://arxiv.org/html/2404.09807v1)) argued that existing sports camera calibration benchmarks "strongly favor methods estimating homographies" and thereby unfairly disadvantage full 3D camera calibration methods.

Each of these papers evaluates one aspect of one evaluation protocol. None uses cross-architecture data across multiple tasks to systematically compare evaluation protocols against each other.

That is what our system enables.

If you run 20+ model configurations (CNN detectors, transformer detectors, hybrid trackers, foundation-model pipelines) across 10 benchmark suites spanning 7 task families, you get a dense matrix. Each row is an architecture. Each column is a metric from a specific protocol on a specific dataset.

This matrix contains information not just about the models, but about the evaluation systems:

1. **Rank correlation analysis**: If Suite A and Suite B both claim to measure "detection quality" but produce Kendall's τ < 0.5 across 20+ architectures, at least one suite has evaluation bias — or they're measuring genuinely different aspects of "detection quality," which is equally important to know.

2. **Architecture × metric interaction effects**: If transformer models consistently gain +X on Suite A relative to Suite B compared to CNN models (while controlling for absolute performance), that interaction reveals a systematic architecture-dependent bias in one or both suites. The COCO-ReM finding (query-based models gain disproportionately on cleaned annotations) is exactly this kind of interaction — but discovered post-hoc. Our framework can detect these interactions proactively.

3. **Evaluation protocol sensitivity**: By measuring how rank orderings change as a function of IoU threshold, temporal tolerance, spatial resolution, and matching algorithm, we can identify the regions of protocol space where rankings are stable versus where they flip — essentially mapping the fragility of each evaluation protocol.

4. **Dataset difficulty decomposition**: Rather than characterizing dataset difficulty as a single scalar, we can profile it as a per-architecture difficulty vector. A dataset that's "hard" for CNNs but "easy" for transformers is telling us something about the failure modes it tests.

The paper ["Towards Universal Soccer Video Understanding"](https://arxiv.org/html/2412.01820v2) (arXiv 2024) explicitly identified the fragmentation problem we're addressing: "the focus has primarily been on designing specialized models tailored to narrow tasks, leading to a significant gap in compatibility among models. Such fragmentation underscores the need for a unified analytical framework." Our Benchmark Lab is that framework.

---

### The Technical Architecture

The system is designed around three principles: capability-gated comparison, protocol-specific evaluation, and full-matrix preservation.

#### Orchestration

`BenchmarkOrchestrator` manages the lifecycle:

```
create_benchmark(suite_ids, recipe_ids)
    → for each suite × recipe pair:
        → capability gate: does recipe satisfy suite requirements?
        → availability gate: are model weights and dataset present?
        → prepare_prediction_exports(): translate outputs to evaluator format
        → run_suite_evaluation(): invoke protocol-specific evaluator
        → persist result with full provenance
```

#### Capability Gate

```python
required = suite["required_capabilities"]    # e.g., ["detection", "tracking"]
capabilities = recipe["capabilities"]         # e.g., {"detection": True, ...}
if not all(capabilities.get(key) for key in required):
    return "not_supported"  # honest N/A, not a fake zero
```

#### Protocol Dispatch

Eight evaluators, each speaking its own language:

```python
PROTOCOL_RUNNERS = {
    "coco_detection": evaluate_coco_detection,    # pycocotools AP
    "synloc": evaluate_synloc,                     # Spiideo mAP-LocSim
    "team_spotting": evaluate_team_spotting,        # SoccerNet Team BAS
    "calibration": evaluate_calibration,            # SN-Calibration JaC
    "tracking": evaluate_tracking,                  # SN-Tracking HOTA
    "pcbas": evaluate_pcbas,                        # FOOTPASS F1
    "gamestate": evaluate_gamestate,                # TrackLab GS-HOTA
    "operational": evaluate_operational,            # Wall-clock FPS
}
```

#### Prediction Translation

The `prediction_exports.py` module (1,800+ lines) is a universal translator:

- YOLO `xyxy` → COCO `[x, y, w, h]` with class alias matching (`ball` ↔ `football` ↔ `soccer ball`)
- Detection + calibration → world-coordinate positions via homography projection
- Detection sequences → MOT-format track files in evaluator-compatible ZIP archives
- Pipeline outputs → TrackLab tracker state files for GS-HOTA evaluation

Each translation embeds assumptions about how the source architecture's outputs should be interpreted — and each assumption is a potential source of comparison unfairness. We document them rather than hiding them.

---

### Is This Even Possible Without Hand-Waving?

We distinguish three zones of rigor:

**Rigorous:**
- Comparing models within the same evaluation protocol on the same dataset using the same metric, with low-threshold full-curve evaluation.
- Measuring rank correlations between suites across 20+ architectures — Kendall's τ > 0.8 is statistically meaningful evidence of evaluation consistency.
- Identifying architecture × metric interactions using stratified analysis or mixed-effects models.

**Requires qualification:**
- Any comparison of absolute metric values across task families (HOTA = 0.45 and AP = 0.72 are incommensurable).
- Latency comparisons across runtime profiles (in-process GPU vs. subprocess invocation).
- Attribution of performance to components in bundled pipelines (detector vs. tracker vs. calibrator contributions to GS-HOTA).

**Hand-waving:**
- Claiming a single "best model" across all tasks.
- Treating meta-evaluation results as ground truth about dataset quality without independent validation.
- Ignoring pre-training data asymmetry when comparing foundation-model pipelines to task-specific pipelines.

The Guimont-Martin et al. replication study ([arXiv 2024](https://arxiv.org/abs/2405.06911)) demonstrated what happens when you try to do this rigorously: they built a unified training/evaluation pipeline on MMDetection, attempted to reproduce published results, and found that DETR and ViTDet could **not** achieve their claimed accuracy or speed. The comparison problem is real, and claiming to solve it completely would itself be hand-waving.

We stay on the rigorous side by preserving the full matrix, flagging every comparison caveat, and letting the data speak through per-suite rankings rather than global scores.

---

### What Comes Next

We are working toward three milestones:

1. **Expanding the recipe catalog** — adding RT-DETR, RF-DETR, YOLOv12, Grounding DINO, and CLIP-based re-identification variants. The [RF-DETR vs. YOLOv12 controlled comparison](https://arxiv.org/html/2504.13099v1) — one of the few studies to enforce truly identical training protocols — found RF-DETR outperforming all YOLOv12 variants in single-class detection. We want to see if that finding holds in the multi-class, multi-task sports domain.

2. **Dense matrix population** — 20+ model configurations across all compatible suites, producing enough data for statistically meaningful meta-evaluation. The target is sufficient diversity to detect architecture × metric interactions at p < 0.05.

3. **Meta-evaluation analysis toolkit** — rank correlations, interaction effect estimation, evaluation protocol sensitivity analysis, and dataset difficulty decomposition. This is the second-order output: not "which model is best" but "which evaluation systems are consistent, which are biased, and which are measuring what they claim to measure."

If we succeed, the result won't just be a benchmarking system for football. It will be a demonstration that cross-architecture evaluation data, accumulated at sufficient scale and diversity, can turn the lens back on the evaluation systems themselves — revealing biases, inconsistencies, and hidden assumptions that no single-architecture study can expose.

The computer vision community has built excellent models. It has built reasonable benchmarks. What it has not yet built — at least not for the multi-task, multi-architecture, domain-specific case we face in sports video analysis — is a system for evaluating whether the benchmarks are doing their job. That's what we're building.

---

### References

#### Cross-Architecture Comparison & Fairness
- Zhang, S. et al. "Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection" (ATSS). CVPR 2020. [(paper)](https://arxiv.org/abs/1912.02424)
- Zoph, B. et al. "Learning Data Augmentation Strategies for Object Detection." ECCV 2020. [(paper)](https://arxiv.org/abs/1906.11172)
- He, K. et al. "Rethinking ImageNet Pre-training." ICCV 2019. [(paper)](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.pdf)
- Zoph, B. et al. "Rethinking Pre-training and Self-training." NeurIPS 2020. [(paper)](https://proceedings.neurips.cc/paper/2020/file/27e9661e033a73a6ad8cefcde965c54d-Paper.pdf)
- Wightman, R. et al. "ResNet Strikes Back." arXiv 2021. [(paper)](https://arxiv.org/pdf/2110.00476)
- Guimont-Martin et al. "Replication Study and Benchmarking of Real-Time Object Detection Models." arXiv 2024. [(paper)](https://arxiv.org/abs/2405.06911)
- RF-DETR vs. YOLOv12 controlled comparison. arXiv 2025. [(paper)](https://arxiv.org/html/2504.13099v1)

#### Confidence Calibration
- Kuppers, F. et al. "Multivariate Confidence Calibration for Object Detection." CVPRW 2020. [(paper)](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.pdf)
- Munir, M. et al. "Bridging Precision and Confidence: A Train-Time Loss for Calibrating Object Detection" (BPC). CVPR 2023. [(paper)](https://openaccess.thecvf.com/content/CVPR2023/papers/Munir_Bridging_Precision_and_Confidence_A_Train-Time_Loss_for_Calibrating_Object_CVPR_2023_paper.pdf)
- Pathiraja et al. "Multiclass Confidence and Localization Calibration for Object Detection." CVPR 2023. [(paper)](https://openaccess.thecvf.com/content/CVPR2023/papers/Pathiraja_Multiclass_Confidence_and_Localization_Calibration_for_Object_Detection_CVPR_2023_paper.pdf)
- Wenkel, S. et al. "Confidence Score: The Forgotten Dimension." Sensors, 2021. [(paper)](https://www.mdpi.com/1424-8220/21/13/4350)
- Gilg, J. et al. "Do We Still Need Non-Maximum Suppression?" WACV 2024. [(paper)](https://openaccess.thecvf.com/content/WACV2024/papers/Gilg_Do_We_Still_Need_Non-Maximum_Suppression_Accurate_Confidence_Estimates_and_WACV_2024_paper.pdf)

#### Benchmark Bias & Meta-Evaluation
- Singh, S. et al. "Benchmarking Object Detectors with COCO: A New Path Forward" (COCO-ReM). ECCV 2024. [(paper)](https://arxiv.org/abs/2403.18819)
- Mao, S. et al. "COCO-O: A Benchmark for Object Detectors under Natural Distribution Shifts." ICCV 2023. [(paper)](https://arxiv.org/abs/2307.12730)
- Zheng, Z. et al. "Zone Evaluation: Revealing Spatial Bias in Object Detection." IEEE TPAMI 2024. [(paper)](https://arxiv.org/abs/2310.13215)
- Oksuz, K. et al. "Imbalance Problems in Object Detection: A Review." IEEE TPAMI 2020. [(paper)](https://arxiv.org/abs/1909.00169)
- OVDEval: "How to Evaluate the Generalization of Detection?" AAAI 2024. [(paper)](https://ojs.aaai.org/index.php/AAAI/article/view/28485)

#### Metrics & Reproducibility
- Padilla, R. et al. "A Survey on Performance Metrics for Object-Detection Algorithms." IWSSIP 2020. [(paper)](https://ieeexplore.ieee.org/document/9145130/)
- Padilla, R. et al. "A Comparative Analysis of Object Detection Metrics." MDPI Electronics 2021. [(paper)](https://www.mdpi.com/2079-9292/10/3/279) [(toolkit)](https://github.com/rafaelpadilla/review_object_detection_metrics)
- "Toward Reproducible Cross-Backend Compatibility for Deep Learning." arXiv 2025. [(paper)](https://arxiv.org/pdf/2509.06977)
- TENSORSCOPE. USENIX Security 2023. [(paper)](https://www.usenix.org/system/files/usenixsecurity23-deng-zizhuang.pdf)

#### Inductive Bias & Architecture
- "Object Detection Based on CNN and Vision-Transformer: A Survey." IET Computer Vision 2025. [(paper)](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.70028)
- "Evolution of Object Detection: CNNs to Transformers." Scientific Reports 2026. [(paper)](https://www.nature.com/articles/s41598-026-37052-6)
- ViTAE. NeurIPS 2021. [(paper)](https://proceedings.neurips.cc/paper/2021/file/efb76cff97aaf057654ef2f38cd77d73-Paper.pdf)
- ConViT. arXiv 2021. [(paper)](https://arxiv.org/pdf/2103.10697)

#### CLIP & Foundation Models in Sports CV
- "CLIP-ReIdent: Player Re-Identification via CLIP." arXiv 2023. [(paper)](https://arxiv.org/abs/2303.11855)
- "YOLO-World: Real-Time Open-Vocabulary Object Detection." CVPR 2024. [(paper)](https://arxiv.org/html/2401.17270v3)
- "Robustness of Open-Vocabulary Detectors." arXiv 2024. [(paper)](https://arxiv.org/html/2405.14874v3)
- "Open-vocabulary vs. Closed-set: Best Practice for Few-shot Object Detection." NeurIPS 2024 submission. [(paper)](https://openreview.net/forum?id=LDwsvLQTLx)

#### Sports CV & SoccerNet
- "SoccerNet 2025 Challenges Results." arXiv 2025. [(paper)](https://arxiv.org/html/2508.19182v1)
- "Towards Universal Soccer Video Understanding." arXiv 2024. [(paper)](https://arxiv.org/html/2412.01820v2)
- ProCC: "A Universal Protocol to Benchmark Camera Calibration for Sports." arXiv 2024. [(paper)](https://arxiv.org/html/2404.09807v1)
- SoccerNet Calibration. [(GitHub)](https://github.com/SoccerNet/sn-calibration)
- SoccerNet Tracking. [(GitHub)](https://github.com/SoccerNet/sn-tracking)
- SoccerNet Game State Reconstruction. [(GitHub)](https://github.com/SoccerNet/sn-gamestate)

#### Convergence & Training Dynamics
- Sun, Z. et al. "Rethinking Transformer-based Set Prediction for Object Detection." ICCV 2021. [(paper)](https://arxiv.org/abs/2011.10881)
- Meng, D. et al. "Conditional DETR for Fast Training Convergence." ICCV 2021. [(paper)](https://openaccess.thecvf.com/content/ICCV2021/papers/Meng_Conditional_DETR_for_Fast_Training_Convergence_ICCV_2021_paper.pdf)
- "DN-DETR: Accelerate DETR Training by Introducing Query Denoising." arXiv 2022. [(paper)](https://arxiv.org/pdf/2203.01305)

#### Unified Frameworks
- Chen, K. et al. "MMDetection: Open MMLab Detection Toolbox and Benchmark." arXiv 2019. [(paper)](https://arxiv.org/abs/1906.07155)
- Detectron2 Model Zoo. [(GitHub)](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
- NMS-Bench. NeurIPS 2024. [(paper)](https://proceedings.neurips.cc/paper_files/paper/2024/file/dcc0ac74ac8b95dc1939804acce0317d-Paper-Conference.pdf)
- net:cal Calibration Framework. [(GitHub)](https://github.com/EFS-OpenSource/calibration-framework)

---

*David Montgomery — Football Tactical Workbench — March 2026*

*This article describes work in progress on the [football-tactical-workbench](https://github.com/DMontgomery40/football-tactical-workbench) project. The benchmarking system, meta-evaluation framework, and all evaluation code are open source.*
