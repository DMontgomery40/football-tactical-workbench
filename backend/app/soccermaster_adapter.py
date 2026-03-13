"""SoccerMaster unified pipeline adapter.

Loads the SoccerMaster backbone (fine-tuned SigLIP2-large) and all task heads:
  - KeypointsDetection: 57 pitch keypoints for homography estimation
  - SoccerNetGSR_Detection: Deformable DETR for player/ball/referee detection
  - LinesDetection: pitch line segmentation via PixelShuffle head

Model source: https://github.com/haolinyang-hlyang/SoccerMaster
Weights: https://huggingface.co/xleprime/SoccerMaster (Apache-2.0)
"""

from __future__ import annotations

import copy
import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tv_ops

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "soccermaster"

# --- 57 pitch keypoint world coordinates (meters, 105 × 68 pitch) ---
# From SoccerMaster/data/pnlcalib_utils/utils_keypoints.py
# Indices 0-56 → keypoint IDs 1-57
SOCCERMASTER_WORLD_COORDS_M = np.array(
    [
        [0.0, 0.0],       # 1: top-left corner
        [52.5, 0.0],      # 2: top center (halfway line × top sideline)
        [105.0, 0.0],     # 3: top-right corner
        [0.0, 13.84],     # 4: left penalty area top
        [16.5, 13.84],    # 5: left penalty area inner top
        [88.5, 13.84],    # 6: right penalty area inner top
        [105.0, 13.84],   # 7: right penalty area top
        [0.0, 24.84],     # 8: left goal area top
        [5.5, 24.84],     # 9: left goal area inner top
        [99.5, 24.84],    # 10: right goal area inner top
        [105.0, 24.84],   # 11: right goal area top
        [0.0, 30.34],     # 12: left goal top post (outer)
        [0.0, 30.34],     # 13: left goal top post (inner)
        [105.0, 30.34],   # 14: right goal top post (outer)
        [105.0, 30.34],   # 15: right goal top post (inner)
        [0.0, 37.66],     # 16: left goal bottom post (outer)
        [0.0, 37.66],     # 17: left goal bottom post (inner)
        [105.0, 37.66],   # 18: right goal bottom post (outer)
        [105.0, 37.66],   # 19: right goal bottom post (inner)
        [0.0, 43.16],     # 20: left goal area bottom
        [5.5, 43.16],     # 21: left goal area inner bottom
        [99.5, 43.16],    # 22: right goal area inner bottom
        [105.0, 43.16],   # 23: right goal area bottom
        [0.0, 54.16],     # 24: left penalty area bottom
        [16.5, 54.16],    # 25: left penalty area inner bottom
        [88.5, 54.16],    # 26: right penalty area inner bottom
        [105.0, 54.16],   # 27: right penalty area bottom
        [0.0, 68.0],      # 28: bottom-left corner
        [52.5, 68.0],     # 29: bottom center (halfway line × bottom sideline)
        [105.0, 68.0],    # 30: bottom-right corner
        # Circle-line intersections (31-36)
        [16.5, 26.68],    # 31: left circle × penalty area top
        [52.5, 24.85],    # 32: center circle top
        [88.5, 26.68],    # 33: right circle × penalty area top
        [16.5, 41.31],    # 34: left circle × penalty area bottom
        [52.5, 43.15],    # 35: center circle bottom
        [88.5, 41.31],    # 36: right circle × penalty area bottom
        # Tangent points (37-44)
        [19.99, 32.29],   # 37: left arc tangent top
        [43.68, 31.53],   # 38: center circle left tangent top
        [61.31, 31.53],   # 39: center circle right tangent top
        [85.0, 32.29],    # 40: right arc tangent top
        [19.99, 35.7],    # 41: left arc tangent bottom
        [43.68, 36.46],   # 42: center circle left tangent bottom
        [61.31, 36.46],   # 43: center circle right tangent bottom
        [85.0, 35.7],     # 44: right arc tangent bottom
        # Homography-projected points (45-57)
        [11.0, 34.0],     # 45: left penalty spot
        [16.5, 34.0],     # 46: left penalty area center
        [20.15, 34.0],    # 47: left arc center
        [46.03, 27.53],   # 48: center circle top-left
        [58.97, 27.53],   # 49: center circle top-right
        [43.35, 34.0],    # 50: center circle left
        [52.5, 34.0],     # 51: center spot
        [61.5, 34.0],     # 52: center circle right
        [46.03, 40.47],   # 53: center circle bottom-left
        [58.97, 40.47],   # 54: center circle bottom-right
        [84.85, 34.0],    # 55: right arc center
        [88.5, 34.0],     # 56: right penalty area center
        [94.0, 34.0],     # 57: right penalty spot
    ],
    dtype=np.float64,
)

# Convert to centimeters for our pipeline
SOCCERMASTER_WORLD_COORDS_CM = SOCCERMASTER_WORLD_COORDS_M * 100.0

# Pitch dimensions in cm (FIFA standard 105 × 68m)
SOCCERMASTER_PITCH_LENGTH_CM = 10500.0
SOCCERMASTER_PITCH_WIDTH_CM = 6800.0

# Scaled to the pipeline's 12000 × 7000 coordinate system so that minimap,
# project_point, and all downstream rendering works unchanged.
_SCALE_X = 12000.0 / 10500.0
_SCALE_Y = 7000.0 / 6800.0
SOCCERMASTER_WORLD_COORDS_PIPELINE = SOCCERMASTER_WORLD_COORDS_CM.copy()
SOCCERMASTER_WORLD_COORDS_PIPELINE[:, 0] *= _SCALE_X
SOCCERMASTER_WORLD_COORDS_PIPELINE[:, 1] *= _SCALE_Y

NUM_KEYPOINTS = 58  # 57 pitch keypoints + 1 background channel


# ---------------------------------------------------------------------------
# KeypointsHead — standalone copy from SoccerMaster repo
# ---------------------------------------------------------------------------

class KeypointsHead(nn.Module):
    """Keypoint heatmap head using PixelShuffle upsampling."""

    def __init__(self, dim_in: int = 1024, num_keypoints: int = 58):
        super().__init__()
        # Stage 1: (dim_in, 32, 32) → (192, 64, 64)
        self.stage1 = nn.Sequential(
            nn.Conv2d(dim_in, 192 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        # Stage 2: (192, 64, 64) → (96, 128, 128)
        self.stage2 = nn.Sequential(
            nn.Conv2d(192, 96 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        # Stage 3: (96, 128, 128) → (num_keypoints, 256, 256)
        self.stage3 = nn.Sequential(
            nn.Conv2d(96, 48 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_keypoints, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_keypoints),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, padding=1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_conv(x)
        return x


# ---------------------------------------------------------------------------
# Backbone key remapping: backbone.pt → transformers SiglipVisionModel
# ---------------------------------------------------------------------------

def _remap_backbone_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap SoccerMaster backbone.pt keys to HuggingFace SiglipVisionModel keys."""
    new_sd: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        # Skip temporal-only keys (not needed for single-image inference)
        if key == "temporal_embedding":
            continue
        if ".temporal_" in key:
            continue

        new_key = key

        # Embeddings
        new_key = new_key.replace(
            "vision_model_embedding.", "vision_model.embeddings."
        )

        # Encoder blocks
        m = re.match(r"encoder_blocks\.(\d+)\.encoder\.(.*)", new_key)
        if m:
            new_key = f"vision_model.encoder.layers.{m.group(1)}.{m.group(2)}"

        # Post-layernorm (backbone.pt uses "post_norm", transformers uses "post_layernorm")
        if new_key.startswith("post_norm."):
            new_key = "vision_model.post_layernorm." + new_key[len("post_norm."):]
        elif new_key.startswith("post_layernorm."):
            new_key = "vision_model." + new_key

        # Head (attention pooling)
        if new_key.startswith("head."):
            new_key = "vision_model." + new_key

        new_sd[new_key] = val
    return new_sd


# ---------------------------------------------------------------------------
# Heatmap peak extraction
# ---------------------------------------------------------------------------

def _extract_peaks(
    heatmap: np.ndarray,
    num_keypoints: int = 57,
    min_confidence: float = 0.01,
) -> list[tuple[float, float, float]]:
    """Extract sub-pixel (x, y, confidence) peak from each of the first num_keypoints channels.

    heatmap: (C, H, W) numpy array (softmax output)
    Returns list of (x_pixel, y_pixel, confidence) per keypoint.
    Coordinates are in the 256x256 heatmap space with sub-pixel precision.
    """
    H, W = heatmap.shape[1], heatmap.shape[2]
    peaks: list[tuple[float, float, float]] = []
    for ch in range(min(num_keypoints, heatmap.shape[0])):
        channel = heatmap[ch]
        max_val = float(channel.max())
        if max_val < min_confidence:
            peaks.append((0.0, 0.0, 0.0))
            continue
        idx = int(channel.argmax())
        y_int, x_int = divmod(idx, W)

        # Sub-pixel refinement via weighted centroid in 3x3 neighborhood
        y_lo = max(0, y_int - 1)
        y_hi = min(H, y_int + 2)
        x_lo = max(0, x_int - 1)
        x_hi = min(W, x_int + 2)
        patch = channel[y_lo:y_hi, x_lo:x_hi]
        total = patch.sum()
        if total > 0:
            ys, xs = np.mgrid[y_lo:y_hi, x_lo:x_hi]
            y_sub = float((ys * patch).sum() / total)
            x_sub = float((xs * patch).sum() / total)
        else:
            x_sub, y_sub = float(x_int), float(y_int)

        peaks.append((x_sub, y_sub, max_val))
    return peaks


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class SoccerMasterKeypointDetector:
    """Pitch keypoint detector using SoccerMaster backbone + keypoint head."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._model: SiglipVisionModelType | None = None
        self._head: KeypointsHead | None = None
        self._loaded = False

    def load(self) -> None:
        """Load backbone and keypoint head weights."""
        if self._loaded:
            return

        from transformers import SiglipVisionConfig, SiglipVisionModel

        backbone_path = MODEL_DIR / "backbone.pt"
        head_path = MODEL_DIR / "KeypointsDetection.pt"

        if not backbone_path.exists() or not head_path.exists():
            raise FileNotFoundError(
                f"SoccerMaster weights not found in {MODEL_DIR}. "
                "Download from https://huggingface.co/xleprime/SoccerMaster"
            )

        # Build backbone from SigLIP2 config (no pretrained weight download)
        config = SiglipVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=512,
            patch_size=16,
        )
        self._model = SiglipVisionModel(config)

        # Load fine-tuned backbone weights
        bb_sd = torch.load(str(backbone_path), map_location="cpu", weights_only=False)
        remapped = _remap_backbone_keys(bb_sd)
        missing, unexpected = self._model.load_state_dict(remapped, strict=False)
        # Expected missing: vision_model.head.probe (not in backbone.pt)
        # Expected unexpected: none after remapping

        # Build and load keypoint head
        self._head = KeypointsHead(dim_in=1024, num_keypoints=NUM_KEYPOINTS)
        head_sd = torch.load(str(head_path), map_location="cpu", weights_only=False)
        # Strip "keypoints_head." prefix from checkpoint keys
        stripped_sd = {
            k.removeprefix("keypoints_head."): v for k, v in head_sd.items()
        }
        self._head.load_state_dict(stripped_sd, strict=False)

        # Move to device and eval mode
        self._model = self._model.to(self.device).eval()
        self._head = self._head.to(self.device).eval()
        self._loaded = True

    @torch.no_grad()
    def detect_raw(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Detect all 57 pitch keypoints. Returns (57, 3) array of [x, y, conf]
        in the original frame coordinate space. Confidence 0 = not detected."""
        self.load()
        assert self._model is not None and self._head is not None

        h_orig, w_orig = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        tensor = (tensor - 0.5) / 0.5  # SigLIP2 normalization
        tensor = tensor.unsqueeze(0).to(self.device)

        outputs = self._model(tensor, output_hidden_states=False)
        local_features = outputs.last_hidden_state  # (1, 1024, 1024)

        N, L, D = local_features.shape
        Hf = Wf = int(math.sqrt(L))
        spatial = local_features.permute(0, 2, 1).contiguous().reshape(N, D, Hf, Wf)

        heatmaps = self._head(spatial)
        heatmap_np = heatmaps[0].cpu().numpy()  # (58, 256, 256)

        peaks = _extract_peaks(heatmap_np, num_keypoints=57, min_confidence=0.0)

        scale_x = w_orig / 256.0
        scale_y = h_orig / 256.0

        result = np.zeros((57, 3), dtype=np.float32)
        for i, (px, py, conf) in enumerate(peaks):
            result[i] = [px * scale_x, py * scale_y, conf]
        return result


# Module-level singleton
_detector: SoccerMasterKeypointDetector | None = None


def get_soccermaster_detector(device: str = "cpu") -> SoccerMasterKeypointDetector:
    """Get or create the singleton SoccerMaster keypoint detector."""
    global _detector
    if _detector is None:
        _detector = SoccerMasterKeypointDetector(device=device)
    return _detector


def detect_pitch_homography_soccermaster(
    frame: np.ndarray,
    device: str = "cpu",
    confidence_threshold: float = 0.15,
) -> tuple[np.ndarray | None, np.ndarray | None, int, int, float]:
    """Drop-in replacement for detect_pitch_homography using SoccerMaster.

    Returns the same 5-tuple as the YOLO version:
        (homography_matrix, detected_keypoints_array, visible_count, inlier_count, reprojection_error)

    detected_keypoints_array is (57, 3) with [x, y, conf] per keypoint channel.
    """
    detector = get_soccermaster_detector(device=device)
    kp_arr = detector.detect_raw(frame)  # (57, 3)

    valid_mask = kp_arr[:, 2] >= confidence_threshold
    visible_count = int(valid_mask.sum())
    if visible_count < 4:
        return None, kp_arr, visible_count, 0, float("inf")

    image_points = kp_arr[valid_mask, :2].astype(np.float32)
    field_points = SOCCERMASTER_WORLD_COORDS_PIPELINE[valid_mask].astype(np.float32)

    homography_matrix, inlier_mask = cv2.findHomography(
        image_points, field_points, cv2.RANSAC, 35.0
    )
    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else visible_count
    if homography_matrix is None or inlier_count < 4:
        return None, kp_arr, visible_count, inlier_count, float("inf")

    projected = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2), homography_matrix.astype(np.float32)
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - field_points, axis=1)
    reprojection_error = float(errors.mean())

    return (
        homography_matrix.astype(np.float32),
        kp_arr,
        visible_count,
        inlier_count,
        reprojection_error,
    )


# ---------------------------------------------------------------------------
# LinesHead — PixelShuffle line segmentation head
# ---------------------------------------------------------------------------

class LinesHead(nn.Module):
    """Line segmentation head using PixelShuffle upsampling.

    Same architecture pattern as KeypointsHead but outputs line class channels.
    Checkpoint keys are prefixed with ``lines_head.``.
    """

    def __init__(self, dim_in: int = 1024, num_classes: int = 24):
        super().__init__()
        # Stage 1: (dim_in, 32, 32) → (192, 64, 64)
        self.stage1 = nn.Sequential(
            nn.Conv2d(dim_in, 192 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        # Stage 2: (192, 64, 64) → (96, 128, 128)
        self.stage2 = nn.Sequential(
            nn.Conv2d(192, 96 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        # Stage 3: (96, 128, 128) → (num_classes, 256, 256)
        self.stage3 = nn.Sequential(
            nn.Conv2d(96, 48 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Deformable DETR Detection Head — pure PyTorch (CPU/MPS safe)
# ---------------------------------------------------------------------------

# Role labels from SoccerNet GSR task
ROLE_LABELS: list[str] = [
    "player",
    "goalkeeper",
    "referee",
    "ball",
    "staff",
    "other",
]


class _MLP(nn.Module):
    """Simple multi-layer perceptron (used for bbox regression)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class _DeformableCrossAttention(nn.Module):
    """CPU/MPS-safe approximation of multi-scale deformable cross-attention.

    Uses bilinear grid_sample instead of the CUDA-only MSDeformAttn kernel.
    Architecture from checkpoint: 4 heads, 1 level, 1 point per head.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, n_levels: int = 1, n_points: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: tuple[int, int],
    ) -> torch.Tensor:
        """
        query: (B, N_q, d_model)
        reference_points: (B, N_q, 2) in [0, 1] normalized coords
        value: (B, H*W, d_model)  -- flattened spatial features
        spatial_shapes: (H, W) of the feature map
        """
        B, N_q, _ = query.shape
        H_feat, W_feat = spatial_shapes
        head_dim = self.d_model // self.n_heads

        # Project values → (B, H*W, n_heads, head_dim)
        val = self.value_proj(value).reshape(B, H_feat * W_feat, self.n_heads, head_dim)
        # Reshape value into spatial grid → (B * n_heads, head_dim, H, W)
        val_spatial = val.permute(0, 2, 3, 1).reshape(B * self.n_heads, head_dim, H_feat, W_feat)

        # Sampling offsets → (B, N_q, n_heads, n_levels * n_points, 2)
        offsets = self.sampling_offsets(query).reshape(
            B, N_q, self.n_heads, self.n_levels * self.n_points, 2
        )
        # Attention weights → (B, N_q, n_heads, n_levels * n_points)
        attn_w = self.attention_weights(query).reshape(
            B, N_q, self.n_heads, self.n_levels * self.n_points
        )
        attn_w = F.softmax(attn_w, dim=-1)

        # Compute sampling locations: reference + offset (both in [0,1])
        # offsets are small deltas; reference_points is (B, N_q, 2)
        ref = reference_points[:, :, None, None, :]  # (B, N_q, 1, 1, 2)
        sample_pts = ref + offsets  # (B, N_q, n_heads, n_points, 2)
        # Convert from [0,1] to grid_sample coords [-1, 1]
        grid = sample_pts * 2.0 - 1.0  # (B, N_q, n_heads, n_points, 2)

        # For grid_sample we need (B*n_heads, N_q, n_points, 2)
        grid = grid.permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, N_q, self.n_points, 2)

        # Bilinear sample → (B*n_heads, head_dim, N_q, n_points)
        sampled = F.grid_sample(
            val_spatial, grid, mode="bilinear", padding_mode="zeros", align_corners=False,
        )
        # → (B, n_heads, head_dim, N_q, n_points)
        sampled = sampled.reshape(B, self.n_heads, head_dim, N_q, self.n_points)
        # → (B, N_q, n_heads, n_points, head_dim)
        sampled = sampled.permute(0, 3, 1, 4, 2)

        # Weighted sum across sampling points
        # attn_w: (B, N_q, n_heads, n_points) → unsqueeze for broadcast
        output = (sampled * attn_w.unsqueeze(-1)).sum(dim=3)  # (B, N_q, n_heads, head_dim)
        output = output.reshape(B, N_q, self.d_model)

        return self.output_proj(output)


class _DeformableDecoderLayer(nn.Module):
    """Single decoder layer: self-attn → cross-attn (deformable) → FFN."""

    def __init__(self, d_model: int = 256, n_heads_self: int = 8, d_ffn: int = 1024,
                 n_heads_cross: int = 4, n_points: int = 1):
        super().__init__()
        # Self-attention (standard multi-head)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads_self, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (deformable)
        self.cross_attn = _DeformableCrossAttention(
            d_model=d_model, n_heads=n_heads_cross, n_levels=1, n_points=n_points,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: tuple[int, int],
    ) -> torch.Tensor:
        # Self-attention
        q = k = tgt
        sa_out, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + sa_out)

        # Cross-attention (deformable)
        ca_out = self.cross_attn(tgt, reference_points, memory, spatial_shapes)
        tgt = self.norm2(tgt + ca_out)

        # FFN
        ffn_out = self.linear2(F.relu(self.linear1(tgt)))
        tgt = self.norm3(tgt + ffn_out)

        return tgt


class _DeformableDecoder(nn.Module):
    """Stack of deformable decoder layers with iterative bbox refinement."""

    def __init__(self, decoder_layer: _DeformableDecoderLayer, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        # Per-layer bbox embed for iterative refinement (loaded from checkpoint)
        self.bbox_embed: nn.ModuleList | None = None

    def forward(
        self,
        tgt: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = tgt
        ref = reference_points
        for idx, layer in enumerate(self.layers):
            output = layer(output, ref, memory, spatial_shapes)
            # Iterative bbox refinement: update reference points
            if self.bbox_embed is not None:
                delta = self.bbox_embed[idx](output)  # (B, N_q, 4)
                new_ref = torch.sigmoid(delta[..., :2] + _inverse_sigmoid(ref))
                ref = new_ref.detach()
        return output, ref


class DetectionHead(nn.Module):
    """Deformable DETR detection head for player/ball/referee detection.

    Loaded from SoccerNetGSR_Detection.pt.  Pure PyTorch, no CUDA custom ops.
    """

    def __init__(
        self,
        d_model: int = 256,
        dim_backbone: int = 1024,
        num_queries: int = 300,
        num_classes: int = 1,
        num_roles: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_roles = num_roles

        # Input projection from backbone dim → d_model
        # Checkpoint uses nn.Sequential(Conv2d, GroupNorm)
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim_backbone, d_model, kernel_size=1),
                nn.GroupNorm(32, d_model),
            )
        ])

        # Query embedding: first d_model dims = content, last d_model dims = positional
        self.query_embed = nn.Embedding(num_queries, d_model * 2)

        # Transformer decoder
        decoder_layer = _DeformableDecoderLayer(
            d_model=d_model, n_heads_self=8, d_ffn=1024,
            n_heads_cross=4, n_points=1,
        )
        self.transformer = nn.Module()
        self.transformer.decoder = _DeformableDecoder(decoder_layer, num_layers=1)
        self.transformer.reference_points = nn.Linear(d_model, 2)

        # Classification and regression heads
        self.class_embed = nn.ModuleList([nn.Linear(d_model, num_classes)])
        self.bbox_embed = nn.ModuleList([
            _MLP(d_model, d_model, 4, num_layers=3)
        ])

        # Auxiliary heads: role, jersey number
        self.role_embed = nn.ModuleList([nn.Linear(d_model, num_roles)])
        self.digit_head_embed = nn.ModuleList([nn.Linear(d_model, 10)])
        self.digit_tail_embed = nn.ModuleList([nn.Linear(d_model, 11)])
        self.jn_holistic_embed = nn.ModuleList([nn.Linear(d_model, 101)])

        # Wire iterative bbox refinement in decoder
        self.transformer.decoder.bbox_embed = self.bbox_embed

    def forward(
        self,
        spatial_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        spatial_features: (B, dim_backbone, H, W) from backbone.

        Returns dict with:
          pred_logits: (B, num_queries, num_classes)
          pred_boxes:  (B, num_queries, 4) as (cx, cy, w, h) in [0,1]
          pred_roles:  (B, num_queries, num_roles)
          pred_jn_holistic: (B, num_queries, 101)
          pred_digit_head:  (B, num_queries, 10)
          pred_digit_tail:  (B, num_queries, 11)
        """
        B = spatial_features.shape[0]

        # Project backbone features
        src = self.input_proj[0](spatial_features)  # (B, d_model, H, W)
        _, _, H_feat, W_feat = src.shape

        # Flatten spatial → sequence
        memory = src.flatten(2).permute(0, 2, 1)  # (B, H*W, d_model)

        # Query embeddings
        qe = self.query_embed.weight  # (num_queries, 2*d_model)
        query_pos, query_content = qe.split(self.d_model, dim=1)
        query_content = query_content.unsqueeze(0).expand(B, -1, -1)

        # Reference points from positional embedding
        ref_pts = torch.sigmoid(
            self.transformer.reference_points(query_pos)
        )  # (num_queries, 2)
        ref_pts = ref_pts.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, 2)

        # Decode
        hs, final_ref = self.transformer.decoder(
            query_content, ref_pts, memory, (H_feat, W_feat),
        )

        # Heads (applied to last decoder layer output)
        logits = self.class_embed[0](hs)
        # Bbox regression: delta + inverse_sigmoid(ref) → sigmoid
        bbox_delta = self.bbox_embed[0](hs)
        ref_for_box = torch.cat([final_ref, final_ref], dim=-1)  # reuse ref for cx,cy
        # Actually, bbox_embed predicts offsets relative to reference for cx,cy and raw for w,h
        pred_boxes = torch.sigmoid(
            bbox_delta + torch.cat([_inverse_sigmoid(final_ref), torch.zeros_like(final_ref)], dim=-1)
        )

        roles = self.role_embed[0](hs)
        jn_hol = self.jn_holistic_embed[0](hs)
        dh = self.digit_head_embed[0](hs)
        dt = self.digit_tail_embed[0](hs)

        return {
            "pred_logits": logits,
            "pred_boxes": pred_boxes,
            "pred_roles": roles,
            "pred_jn_holistic": jn_hol,
            "pred_digit_head": dh,
            "pred_digit_tail": dt,
        }


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def _load_detection_head(path: Path, device: torch.device) -> DetectionHead:
    """Build DetectionHead and load SoccerNetGSR_Detection.pt weights."""
    sd = torch.load(str(path), map_location="cpu", weights_only=False)

    # Infer dimensions from checkpoint
    num_classes = sd["class_embed.0.bias"].shape[0]
    num_roles = sd["role_embed.0.bias"].shape[0]

    head = DetectionHead(
        d_model=256,
        dim_backbone=1024,
        num_queries=300,
        num_classes=num_classes,
        num_roles=num_roles,
    )

    # Load state dict — keys match our module structure
    missing, unexpected = head.load_state_dict(sd, strict=False)
    if unexpected:
        logger.debug("Detection head unexpected keys: %s", unexpected)
    if missing:
        logger.debug("Detection head missing keys: %s", missing)

    return head.to(device).eval()


def _load_lines_head(path: Path, device: torch.device) -> LinesHead:
    """Build LinesHead and load LinesDetection.pt weights."""
    sd = torch.load(str(path), map_location="cpu", weights_only=False)

    # Infer num_classes from final_conv output shape
    num_classes = sd["lines_head.final_conv.0.bias"].shape[0]

    head = LinesHead(dim_in=1024, num_classes=num_classes)

    # Strip "lines_head." prefix
    stripped = {k.removeprefix("lines_head."): v for k, v in sd.items()}
    missing, unexpected = head.load_state_dict(stripped, strict=False)
    if unexpected:
        logger.debug("Lines head unexpected keys: %s", unexpected)

    return head.to(device).eval()


# ---------------------------------------------------------------------------
# SoccerMasterPipeline — unified backbone + all heads
# ---------------------------------------------------------------------------

class SoccerMasterPipeline:
    """Unified SoccerMaster pipeline: single backbone forward → all task heads.

    Heads:
      - keypoints: 57 pitch keypoints for homography
      - detection: Deformable DETR for players/ball/referee
      - lines: pitch line segmentation
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._backbone: Any = None
        self._keypoints_head: KeypointsHead | None = None
        self._detection_head: DetectionHead | None = None
        self._lines_head: LinesHead | None = None
        self._loaded = False

    def load(self) -> None:
        """Load backbone and all task head weights."""
        if self._loaded:
            return

        from transformers import SiglipVisionConfig, SiglipVisionModel

        backbone_path = MODEL_DIR / "backbone.pt"
        kp_path = MODEL_DIR / "KeypointsDetection.pt"
        det_path = MODEL_DIR / "SoccerNetGSR_Detection.pt"
        lines_path = MODEL_DIR / "LinesDetection.pt"

        required = [backbone_path, kp_path, det_path, lines_path]
        missing_files = [p for p in required if not p.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"SoccerMaster weights missing: {[str(p) for p in missing_files]}. "
                "Download from https://huggingface.co/xleprime/SoccerMaster"
            )

        # Backbone (shared)
        config = SiglipVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=512,
            patch_size=16,
        )
        self._backbone = SiglipVisionModel(config)
        bb_sd = torch.load(str(backbone_path), map_location="cpu", weights_only=False)
        remapped = _remap_backbone_keys(bb_sd)
        self._backbone.load_state_dict(remapped, strict=False)
        self._backbone = self._backbone.to(self.device).eval()

        # Keypoints head
        self._keypoints_head = KeypointsHead(dim_in=1024, num_keypoints=NUM_KEYPOINTS)
        kp_sd = torch.load(str(kp_path), map_location="cpu", weights_only=False)
        stripped_kp = {k.removeprefix("keypoints_head."): v for k, v in kp_sd.items()}
        self._keypoints_head.load_state_dict(stripped_kp, strict=False)
        self._keypoints_head = self._keypoints_head.to(self.device).eval()

        # Detection head
        self._detection_head = _load_detection_head(det_path, self.device)

        # Lines head
        self._lines_head = _load_lines_head(lines_path, self.device)

        self._loaded = True
        logger.info("SoccerMasterPipeline loaded on %s", self.device)

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Convert BGR frame to normalized 512x512 tensor."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        tensor = (tensor - 0.5) / 0.5  # SigLIP2 normalization
        return tensor.unsqueeze(0).to(self.device)

    def _extract_backbone_features(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run backbone, return (B, 1024, H_feat, W_feat) spatial features."""
        outputs = self._backbone(tensor, output_hidden_states=False)
        feats = outputs.last_hidden_state  # (B, L, 1024)
        B, L, D = feats.shape
        Hf = Wf = int(math.sqrt(L))
        return feats.permute(0, 2, 1).contiguous().reshape(B, D, Hf, Wf)

    @torch.no_grad()
    def detect_frame(
        self,
        frame_bgr: np.ndarray,
        det_confidence: float = 0.3,
        nms_iou: float = 0.5,
    ) -> dict[str, Any]:
        """Run all heads on a single frame.

        Returns dict with:
          detections: list of dicts {bbox, class_id, class_name, confidence,
                                     role, role_id, jersey_number}
          keypoints: (57, 3) array [x, y, conf] in original frame coords
          lines: (num_classes, 256, 256) numpy line segmentation mask
        """
        self.load()
        assert self._backbone is not None
        assert self._keypoints_head is not None
        assert self._detection_head is not None
        assert self._lines_head is not None

        h_orig, w_orig = frame_bgr.shape[:2]
        tensor = self._preprocess(frame_bgr)

        # Shared backbone forward pass
        spatial = self._extract_backbone_features(tensor)  # (1, 1024, 32, 32)

        # --- Keypoints ---
        heatmaps = self._keypoints_head(spatial)
        heatmap_np = heatmaps[0].cpu().numpy()
        peaks = _extract_peaks(heatmap_np, num_keypoints=57, min_confidence=0.0)
        scale_x = w_orig / 256.0
        scale_y = h_orig / 256.0
        kp_result = np.zeros((57, 3), dtype=np.float32)
        for i, (px, py, conf) in enumerate(peaks):
            kp_result[i] = [px * scale_x, py * scale_y, conf]

        # --- Detection ---
        det_out = self._detection_head(spatial)
        detections = self._postprocess_detections(
            det_out, h_orig, w_orig, det_confidence, nms_iou,
        )

        # --- Lines ---
        lines_out = self._lines_head(spatial)
        lines_np = lines_out[0].cpu().numpy()  # (num_classes, 256, 256)

        return {
            "detections": detections,
            "keypoints": kp_result,
            "lines": lines_np,
        }

    def _postprocess_detections(
        self,
        det_out: dict[str, torch.Tensor],
        h_orig: int,
        w_orig: int,
        confidence: float,
        nms_iou: float,
    ) -> list[dict[str, Any]]:
        """Convert raw detection head output to list of detection dicts."""
        logits = det_out["pred_logits"][0]    # (num_queries, num_classes)
        boxes = det_out["pred_boxes"][0]       # (num_queries, 4) cx,cy,w,h in [0,1]
        roles = det_out["pred_roles"][0]       # (num_queries, num_roles)
        jn_hol = det_out["pred_jn_holistic"][0]

        # Binary objectness: sigmoid on logits
        scores = torch.sigmoid(logits).max(dim=-1).values  # (num_queries,)
        class_ids = torch.sigmoid(logits).argmax(dim=-1)

        # Filter by confidence
        keep = scores >= confidence
        scores = scores[keep]
        boxes = boxes[keep]
        class_ids = class_ids[keep]
        roles_keep = roles[keep]
        jn_hol_keep = jn_hol[keep]

        if len(scores) == 0:
            return []

        # Convert cx,cy,w,h → x1,y1,x2,y2 in pixel coords
        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - w / 2) * w_orig
        y1 = (cy - h / 2) * h_orig
        x2 = (cx + w / 2) * w_orig
        y2 = (cy + h / 2) * h_orig
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        # NMS
        nms_keep = tv_ops.nms(boxes_xyxy, scores, nms_iou)
        boxes_xyxy = boxes_xyxy[nms_keep]
        scores = scores[nms_keep]
        class_ids = class_ids[nms_keep]
        roles_keep = roles_keep[nms_keep]
        jn_hol_keep = jn_hol_keep[nms_keep]

        # Decode roles and jersey numbers
        role_ids = roles_keep.argmax(dim=-1)
        jn_ids = jn_hol_keep.argmax(dim=-1)

        results: list[dict[str, Any]] = []
        for i in range(len(scores)):
            role_id = int(role_ids[i])
            role_name = ROLE_LABELS[role_id] if role_id < len(ROLE_LABELS) else "unknown"
            jn = int(jn_ids[i])
            jersey_number = jn if jn < 100 else None  # 100 = "none"

            bbox = boxes_xyxy[i].cpu().numpy().tolist()
            results.append({
                "bbox": [float(v) for v in bbox],
                "class_id": int(class_ids[i]),
                "class_name": role_name,  # map role as class name for downstream
                "confidence": float(scores[i]),
                "role": role_name,
                "role_id": role_id,
                "jersey_number": jersey_number,
            })

        return results


# ---------------------------------------------------------------------------
# Module-level pipeline singleton
# ---------------------------------------------------------------------------

_pipeline: SoccerMasterPipeline | None = None


def get_soccermaster_pipeline(device: str = "cpu") -> SoccerMasterPipeline:
    """Get or create the singleton SoccerMasterPipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = SoccerMasterPipeline(device=device)
    return _pipeline


# ---------------------------------------------------------------------------
# Drop-in detection function (replaces YOLO detect_players_for_frame)
# ---------------------------------------------------------------------------

def detect_players_soccermaster(
    frame_bgr: np.ndarray,
    device: str = "cpu",
    confidence: float = 0.3,
    nms_iou: float = 0.5,
) -> list[dict[str, Any]]:
    """Drop-in replacement for YOLO-based detect_players_for_frame.

    Returns list of dicts with keys matching the pipeline's player format:
      track_id, confidence, bbox (x1,y1,x2,y2), anchor (x,y), field_point,
      role, jersey_number
    """
    pipeline = get_soccermaster_pipeline(device=device)
    result = pipeline.detect_frame(frame_bgr, det_confidence=confidence, nms_iou=nms_iou)

    h_orig, w_orig = frame_bgr.shape[:2]
    players: list[dict[str, Any]] = []

    for idx, det in enumerate(result["detections"]):
        x1, y1, x2, y2 = det["bbox"]
        # Anchor at bottom-center of bbox (foot position)
        anchor_x = (x1 + x2) / 2.0
        anchor_y = y2

        players.append({
            "track_id": idx,  # temporary ID; real tracker assigns persistent IDs
            "confidence": det["confidence"],
            "bbox": [x1, y1, x2, y2],
            "anchor": [anchor_x, anchor_y],
            "field_point": None,  # needs homography to compute
            "role": det["role"],
            "jersey_number": det["jersey_number"],
            "class_name": det["class_name"],
        })

    return players
