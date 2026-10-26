"""
YOLO26 heatmap generator.

Only edit MODEL_PATH and IMAGE_PATH for the common case, then run:

    python YOLO26Heatmap.py

The script writes all results to OUTPUT_DIR:
    original.jpg
    detection_result.jpg
    eigen_cam_overlay.jpg
    grad_cam_overlay.jpg
    grad_cam_plus_plus_overlay.jpg
    xgrad_cam_overlay.jpg
    detection_density_overlay.jpg
    heatmap_methods_compare.jpg

Notes:
    - CAM methods explain model feature responses. They are not the same as
      tracking heatmaps or object-counting heatmaps.
    - Detection density is a confidence-weighted Gaussian map drawn from boxes.
      It is included for comparison and business/reporting visualization.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# =============================================================================
# 1. Common configuration: usually you only need to change these two paths.
# =============================================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "yolo26n.pt"  # Change to your trained .pt model
IMAGE_PATH = ROOT / "test1.jpg"  # Change to your image

OUTPUT_DIR = ROOT / "runs" / "yolo26_heatmap"
IMG_SIZE = 640
DEVICE = "auto"  # auto, cpu, cuda, cuda:0, 0

CONF_THRESHOLD = 0.25
TARGET_CLASS: Optional[int] = None  # None means use the highest-confidence class
TARGET_LAYER: str = "auto"  # auto or a layer index/name, e.g. "22", "model.22"

METHODS = ("eigen_cam", "grad_cam", "grad_cam_plus_plus", "xgrad_cam", "detection_density")
COLORMAP = cv2.COLORMAP_JET
ALPHA = 0.45
DRAW_BOXES = True
SAVE_RAW_HEATMAP = True


# =============================================================================
# 2. Data structures.
# =============================================================================


@dataclass
class LetterboxInfo:
    scale: float
    pad_left: int
    pad_top: int
    new_width: int
    new_height: int
    input_size: int
    original_width: int
    original_height: int


@dataclass
class Detection:
    xyxy: List[float]
    conf: float
    cls: int
    name: str


@dataclass
class HeatmapResult:
    method: str
    raw_path: str
    overlay_path: str
    note: str


# =============================================================================
# 3. Utility functions.
# =============================================================================


def ensure_repo_import() -> None:
    """Prefer the local ultralytics package in this repository."""
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def require_dependencies() -> Tuple[Any, Any]:
    """Import heavy dependencies with a clear error message."""
    ensure_repo_import()
    try:
        import torch
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        name = exc.name or "dependency"
        raise RuntimeError(
            f"Missing dependency: {name}\n\n"
            "Recommended environment commands:\n"
            "  conda create -n yolo26_heatmap python=3.11 -y\n"
            "  conda activate yolo26_heatmap\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install ultralytics opencv-python matplotlib pillow numpy\n\n"
            "If you already have a YOLO environment, run this script with that environment's python.exe."
        ) from exc
    return torch, YOLO


def select_device(torch: Any, request: str) -> str:
    if request == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if request.isdigit():
        return f"cuda:{request}"
    if request == "cuda":
        return "cuda:0"
    return request


def read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def letterbox_bgr(image: np.ndarray, size: int = 640, fill: int = 114) -> Tuple[np.ndarray, LetterboxInfo]:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    info = LetterboxInfo(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        new_width=new_w,
        new_height=new_h,
        input_size=size,
        original_width=w,
        original_height=h,
    )
    return canvas, info


def image_to_tensor(torch: Any, image_bgr: np.ndarray, device: str) -> Any:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor / 255.0
    return tensor.to(device)


def normalize_map(cam: np.ndarray) -> np.ndarray:
    cam = np.nan_to_num(cam.astype(np.float32), copy=False)
    if cam.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    low, high = np.percentile(cam, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-8:
        low, high = float(np.min(cam)), float(np.max(cam))
    if abs(high - low) < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    cam = (cam - low) / (high - low)
    return np.clip(cam, 0.0, 1.0).astype(np.float32)


def crop_letterbox_to_original(cam: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Convert a square input heatmap back to original image size."""
    cam = cv2.resize(cam.astype(np.float32), (info.input_size, info.input_size), interpolation=cv2.INTER_CUBIC)
    y1, y2 = info.pad_top, info.pad_top + info.new_height
    x1, x2 = info.pad_left, info.pad_left + info.new_width
    cam = cam[y1:y2, x1:x2]
    cam = cv2.resize(cam, (info.original_width, info.original_height), interpolation=cv2.INTER_CUBIC)
    return normalize_map(cam)


def colorize_heatmap(cam: np.ndarray, colormap: int = COLORMAP) -> np.ndarray:
    cam_uint8 = np.uint8(np.clip(cam, 0.0, 1.0) * 255)
    return cv2.applyColorMap(cam_uint8, colormap)


def overlay_heatmap(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = ALPHA, colormap: int = COLORMAP) -> np.ndarray:
    colored = colorize_heatmap(cam, colormap)
    if colored.shape[:2] != image_bgr.shape[:2]:
        colored = cv2.resize(colored, (image_bgr.shape[1], image_bgr.shape[0]))
    return cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0)


def draw_detections(image_bgr: np.ndarray, detections: Sequence[Detection], names: Dict[int, str]) -> np.ndarray:
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy)
        color = (50, 220, 50)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(det.cls, str(det.cls))} {det.conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1 - th - baseline - 4)
        cv2.rectangle(out, (x1, y_text), (x1 + tw + 6, y_text + th + baseline + 6), color, -1)
        cv2.putText(out, label, (x1 + 3, y_text + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return out


def save_heatmap_pair(
    image_bgr: np.ndarray,
    cam: np.ndarray,
    detections: Sequence[Detection],
    names: Dict[int, str],
    out_dir: Path,
    method: str,
    note: str,
) -> HeatmapResult:
    raw = colorize_heatmap(cam)
    overlay = overlay_heatmap(image_bgr, cam)
    if DRAW_BOXES and detections:
        overlay = draw_detections(overlay, detections, names)

    raw_path = out_dir / f"{method}_raw.jpg"
    overlay_path = out_dir / f"{method}_overlay.jpg"
    if SAVE_RAW_HEATMAP:
        cv2.imwrite(str(raw_path), raw)
    cv2.imwrite(str(overlay_path), overlay)
    return HeatmapResult(method=method, raw_path=str(raw_path), overlay_path=str(overlay_path), note=note)


def first_tensor(value: Any) -> Optional[Any]:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def find_target_layer(model: Any, target: str) -> Tuple[str, Any]:
    layers = list(getattr(model, "model", []))
    named = dict(model.named_modules())

    if not layers:
        raise RuntimeError("The loaded YOLO model does not expose model.model layers.")

    if target == "auto":
        # Prefer the last feature layer before the Detect head.
        index = max(0, len(layers) - 2)
        return f"model.{index}", layers[index]

    clean = target.replace("model.", "")
    if clean.lstrip("-").isdigit():
        index = int(clean)
        if index < 0:
            index = len(layers) + index
        if index < 0 or index >= len(layers):
            raise ValueError(f"Layer index {target} is out of range. Valid range: 0-{len(layers) - 1}.")
        return f"model.{index}", layers[index]

    for candidate in (target, f"model.{target}"):
        if candidate in named:
            return candidate, named[candidate]

    available = ", ".join(f"model.{i}:{m.__class__.__name__}" for i, m in enumerate(layers[-8:], start=max(0, len(layers) - 8)))
    raise ValueError(f"Could not find target layer '{target}'. Last layers: {available}")


def find_target_layers(model: Any, target: str) -> List[Tuple[str, Any]]:
    """Find one explicit layer or YOLO Detect input layers for auto mode."""
    layers = list(getattr(model, "model", []))
    if not layers:
        raise RuntimeError("The loaded YOLO model does not expose model.model layers.")

    if target != "auto":
        return [find_target_layer(model, target)]

    # YOLO detection heads usually receive a list of feature layers, e.g. [16, 19, 22].
    detect_head = layers[-1]
    from_layers = getattr(detect_head, "f", None)
    indices: List[int] = []
    if isinstance(from_layers, int):
        indices = [from_layers if from_layers >= 0 else len(layers) + from_layers]
    elif isinstance(from_layers, (list, tuple)):
        indices = [int(i if i >= 0 else len(layers) + i) for i in from_layers]

    indices = [i for i in indices if 0 <= i < len(layers)]
    if not indices:
        indices = [max(0, len(layers) - 2)]

    return [(f"model.{i}", layers[i]) for i in indices]


def flatten_scores(scores: Any, target_class: Optional[int]) -> Any:
    """Return a differentiable scalar from YOLO raw classification logits."""
    # scores shape is usually [B, C, N].
    if scores.ndim != 3:
        return scores.max()
    if target_class is not None and 0 <= target_class < scores.shape[1]:
        return scores[:, target_class, :].max()
    return scores.max()


def choose_cam_score(output: Any, target_class: Optional[int]) -> Any:
    """Pick a differentiable class score from YOLO26 end-to-end output."""
    if isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], dict):
        raw = output[1]
        if "one2many" in raw and isinstance(raw["one2many"], dict) and "scores" in raw["one2many"]:
            return flatten_scores(raw["one2many"]["scores"], target_class)
        if "one2one" in raw and isinstance(raw["one2one"], dict) and "scores" in raw["one2one"]:
            return flatten_scores(raw["one2one"]["scores"], target_class)

    if isinstance(output, dict):
        if "one2many" in output and "scores" in output["one2many"]:
            return flatten_scores(output["one2many"]["scores"], target_class)
        if "scores" in output:
            return flatten_scores(output["scores"], target_class)

    tensor = first_tensor(output)
    if tensor is None:
        raise RuntimeError("No tensor output was found for CAM backward.")
    if not getattr(tensor, "requires_grad", False):
        raise RuntimeError("The selected output tensor does not require gradients.")
    return tensor.max()


def tensor_to_numpy_4d(tensor: Any) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4:
        raise RuntimeError(f"Expected a 4D activation tensor, got shape {arr.shape}.")
    return arr


def activation_to_chw(arr: np.ndarray) -> np.ndarray:
    """Convert [B, C, H, W] to [C, H, W]."""
    arr = np.nan_to_num(arr.astype(np.float32), copy=False)
    if arr.ndim != 4:
        raise RuntimeError(f"Expected 4D activation array, got {arr.shape}.")
    return arr[0]


def compute_eigen_cam(activation: np.ndarray) -> Tuple[np.ndarray, str]:
    """EigenCAM: first principal component of activation maps."""
    act = activation_to_chw(activation)
    c, h, w = act.shape
    matrix = act.reshape(c, h * w).T
    matrix = matrix - matrix.mean(axis=0, keepdims=True)

    try:
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        weights = vt[0]
        cam = matrix @ weights
        cam = cam.reshape(h, w)
    except np.linalg.LinAlgError:
        cam = np.mean(np.abs(act), axis=0)

    if abs(float(cam.min())) > abs(float(cam.max())):
        cam = -cam
    return normalize_map(cam), "EigenCAM uses the first principal component of the target layer activations; no gradient is needed."


def compute_grad_cam(activation: np.ndarray, gradient: np.ndarray) -> Tuple[np.ndarray, str]:
    act = activation_to_chw(activation)
    grad = activation_to_chw(gradient)
    weights = grad.mean(axis=(1, 2), keepdims=True)
    cam = np.sum(weights * act, axis=0)
    cam = np.maximum(cam, 0)
    if float(cam.max()) <= 1e-8:
        cam = np.abs(cam)
    return normalize_map(cam), "Grad-CAM weights channels by global-average-pooled gradients."


def compute_grad_cam_plus_plus(activation: np.ndarray, gradient: np.ndarray) -> Tuple[np.ndarray, str]:
    act = activation_to_chw(activation)
    grad = activation_to_chw(gradient)
    grad2 = grad * grad
    grad3 = grad2 * grad
    eps = 1e-8
    denominator = 2.0 * grad2 + np.sum(act * grad3, axis=(1, 2), keepdims=True)
    alpha = grad2 / (denominator + eps)
    weights = np.sum(alpha * np.maximum(grad, 0), axis=(1, 2), keepdims=True)
    cam = np.sum(weights * act, axis=0)
    cam = np.maximum(cam, 0)
    if float(cam.max()) <= 1e-8:
        cam = np.abs(cam)
    return normalize_map(cam), "Grad-CAM++ improves multi-instance localization by using higher-order gradient terms."


def compute_xgrad_cam(activation: np.ndarray, gradient: np.ndarray) -> Tuple[np.ndarray, str]:
    act = activation_to_chw(activation)
    grad = activation_to_chw(gradient)
    eps = 1e-8
    numerator = np.sum(grad * act, axis=(1, 2), keepdims=True)
    denominator = np.sum(act, axis=(1, 2), keepdims=True) + eps
    weights = numerator / denominator
    cam = np.sum(weights * act, axis=0)
    cam = np.maximum(cam, 0)
    if float(cam.max()) <= 1e-8:
        cam = np.abs(cam)
    return normalize_map(cam), "XGrad-CAM uses activation-normalized gradient weights."


def detection_density_map(
    detections: Sequence[Detection],
    image_shape: Tuple[int, int, int],
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    """Confidence-weighted Gaussian heatmap from detected boxes."""
    h, w = image_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.ogrid[:h, :w]

    used = 0
    for det in detections:
        if target_class is not None and det.cls != target_class:
            continue
        x1, y1, x2, y2 = det.xyxy
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        sigma_x = max(4.0, bw / 3.5)
        sigma_y = max(4.0, bh / 3.5)
        gaussian = np.exp(-(((xx - cx) ** 2) / (2 * sigma_x**2) + ((yy - cy) ** 2) / (2 * sigma_y**2)))
        heatmap += gaussian.astype(np.float32) * float(det.conf)
        used += 1

    if used == 0:
        heatmap[h // 2, w // 2] = 1.0
        note = "No detection matched the target class; a center marker was used only to avoid an empty visualization."
    else:
        note = "Detection density draws confidence-weighted Gaussian kernels from predicted boxes; it is not a CAM explanation."

    k = max(3, int(round(min(h, w) * 0.025)))
    if k % 2 == 0:
        k += 1
    heatmap = cv2.GaussianBlur(heatmap, (k, k), 0)
    return normalize_map(heatmap), note


def parse_names(names: Any) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def run_detection(
    yolo: Any,
    image_path: Path,
    imgsz: int,
    conf: float,
    target_class: Optional[int],
    device: str,
) -> List[Detection]:
    kwargs = dict(imgsz=imgsz, conf=conf, verbose=False, device=device)
    if target_class is not None:
        kwargs["classes"] = [target_class]
    results = yolo(str(image_path), **kwargs)
    names = parse_names(getattr(yolo, "names", {}))
    detections: List[Detection] = []
    if not results or len(results[0].boxes) == 0:
        return detections

    boxes = results[0].boxes
    for xyxy, cls, score in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
        cls_i = int(cls)
        detections.append(
            Detection(
                xyxy=[float(v) for v in xyxy],
                conf=float(score),
                cls=cls_i,
                name=names.get(cls_i, str(cls_i)),
            )
        )
    return detections


def choose_target_class(detections: Sequence[Detection], configured_class: Optional[int]) -> Optional[int]:
    if configured_class is not None:
        return configured_class
    if not detections:
        return None
    return max(detections, key=lambda det: det.conf).cls


def build_comparison_figure(
    image_bgr: np.ndarray,
    result_paths: Sequence[HeatmapResult],
    detections: Sequence[Detection],
    names: Dict[int, str],
    output_path: Path,
) -> None:
    tiles: List[Tuple[str, np.ndarray]] = []
    original = image_bgr.copy()
    if DRAW_BOXES and detections:
        original = draw_detections(original, detections, names)
    tiles.append(("Original + detections", original))

    for item in result_paths:
        overlay = cv2.imread(item.overlay_path, cv2.IMREAD_COLOR)
        if overlay is not None:
            tiles.append((item.method, overlay))

    if not tiles:
        return

    tile_w, tile_h = 420, 360
    label_h = 44
    cols = min(3, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    canvas = np.full((rows * (tile_h + label_h), cols * tile_w, 3), 245, dtype=np.uint8)

    for idx, (title, img) in enumerate(tiles):
        r, c = divmod(idx, cols)
        x0 = c * tile_w
        y0 = r * (tile_h + label_h)
        resized = resize_to_fit(img, tile_w, tile_h)
        y_img = y0 + label_h
        canvas[y_img : y_img + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        cv2.putText(
            canvas,
            title[:38],
            (x0 + 12, y0 + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def resize_to_fit(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def list_layers(model: Any) -> List[Dict[str, Any]]:
    layers = []
    for i, module in enumerate(list(getattr(model, "model", []))):
        params = 0
        try:
            params = int(sum(p.numel() for p in module.parameters()))
        except Exception:
            pass
        layers.append(
            {
                "index": i,
                "name": f"model.{i}",
                "class": module.__class__.__name__,
                "type": str(getattr(module, "type", "")),
                "from": getattr(module, "f", None),
                "params": params,
            }
        )
    return layers


class YOLO26HeatmapGenerator:
    """Generate CAM and detection-density heatmaps for YOLO26 detection models."""

    def __init__(
        self,
        model_path: Path | str,
        imgsz: int = IMG_SIZE,
        device: str = DEVICE,
        target_layer: str = TARGET_LAYER,
        conf: float = CONF_THRESHOLD,
    ) -> None:
        self.torch, YOLO = require_dependencies()
        self.model_path = Path(model_path)
        self.imgsz = int(imgsz)
        self.device = select_device(self.torch, device)
        self.conf = float(conf)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        print(f"[YOLO26Heatmap] Loading model: {self.model_path}")
        # Keep two model instances:
        # - self.yolo/self.model: unfused differentiable model for CAM.
        # - self.detector: normal prediction model for boxes.
        # Ultralytics prediction may fuse/strip one2many branches for inference,
        # which is excellent for speed but unsuitable for Grad-CAM.
        self.yolo = YOLO(str(self.model_path))
        self.detector = YOLO(str(self.model_path))
        self.names = parse_names(getattr(self.yolo, "names", {}))
        self.model = self.yolo.model.to(self.device)
        self.model.train()

        self.target_layers = find_target_layers(self.model, target_layer)
        self.layer_name = "+".join(name for name, _ in self.target_layers)
        self.activations: Dict[str, Any] = {}
        self.gradients: Dict[str, Any] = {}
        self._hook_handles = []
        for name, layer in self.target_layers:
            self._hook_handles.append(layer.register_forward_hook(self._make_forward_hook(name)))
        layer_desc = ", ".join(f"{name}({layer.__class__.__name__})" for name, layer in self.target_layers)
        print(f"[YOLO26Heatmap] Target layer(s): {layer_desc}")
        print(f"[YOLO26Heatmap] Device: {self.device}")

    def close(self) -> None:
        for handle in getattr(self, "_hook_handles", []):
            handle.remove()
        self._hook_handles = []

    def _make_forward_hook(self, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = first_tensor(output)
            if tensor is None:
                return
            self.activations[name] = tensor

            def save_gradient(grad: Any) -> None:
                self.gradients[name] = grad

            if getattr(tensor, "requires_grad", False):
                tensor.register_hook(save_gradient)

        return hook

    def _fused_cam(self, method_key: str) -> Tuple[np.ndarray, str]:
        cams: List[np.ndarray] = []
        notes: List[str] = []

        for name in sorted(self.activations.keys()):
            activation_np = tensor_to_numpy_4d(self.activations[name])
            gradient = self.gradients.get(name)
            gradient_np = tensor_to_numpy_4d(gradient) if gradient is not None else None

            if method_key == "eigen_cam":
                cam, note = compute_eigen_cam(activation_np)
            elif gradient_np is None:
                cam, note = compute_eigen_cam(activation_np)
                note = f"{method_key} fallback to EigenCAM on {name}: gradient was unavailable."
            elif method_key == "grad_cam":
                cam, note = compute_grad_cam(activation_np, gradient_np)
            elif method_key == "grad_cam_plus_plus":
                cam, note = compute_grad_cam_plus_plus(activation_np, gradient_np)
            elif method_key == "xgrad_cam":
                cam, note = compute_xgrad_cam(activation_np, gradient_np)
            else:
                raise ValueError(f"Unsupported CAM method: {method_key}")

            cam = cv2.resize(cam, (self.imgsz, self.imgsz), interpolation=cv2.INTER_CUBIC)
            cams.append(normalize_map(cam))
            notes.append(f"{name}: {note}")

        if not cams:
            raise RuntimeError("No CAM map could be computed from captured activations.")

        fused = normalize_map(np.mean(np.stack(cams, axis=0), axis=0))
        layer_note = f"Fused {len(cams)} target layer(s): {', '.join(sorted(self.activations.keys()))}. "
        return fused, layer_note + " ".join(notes)

    def generate(
        self,
        image_path: Path | str,
        output_dir: Path | str = OUTPUT_DIR,
        methods: Sequence[str] = METHODS,
        target_class: Optional[int] = TARGET_CLASS,
    ) -> Dict[str, Any]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        image_bgr = read_image_bgr(image_path)
        cv2.imwrite(str(output_dir / "original.jpg"), image_bgr)

        detections = run_detection(self.detector, image_path, self.imgsz, self.conf, target_class, self.device)
        self.model = self.yolo.model.to(self.device)
        self.model.train()
        target_class = choose_target_class(detections, target_class)

        detection_img = draw_detections(image_bgr, detections, self.names) if detections else image_bgr.copy()
        cv2.imwrite(str(output_dir / "detection_result.jpg"), detection_img)

        input_bgr, letterbox = letterbox_bgr(image_bgr, self.imgsz)
        input_tensor = image_to_tensor(self.torch, input_bgr, self.device)
        input_tensor.requires_grad_(True)

        self.activations = {}
        self.gradients = {}
        self.model.zero_grad(set_to_none=True)

        with self.torch.enable_grad():
            # Train mode returns the raw one2many branch with gradients in YOLO26.
            # We do not call optimizer.step(), so model weights are not changed.
            output = self.model(input_tensor)
            score = choose_cam_score(output, target_class)
            if not getattr(score, "requires_grad", False):
                raise RuntimeError(
                    "The selected CAM score has no gradient. Try --layer 22 or run with a PyTorch/Ultralytics "
                    "environment that supports differentiable YOLO26 forward."
                )
            score.backward(retain_graph=True)

        if not self.activations:
            raise RuntimeError(
                f"No activation was captured from {self.layer_name}. "
                "Try another target layer, for example --layer 22 or --layer 19."
            )

        results: List[HeatmapResult] = []
        for method in methods:
            method_key = method.lower().strip().replace("-", "_")
            if method_key == "eigen_cam":
                cam_small, note = self._fused_cam(method_key)
            elif method_key == "grad_cam":
                cam_small, note = self._fused_cam(method_key)
            elif method_key == "grad_cam_plus_plus":
                cam_small, note = self._fused_cam(method_key)
            elif method_key == "xgrad_cam":
                cam_small, note = self._fused_cam(method_key)
            elif method_key in {"detection_density", "box_density", "gaussian"}:
                cam, note = detection_density_map(detections, image_bgr.shape, target_class)
                results.append(save_heatmap_pair(image_bgr, cam, detections, self.names, output_dir, "detection_density", note))
                continue
            else:
                print(f"[YOLO26Heatmap] Skip unknown method: {method}")
                continue

            cam = crop_letterbox_to_original(cam_small, letterbox)
            results.append(save_heatmap_pair(image_bgr, cam, detections, self.names, output_dir, method_key, note))

        comparison_path = output_dir / "heatmap_methods_compare.jpg"
        build_comparison_figure(image_bgr, results, detections, self.names, comparison_path)

        metadata = {
            "model_path": str(self.model_path),
            "image_path": str(image_path),
            "output_dir": str(output_dir),
            "imgsz": self.imgsz,
            "device": self.device,
            "confidence_threshold": self.conf,
            "target_class": target_class,
            "target_class_name": self.names.get(target_class, None) if target_class is not None else None,
            "target_layer": self.layer_name,
            "detections": [asdict(det) for det in detections],
            "results": [asdict(item) for item in results],
            "comparison_path": str(comparison_path),
            "layers": list_layers(self.model),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"[YOLO26Heatmap] Done. Results saved to: {output_dir}")
        print(f"[YOLO26Heatmap] Comparison image: {comparison_path}")
        return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO26 CAM heatmaps and comparison figures.")
    parser.add_argument("--model", "-m", type=Path, default=MODEL_PATH, help="YOLO26 .pt model path.")
    parser.add_argument("--image", "-i", type=Path, default=IMAGE_PATH, help="Input image path.")
    parser.add_argument("--out", "-o", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Model input size.")
    parser.add_argument("--device", default=DEVICE, help="auto/cpu/cuda/cuda:0/0.")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Detection confidence threshold.")
    parser.add_argument("--class-id", type=int, default=TARGET_CLASS, help="Target class id. Omit for highest-confidence class.")
    parser.add_argument("--layer", default=TARGET_LAYER, help="17")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        help="Methods: eigen_cam grad_cam grad_cam_plus_plus xgrad_cam detection_density.",
    )
    parser.add_argument("--list-layers", action="store_true", help="Print model layers and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = YOLO26HeatmapGenerator(
        model_path=args.model,
        imgsz=args.imgsz,
        device=args.device,
        target_layer=args.layer,
        conf=args.conf,
    )
    try:
        if args.list_layers:
            for item in list_layers(generator.model):
                print(f"{item['index']:>3} {item['name']:<10} {item['class']:<18} params={item['params']}")
            return
        generator.generate(
            image_path=args.image,
            output_dir=args.out,
            methods=args.methods,
            target_class=args.class_id,
        )
    finally:
        generator.close()


if __name__ == "__main__":
    main()
