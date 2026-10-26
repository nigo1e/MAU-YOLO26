

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


NATURE_COLORS = {
    "ink": "#202124",
    "muted": "#68707a",
    "faint": "#d8dde3",
    "paper": "#fbfbf8",
    "white": "#ffffff",
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "purple": "#CC79A7",
}


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.strip().lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


RGB = {name: hex_to_rgb(value) for name, value in NATURE_COLORS.items()}


@dataclass
class LayerRun:
    index: int
    module: str
    section: str
    from_: Any
    params: int
    args: List[Any] = field(default_factory=list)
    input_shape: str = "unknown"
    output_shape: str = "unknown"
    input_map: Optional[np.ndarray] = None
    output_map: Optional[np.ndarray] = None
    input_grid: Optional[np.ndarray] = None
    output_grid: Optional[np.ndarray] = None
    gradcam_map: Optional[np.ndarray] = None
    gradcam_source: str = "Grad-CAM"
    kernel: Optional[np.ndarray] = None
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)
    notes: List[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"L{self.index:02d} - {self.module}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Nature-style clickable YOLO26 per-layer computation and Grad-CAM report."
    )
    parser.add_argument("positional", nargs="*", help="Optional shorthand: IMAGE WEIGHTS")
    parser.add_argument("--image", "-i", type=Path, help="Input image.")
    parser.add_argument("--weights", "-w", type=Path, help="YOLO26 .pt/.pth weight file.")
    parser.add_argument("--outdir", "-o", type=Path, default=Path("yolo26_gradcam_viz"))
    parser.add_argument("--imgsz", type=int, default=640, help="Square model input size.")
    parser.add_argument("--frames-per-layer", type=int, default=18)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=130, help="Controls GIF frame resolution.")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/0.")
    parser.add_argument("--max-layers", type=int, default=0, help="0 means all layers.")
    parser.add_argument("--open", action="store_true", help="Open generated report.")
    args = parser.parse_args()

    if args.positional:
        if len(args.positional) != 2:
            parser.error("Use shorthand as: IMAGE WEIGHTS")
        if args.image is None:
            args.image = Path(args.positional[0])
        if args.weights is None:
            args.weights = Path(args.positional[1])

    if args.image is None:
        parser.error("--image is required.")
    if args.weights is None:
        parser.error("--weights is required.")
    if args.frames_per_layer < 2:
        parser.error("--frames-per-layer must be at least 2.")
    if args.fps < 1:
        parser.error("--fps must be at least 1.")
    return args


def log(message: str) -> None:
    print(f"[yolo26-viz] {message}", flush=True)


def load_rgb_image(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def letterbox_rgb(image: np.ndarray, size: int, fill: Tuple[int, int, int] = (245, 245, 242)) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def pick_torch_device(torch_module: Any, request: str) -> str:
    if request == "auto":
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"
    if request.isdigit():
        return f"cuda:{request}"
    return request


def first_tensor(value: Any) -> Optional[Any]:
    if hasattr(value, "detach") and hasattr(value, "shape"):
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


def shape_summary(value: Any) -> str:
    if hasattr(value, "shape"):
        return "x".join(str(x) for x in tuple(value.shape))
    if isinstance(value, (list, tuple)):
        parts = [shape_summary(v) for v in value[:4]]
        suffix = "" if len(value) <= 4 else f", ... +{len(value) - 4}"
        return f"{type(value).__name__}[{'; '.join(parts)}{suffix}]"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    return type(value).__name__


def robust_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), copy=False)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-8:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if abs(hi - lo) < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def safe_resize(
    arr: np.ndarray,
    size: Tuple[int, int],
    interpolation: int = cv2.INTER_AREA,
    fallback_shape: Tuple[int, int] = (1, 1),
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size == 0 or arr.ndim < 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
        arr = np.zeros(fallback_shape, dtype=np.float32)
    return cv2.resize(arr.astype(np.float32), size, interpolation=interpolation)


def tensor_to_map(tensor_or_array: Any, target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if tensor_or_array is None:
        return None
    if hasattr(tensor_or_array, "detach"):
        arr = tensor_or_array.detach().float().cpu().numpy()
    else:
        arr = np.asarray(tensor_or_array)
    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr.astype(np.float32), copy=False)
    if arr.ndim >= 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] <= 4096 and arr.shape[1] > 1 and arr.shape[2] > 1:
            arr = np.mean(np.abs(arr), axis=0)
        else:
            arr = np.mean(np.abs(arr), axis=-1)
    elif arr.ndim == 2:
        arr = np.abs(arr)
    elif arr.ndim == 1:
        side = int(math.ceil(math.sqrt(arr.size)))
        padded = np.zeros(side * side, dtype=np.float32)
        padded[: arr.size] = np.abs(arr)
        arr = padded.reshape(side, side)
    else:
        arr = np.array([[float(np.mean(np.abs(arr)))]], dtype=np.float32)
    arr = safe_resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC)
    return robust_normalize(arr)


def tensor_to_grid(tensor_or_array: Any, size: int = 13) -> Optional[np.ndarray]:
    if tensor_or_array is None:
        return None
    if hasattr(tensor_or_array, "detach"):
        arr = tensor_or_array.detach().float().cpu().numpy()
    else:
        arr = np.asarray(tensor_or_array)
    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr.astype(np.float32), copy=False)
    if arr.ndim >= 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] <= 4096 and arr.shape[1] > 1 and arr.shape[2] > 1:
            arr = np.mean(arr, axis=0)
        else:
            arr = np.mean(arr, axis=-1)
    elif arr.ndim == 2:
        pass
    elif arr.ndim == 1:
        side = int(math.ceil(math.sqrt(arr.size)))
        padded = np.zeros(side * side, dtype=np.float32)
        padded[: arr.size] = arr
        arr = padded.reshape(side, side)
    else:
        arr = np.array([[float(np.mean(arr))]], dtype=np.float32)
    arr = safe_resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr.astype(np.float32)


def compute_gradcam(
    activation: Optional[np.ndarray],
    gradient: Optional[np.ndarray],
    target_hw: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], str]:
    if activation is None or gradient is None:
        return None, "activation map fallback"
    act = np.asarray(activation, dtype=np.float32)
    grad = np.asarray(gradient, dtype=np.float32)
    if act.size == 0 or grad.size == 0:
        return None, "activation map fallback"
    if act.ndim >= 4:
        act = act[0]
    if grad.ndim >= 4:
        grad = grad[0]
    try:
        if act.ndim == 3 and grad.ndim == 3 and act.shape == grad.shape:
            if act.shape[0] <= 4096:
                weights = grad.mean(axis=(1, 2), keepdims=True)
                cam = np.sum(weights * act, axis=0)
            else:
                weights = grad.mean(axis=(0, 1), keepdims=True)
                cam = np.sum(weights * act, axis=-1)
        elif act.ndim == 2 and grad.ndim == 2:
            cam = act * grad
        else:
            return None, "activation map fallback"
    except Exception:
        return None, "activation map fallback"
    cam = np.maximum(cam, 0)
    if float(np.max(cam)) <= 1e-8:
        cam = np.abs(cam)
    cam = safe_resize(cam, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC)
    return robust_normalize(cam), "Grad-CAM"


def module_display_name(module: Any) -> str:
    value = getattr(module, "type", None)
    text = str(value) if value else module.__class__.__name__
    if "." in text:
        text = text.split(".")[-1]
    return text.strip("<>'\"")


def count_parameters(module: Any) -> int:
    try:
        return int(sum(p.numel() for p in module.parameters()))
    except Exception:
        return 0


def infer_section(index: int, total: int, name: str) -> str:
    lowered = name.lower()
    if "detect" in lowered or "segment" in lowered or "pose" in lowered or index >= max(0, total - 6):
        return "head"
    return "backbone"


def extract_conv_kernel(module: Any) -> Optional[np.ndarray]:
    candidates = [module]
    for attr in ("conv", "cv1", "cv2"):
        child = getattr(module, attr, None)
        if child is not None:
            candidates.append(child)
    for candidate in candidates:
        weight = getattr(candidate, "weight", None)
        if weight is None and hasattr(candidate, "conv"):
            weight = getattr(getattr(candidate, "conv"), "weight", None)
        if weight is None or not hasattr(weight, "detach"):
            continue
        try:
            arr = weight.detach().float().cpu().numpy()
        except Exception:
            continue
        if arr.ndim >= 4:
            kernel = arr[0, 0]
            if kernel.ndim == 2 and 1 <= kernel.shape[0] <= 9 and 1 <= kernel.shape[1] <= 9:
                return kernel.astype(np.float32)
    return None


def to_pair(value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return (int(value[0]), int(value[0]))
    try:
        return (int(value), int(value))
    except Exception:
        return default


def extract_conv_stride_padding(module: Any) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    candidates = [module]
    for attr in ("conv", "cv1", "cv2"):
        child = getattr(module, attr, None)
        if child is not None:
            candidates.append(child)
    for candidate in candidates:
        if hasattr(candidate, "stride") or hasattr(candidate, "padding"):
            stride = to_pair(getattr(candidate, "stride", None), (1, 1))
            padding = to_pair(getattr(candidate, "padding", None), (0, 0))
            return stride, padding
        conv = getattr(candidate, "conv", None)
        if conv is not None and (hasattr(conv, "stride") or hasattr(conv, "padding")):
            stride = to_pair(getattr(conv, "stride", None), (1, 1))
            padding = to_pair(getattr(conv, "padding", None), (0, 0))
            return stride, padding
    return (1, 1), (0, 0)


def build_layer_records_from_modules(modules: Sequence[Any]) -> List[LayerRun]:
    records: List[LayerRun] = []
    total = len(modules)
    for i, module in enumerate(modules):
        name = module_display_name(module)
        stride, padding = extract_conv_stride_padding(module)
        records.append(
            LayerRun(
                index=i,
                module=name,
                section=infer_section(i, total, name),
                from_=getattr(module, "f", -1),
                params=count_parameters(module),
                args=[{"params": count_parameters(module)}],
                kernel=extract_conv_kernel(module),
                stride=stride,
                padding=padding,
            )
        )
    return records


def tensor_score(tensor: Any) -> Optional[Any]:
    if tensor is None or not hasattr(tensor, "requires_grad") or not tensor.requires_grad:
        return None
    if not hasattr(tensor, "reshape"):
        return None
    flat = tensor.float().reshape(-1)
    if flat.numel() == 0:
        return None
    score = flat.max()
    if not getattr(score, "requires_grad", False):
        return None
    return score


def choose_backward_target(output: Any, grad_targets: Dict[int, Any]) -> Tuple[Any, Optional[int]]:
    final_score = tensor_score(first_tensor(output))
    if final_score is not None:
        return final_score, None

    for layer_index in sorted(grad_targets.keys(), reverse=True):
        score = tensor_score(grad_targets[layer_index])
        if score is not None:
            return score, layer_index

    raise RuntimeError(
        "No differentiable tensor was found for Grad-CAM. "
        "Check whether the model forward is running under inference/no_grad mode."
    )


def capture_true_layer_runs(
    weights_path: Path,
    image: np.ndarray,
    device_request: str,
) -> Tuple[List[LayerRun], Dict[str, Any]]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch is required for true layer visualization: {exc}") from exc
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(f"Ultralytics is required for YOLO weight loading: {exc}") from exc

    device = pick_torch_device(torch, device_request)
    log(f"Loading YOLO weights on {device}: {weights_path}")
    yolo = YOLO(str(weights_path))
    model = yolo.model.to(device)
    model.eval()
    modules = list(getattr(model, "model", []))
    if not modules:
        raise RuntimeError("The loaded YOLO model does not expose model.model layers.")

    records = build_layer_records_from_modules(modules)
    activations: Dict[int, np.ndarray] = {}
    gradients: Dict[int, np.ndarray] = {}
    grad_targets: Dict[int, Any] = {}
    handles = []
    image_hw = image.shape[:2]

    def make_hook(index: int):
        def hook(_module: Any, inputs: Any, output: Any) -> None:
            record = records[index]
            record.input_shape = shape_summary(inputs)
            record.output_shape = shape_summary(output)
            input_tensor = first_tensor(inputs)
            output_tensor = first_tensor(output)
            record.input_map = tensor_to_map(input_tensor, image_hw)
            record.output_map = tensor_to_map(output_tensor, image_hw)
            record.input_grid = tensor_to_grid(input_tensor, size=13)
            record.output_grid = tensor_to_grid(output_tensor, size=13)
            if output_tensor is not None and getattr(output_tensor, "requires_grad", False):
                activations[index] = output_tensor.detach().float().cpu().numpy()
                grad_targets[index] = output_tensor

                def save_gradient(grad: Any, layer_index: int = index) -> None:
                    gradients[layer_index] = grad.detach().float().cpu().numpy()

                output_tensor.register_hook(save_gradient)

        return hook

    for index, module in enumerate(modules):
        handles.append(module.register_forward_hook(make_hook(index)))

    tensor = torch.from_numpy(image).to(device)
    tensor = tensor.float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor.requires_grad_(True)

    try:
        model.zero_grad(set_to_none=True)
        output = model(tensor)
        target_score, target_layer = choose_backward_target(output, grad_targets)
        log(f"Backward target score: {float(target_score.detach().cpu()):.6f}")
        if target_layer is not None:
            log(f"Backward target layer: L{target_layer:02d}")
        target_score.backward()
    finally:
        for handle in handles:
            handle.remove()

    for record in records:
        gradcam, source = compute_gradcam(activations.get(record.index), gradients.get(record.index), image_hw)
        record.gradcam_map = gradcam if gradcam is not None else record.output_map
        record.gradcam_source = source if gradcam is not None else "output activation fallback"
        if record.input_map is None:
            record.input_map = tensor_to_map(image, image_hw)
        if record.input_grid is None:
            record.input_grid = tensor_to_grid(image, size=13)
        if record.output_map is None:
            record.output_map = record.input_map
        if record.output_grid is None:
            record.output_grid = record.input_grid
        if record.gradcam_map is None:
            record.gradcam_map = record.output_map
        record.notes = layer_notes(record)

    names = getattr(model, "names", None)
    cfg = {
        "layers": len(records),
        "nc": len(names) if isinstance(names, (list, tuple, dict)) else None,
        "names": names if isinstance(names, (list, tuple, dict)) else None,
        "weights": str(weights_path),
    }
    return records, cfg


def source_description(record: LayerRun) -> str:
    if record.from_ == -1:
        if record.index == 0:
            return "原始输入图片张量"
        return f"上一层 L{record.index - 1:02d} 的真实输出"
    if isinstance(record.from_, (list, tuple)):
        return "多个历史层的真实输出: " + ", ".join(str(x) for x in record.from_)
    return f"来自层索引 {record.from_} 的真实输出"


def layer_notes(record: LayerRun) -> List[str]:
    module = record.module.lower()
    notes = [
        "计算机视觉领域精选UP主推荐！",
        "本代码为UP主「Ai学术叫叫兽」免费开源的粉丝专属资源，仅用于科研教学展示。",
        "官方微信公众号：Ai计算机视觉。",
        f"输入来源: {source_description(record)}。本脚本通过 forward hook 捕获该层真实输入，而不是用演示图替代。",
        f"真实输入形状: {record.input_shape}; 真实输出形状: {record.output_shape}。",
    ]
    if "conv" in module:
        notes.extend(
            [
                "卷积计算: 上一层输出的特征图作为输入，卷积核在空间位置上滑动；每个窗口与卷积核逐项相乘，然后求和生成输出特征图中的一个位置。",
                "动画下方的数学运算区使用真实输入特征的小网格、权重文件中的卷积核、逐项乘积和真实输出格值，展示当前层如何把输入写成输出。",
                "注意: YOLO卷积层后通常还会接BN/激活函数，因此 sum(patch × kernel) 与最终输出格值可能不同；页面会同时展示卷积求和与真实forward输出。",
            ]
        )
    elif "concat" in module:
        notes.append("拼接计算: 多个来源层的真实输出在通道维度合并，空间位置保持对齐，形成更丰富的特征表示。")
    elif "upsample" in module:
        notes.append("上采样计算: 真实输入特征图被插值放大，输出保留语义信息并提高空间分辨率。")
    elif "spp" in module or "pool" in module:
        notes.append("池化计算: 局部区域被最大值或平均值概括，帮助当前层聚合更大范围的上下文。")
    elif "detect" in module:
        notes.append("检测头计算: 当前层将特征解释为目标框、类别和置信度等检测证据。")
    else:
        notes.append("层计算: 当前模块对真实输入特征进行变换、路由或混合，并把结果传递给下一层。")
    notes.append(
        f"Grad-CAM说明: 热力图由该层激活和反向传播梯度共同计算，表示当前层对最终目标响应最敏感的空间区域。来源: {record.gradcam_source}。"
    )
    notes.append("版式说明: 上半部分是人眼可读的真实输入/输出特征图，下半部分是计算机视角的数学运算过程，两者用于对比同一层的可视化结果和数值计算。")
    return notes


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    windir = Path("C:/Windows/Fonts")
    candidates = (
        [windir / "msyhbd.ttc", windir / "simhei.ttf", windir / "segoeuib.ttf", windir / "arialbd.ttf"]
        if bold
        else [windir / "msyh.ttc", windir / "simhei.ttf", windir / "segoeui.ttf", windir / "arial.ttf"]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), str(text), font=font)
    return int(box[2] - box[0])


def ellipsize(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    text = str(text)
    if text_width(draw, text, font) <= max_width:
        return text
    suffix = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if text_width(draw, text[:mid] + suffix, font) <= max_width:
            lo = mid + 1
        else:
            hi = mid
    return text[: max(0, lo - 1)] + suffix


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
    max_width: Optional[int] = None,
) -> None:
    if max_width is not None:
        text = ellipsize(draw, text, font, max_width)
    draw.text(xy, text, font=font, fill=fill)


def rgba_on_white(color: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return tuple(int(round(c * alpha + 255 * (1.0 - alpha))) for c in color)


def cividis_like_heatmap(values: np.ndarray) -> np.ndarray:
    anchors = np.array(
        [
            [0.00, 32, 44, 63],
            [0.18, 43, 74, 103],
            [0.38, 75, 104, 122],
            [0.58, 122, 127, 113],
            [0.78, 179, 155, 92],
            [1.00, 240, 228, 66],
        ],
        dtype=np.float32,
    )
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    flat = values.reshape(-1)
    out = np.zeros((flat.size, 3), dtype=np.float32)
    for channel in range(3):
        out[:, channel] = np.interp(flat, anchors[:, 0], anchors[:, channel + 1])
    return out.reshape(values.shape + (3,)).astype(np.uint8)


def blend_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.78) -> Image.Image:
    base = image.astype(np.float32)
    heat = cividis_like_heatmap(heatmap).astype(np.float32)
    alpha_map = (np.clip(heatmap, 0.0, 1.0) ** 0.7) * alpha
    blended = base * (1.0 - alpha_map[..., None]) + heat * alpha_map[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def resize_np_to_box(arr: np.ndarray, box_w: int, box_h: int, fill: Tuple[int, int, int]) -> Image.Image:
    img = Image.fromarray(arr)
    scale = min(box_w / img.width, box_h / img.height)
    nw, nh = max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale)))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (box_w, box_h), fill)
    canvas.paste(resized, ((box_w - nw) // 2, (box_h - nh) // 2))
    return canvas


def draw_panel_border(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int]) -> None:
    draw.rectangle(box, outline=RGB["faint"], width=1)


def reveal_map(base_map: np.ndarray, frame_index: int, frame_count: int) -> np.ndarray:
    h, w = base_map.shape
    progress = (frame_index + 1) / max(frame_count, 1)
    yy, xx = np.mgrid[0:h, 0:w]
    sweep = 0.72 * (xx / max(w - 1, 1)) + 0.28 * (yy / max(h - 1, 1))
    mask = 1.0 / (1.0 + np.exp((sweep - progress) * 18.0))
    return robust_normalize(base_map * mask)


def draw_arrow(draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    draw.line([start, end], fill=color, width=2)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    for offset in (math.pi * 0.78, -math.pi * 0.78):
        p = (end[0] + int(math.cos(angle + offset) * 9), end[1] + int(math.sin(angle + offset) * 9))
        draw.line([end, p], fill=color, width=2)


def draw_matrix(
    draw: ImageDraw.ImageDraw,
    matrix: np.ndarray,
    box: Tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    highlight: Optional[Tuple[int, int]] = None,
    show_numbers: bool = True,
) -> None:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        arr = np.array([[float(np.mean(arr))]], dtype=np.float32)
    rows, cols = arr.shape
    max_abs = max(float(np.max(np.abs(arr))), 1e-6)
    x0, y0, x1, y1 = box
    cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
    for r in range(rows):
        for c in range(cols):
            val = float(arr[r, c])
            color = RGB["blue"] if val >= 0 else RGB["orange"]
            fill = rgba_on_white(color, min(abs(val) / max_abs, 1.0) * 0.65)
            cx0, cy0 = int(x0 + c * cw), int(y0 + r * ch)
            cx1, cy1 = int(x0 + (c + 1) * cw), int(y0 + (r + 1) * ch)
            draw.rectangle((cx0, cy0, cx1, cy1), fill=fill, outline=RGB["white"], width=1)
            if show_numbers and rows <= 5 and cols <= 5:
                text = f"{val:+.2f}"
                tw = text_width(draw, text, font)
                th = draw.textbbox((0, 0), text, font=font)[3]
                draw.text((cx0 + (cx1 - cx0 - tw) / 2, cy0 + (cy1 - cy0 - th) / 2), text, font=font, fill=RGB["ink"])
            if highlight is not None and (r, c) == highlight:
                draw.rectangle((cx0, cy0, cx1, cy1), outline=RGB["orange"], width=3)
    draw_panel_border(draw, box)


def display_kernel(record: LayerRun) -> Optional[np.ndarray]:
    if record.kernel is None:
        return None
    kernel = np.asarray(record.kernel, dtype=np.float32)
    if kernel.ndim != 2:
        kernel = np.squeeze(kernel)
    if kernel.ndim != 2:
        return None
    if kernel.shape[0] > 5 or kernel.shape[1] > 5:
        cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2
        kernel = kernel[max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2]
    if kernel.shape != (3, 3):
        kernel = safe_resize(kernel, (3, 3), interpolation=cv2.INTER_AREA)
    return kernel.astype(np.float32)


def extract_real_patch(record: LayerRun, out_y: int, out_x: int, kernel_shape: Tuple[int, int]) -> np.ndarray:
    grid = record.input_grid
    if grid is None or grid.size == 0 or grid.ndim != 2:
        return np.zeros(kernel_shape, dtype=np.float32)
    ky, kx = kernel_shape
    ky = max(1, int(ky))
    kx = max(1, int(kx))
    sy, sx = record.stride
    py, px = record.padding
    center_y = out_y * sy - py + ky // 2
    center_x = out_x * sx - px + kx // 2
    padded = np.pad(grid, ((ky, ky), (kx, kx)), mode="edge")
    start_y = int(center_y + ky - ky // 2)
    start_x = int(center_x + kx - kx // 2)
    start_y = int(np.clip(start_y, 0, max(0, padded.shape[0] - ky)))
    start_x = int(np.clip(start_x, 0, max(0, padded.shape[1] - kx)))
    patch = padded[start_y : start_y + ky, start_x : start_x + kx]
    if patch.shape != (ky, kx):
        patch = safe_resize(patch.astype(np.float32), (kx, ky), interpolation=cv2.INTER_AREA, fallback_shape=(ky, kx))
    return patch.astype(np.float32)


def draw_computation_detail(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    record: LayerRun,
    frame_index: int,
    frame_count: int,
    small_font: ImageFont.ImageFont,
    tiny_font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    pad = 14
    draw.rounded_rectangle(box, radius=6, fill=RGB["white"], outline=RGB["faint"], width=1)
    draw_text(draw, (x0 + pad, y0 + 10), "计算机数学运算", small_font, RGB["ink"], max_width=x1 - x0 - pad * 2)

    out_grid = record.output_grid if record.output_grid is not None else np.zeros((13, 13), dtype=np.float32)
    rows, cols = out_grid.shape
    total = max(1, rows * cols)
    pos = min(total - 1, int((frame_index / max(frame_count - 1, 1)) * total))
    oy, ox = divmod(pos, cols)
    kernel = display_kernel(record)

    col_gap = 14
    usable_w = x1 - x0 - pad * 2 - col_gap * 3
    cell_w = usable_w // 4
    top = y0 + 48
    m_h = min(cell_w, y1 - top - 44)
    lefts = [x0 + pad + i * (cell_w + col_gap) for i in range(4)]

    if kernel is not None:
        patch = extract_real_patch(record, oy, ox, kernel.shape)
        products = patch * kernel
        conv_value = float(np.sum(products))
        output_value = float(out_grid[oy, ox])
        draw_text(draw, (lefts[0], top - 20), "真实输入patch", tiny_font, RGB["muted"], max_width=cell_w)
        draw_text(draw, (lefts[1], top - 20), "真实权重核", tiny_font, RGB["muted"], max_width=cell_w)
        draw_text(draw, (lefts[2], top - 20), "逐项乘积", tiny_font, RGB["muted"], max_width=cell_w)
        draw_text(draw, (lefts[3], top - 20), "输出写入", tiny_font, RGB["muted"], max_width=cell_w)
        draw_matrix(draw, patch, (lefts[0], top, lefts[0] + cell_w, top + m_h), tiny_font)
        draw_matrix(draw, kernel, (lefts[1], top, lefts[1] + cell_w, top + m_h), tiny_font)
        draw_matrix(draw, products, (lefts[2], top, lefts[2] + cell_w, top + m_h), tiny_font)
        draw_matrix(draw, out_grid, (lefts[3], top, lefts[3] + cell_w, top + m_h), tiny_font, highlight=(oy, ox), show_numbers=False)
        formula = f"sum(patch × kernel)={conv_value:+.3f}; after layer ops -> output[{oy},{ox}]={output_value:+.3f}"
    else:
        input_grid = record.input_grid if record.input_grid is not None else np.zeros((13, 13), dtype=np.float32)
        draw_text(draw, (lefts[0], top - 20), "真实输入", tiny_font, RGB["muted"], max_width=cell_w)
        draw_text(draw, (lefts[1], top - 20), "层运算", tiny_font, RGB["muted"], max_width=cell_w)
        draw_text(draw, (lefts[2], top - 20), "真实输出", tiny_font, RGB["muted"], max_width=cell_w)
        draw_text(draw, (lefts[3], top - 20), "当前位置", tiny_font, RGB["muted"], max_width=cell_w)
        draw_matrix(draw, input_grid, (lefts[0], top, lefts[0] + cell_w, top + m_h), tiny_font, show_numbers=False)
        draw_text(draw, (lefts[1], top + m_h // 2 - 10), record.module, small_font, RGB["blue"], max_width=cell_w)
        draw_matrix(draw, out_grid, (lefts[2], top, lefts[2] + cell_w, top + m_h), tiny_font, highlight=(oy, ox), show_numbers=False)
        draw_matrix(draw, out_grid, (lefts[3], top, lefts[3] + cell_w, top + m_h), tiny_font, highlight=(oy, ox), show_numbers=False)
        formula = f"真实输入经 {record.module} 变换后写入 output[{oy},{ox}]={float(out_grid[oy, ox]):+.3f}"

    draw_text(draw, (x0 + pad, y1 - 30), formula, tiny_font, RGB["ink"], max_width=x1 - x0 - pad * 2)


def draw_operation_panel(
    draw: ImageDraw.ImageDraw,
    frame: Image.Image,
    box: Tuple[int, int, int, int],
    record: LayerRun,
    frame_index: int,
    frame_count: int,
    label_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    tiny_font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    pad = 18
    draw_text(draw, (x0, y0 - 30), "A  真实输入输出特征 + 数学运算", label_font, RGB["ink"])
    draw_panel_border(draw, box)

    input_map = record.input_map if record.input_map is not None else np.zeros((64, 64), dtype=np.float32)
    output_map = record.output_map if record.output_map is not None else input_map
    progressive_output = reveal_map(output_map, frame_index, frame_count)
    map_w = int((x1 - x0 - pad * 2 - 42) * 0.42)
    map_h = int((y1 - y0 - pad * 2) * 0.38)
    map_h = min(map_h, map_w)
    top = y0 + pad + 48
    in_left = x0 + pad
    out_left = x1 - pad - map_w
    mid_left = in_left + map_w + 22
    mid_right = out_left - 22

    input_img = resize_np_to_box(cividis_like_heatmap(input_map), map_w, map_h, RGB["paper"])
    output_img = resize_np_to_box(cividis_like_heatmap(progressive_output), map_w, map_h, RGB["paper"])
    frame.paste(input_img, (in_left, top))
    frame.paste(output_img, (out_left, top))
    draw_panel_border(draw, (in_left, top, in_left + map_w, top + map_h))
    draw_panel_border(draw, (out_left, top, out_left + map_w, top + map_h))
    draw_text(draw, (in_left, y0 + pad), "真实输入特征", small_font, RGB["ink"], max_width=map_w)
    draw_text(draw, (in_left, y0 + pad + 22), source_description(record), tiny_font, RGB["muted"], max_width=map_w)
    draw_text(draw, (out_left, y0 + pad), "真实输出特征", small_font, RGB["ink"], max_width=map_w)
    draw_text(draw, (out_left, y0 + pad + 22), f"output: {record.output_shape}", tiny_font, RGB["muted"], max_width=map_w)

    arrow_y = top + map_h // 2
    draw_arrow(draw, (in_left + map_w + 8, arrow_y), (out_left - 8, arrow_y), RGB["blue"])
    draw_text(draw, (mid_left, arrow_y - 40), record.module, small_font, RGB["blue"], max_width=max(40, mid_right - mid_left))
    draw_text(draw, (mid_left, arrow_y - 18), "forward真实计算", tiny_font, RGB["muted"], max_width=max(40, mid_right - mid_left))

    detail_top = top + map_h + 42
    detail_box = (x0 + pad, detail_top, x1 - pad, y1 - pad - 52)
    if detail_box[3] > detail_box[1] + 80:
        draw_computation_detail(draw, detail_box, record, frame_index, frame_count, small_font, tiny_font)

    progress = (frame_index + 1) / frame_count
    bar = (x0 + pad, y1 - pad - 18, x1 - pad, y1 - pad - 6)
    draw.rectangle(bar, fill=rgba_on_white(RGB["blue"], 0.10), outline=RGB["faint"], width=1)
    draw.rectangle((bar[0], bar[1], bar[0] + int((bar[2] - bar[0]) * progress), bar[3]), fill=RGB["blue"])
    draw_text(draw, (x0 + pad, y1 - pad - 46), "上方是人眼可视化；下方是当前输出位置对应的真实数学运算。", tiny_font, RGB["muted"])


def draw_gradcam_panel(
    draw: ImageDraw.ImageDraw,
    frame: Image.Image,
    box: Tuple[int, int, int, int],
    image: np.ndarray,
    record: LayerRun,
    label_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    tiny_font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    pad = 16
    draw_text(draw, (x0, y0 - 30), "B  特征提取图与Grad-CAM", label_font, RGB["ink"])
    draw_panel_border(draw, box)
    feature = record.output_map if record.output_map is not None else np.zeros(image.shape[:2], dtype=np.float32)
    gradcam = record.gradcam_map if record.gradcam_map is not None else feature
    visual_h = (y1 - y0 - pad * 2 - 72) // 2
    visual_w = x1 - x0 - pad * 2
    top1 = y0 + pad + 26
    top2 = top1 + visual_h + 42
    feature_img = resize_np_to_box(cividis_like_heatmap(feature), visual_w, visual_h, RGB["paper"])
    grad_img = resize_np_to_box(np.asarray(blend_heatmap(image, gradcam, 0.82)), visual_w, visual_h, RGB["paper"])
    frame.paste(feature_img, (x0 + pad, top1))
    frame.paste(grad_img, (x0 + pad, top2))
    draw_panel_border(draw, (x0 + pad, top1, x0 + pad + feature_img.width, top1 + feature_img.height))
    draw_panel_border(draw, (x0 + pad, top2, x0 + pad + grad_img.width, top2 + grad_img.height))
    draw_text(draw, (x0 + pad, y0 + pad), "当前层特征提取图", small_font, RGB["ink"], max_width=visual_w)
    draw_text(draw, (x0 + pad, top2 - 24), "Grad-CAM目标响应热力图", small_font, RGB["ink"], max_width=visual_w)
    draw_text(draw, (x0 + pad, y1 - pad - 20), f"source: {record.gradcam_source}", tiny_font, RGB["blue"], max_width=visual_w)


def draw_layer_strip(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    records: Sequence[LayerRun],
    current: LayerRun,
    scale: float,
    label_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    margin = int(round(48 * scale))
    top = height - int(round(118 * scale))
    y = top + int(round(58 * scale))
    left, right = margin + int(round(42 * scale)), width - margin
    draw_text(draw, (margin, top), "C  网络执行路径", label_font, RGB["ink"])
    draw.line([(left, y), (right, y)], fill=RGB["faint"], width=max(1, int(2 * scale)))
    xs = [int((left + right) / 2)] if len(records) == 1 else [int(v) for v in np.linspace(left, right, len(records))]
    radius = max(4, int(round((9 if len(records) <= 48 else 6) * scale)))
    for x, rec in zip(xs, records):
        active = rec.index == current.index
        color = RGB["orange"] if "detect" in rec.module.lower() else RGB["blue"]
        if active:
            draw.ellipse((x - radius * 2, y - radius * 2, x + radius * 2, y + radius * 2), fill=rgba_on_white(color, 0.14))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=RGB["white"], outline=color, width=2)
            draw.ellipse((x - radius // 2, y - radius // 2, x + radius // 2, y + radius // 2), fill=color)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=RGB["white"], outline=RGB["faint"], width=1)
    current_x = xs[min(current.index, len(xs) - 1)]
    tag = f"L{current.index:02d}  {current.module}"
    tag_w = min(int(360 * scale), max(int(170 * scale), text_width(draw, tag, small_font) + int(34 * scale)))
    tag_x = int(np.clip(current_x - tag_w / 2, margin, width - margin - tag_w))
    tag_y = top + int(82 * scale)
    draw.rounded_rectangle((tag_x, tag_y, tag_x + tag_w, tag_y + int(30 * scale)), radius=5, fill=RGB["white"], outline=RGB["blue"])
    draw_text(draw, (tag_x + int(14 * scale), tag_y + int(6 * scale)), tag, small_font, RGB["ink"], max_width=tag_w - int(28 * scale))


def render_frame(
    image: np.ndarray,
    record: LayerRun,
    records: Sequence[LayerRun],
    frame_index: int,
    frame_count: int,
    dpi: int,
) -> np.ndarray:
    width = max(1200, int(14.8 * dpi))
    height = max(720, int(8.4 * dpi))
    scale = width / 2220.0
    margin = int(48 * scale)
    gap = int(30 * scale)
    title_h = int(150 * scale)
    flow_h = int(128 * scale)
    panel_top = title_h + int(12 * scale)
    panel_bottom = height - flow_h - int(28 * scale)
    panel_h = panel_bottom - panel_top
    left_w = int((width - margin * 2 - gap) * 0.62)
    right_w = width - margin * 2 - gap - left_w
    left_box = (margin, panel_top, margin + left_w, panel_top + panel_h)
    right_box = (left_box[2] + gap, panel_top, left_box[2] + gap + right_w, panel_top + panel_h)

    frame = Image.new("RGB", (width, height), RGB["white"])
    draw = ImageDraw.Draw(frame)
    title_font = load_font(max(18, int(30 * scale)), bold=True)
    subtitle_font = load_font(max(10, int(15 * scale)))
    label_font = load_font(max(10, int(16 * scale)), bold=True)
    small_font = load_font(max(8, int(13 * scale)))
    tiny_font = load_font(max(7, int(10 * scale)))

    draw_text(draw, (margin, int(42 * scale)), f"YOLO26真实逐层运算与Grad-CAM  |  {record.title}", title_font, RGB["ink"], width - margin * 2)
    subtitle = f"input: {record.input_shape}   |   output: {record.output_shape}   |   from={record.from_}   |   params={record.params}"
    draw_text(draw, (margin, int(92 * scale)), subtitle, subtitle_font, RGB["muted"], width - margin * 2)
    draw.line([(margin, int(132 * scale)), (width - margin, int(132 * scale))], fill=RGB["faint"], width=1)

    draw_operation_panel(draw, frame, left_box, record, frame_index, frame_count, label_font, small_font, tiny_font)
    draw_gradcam_panel(draw, frame, right_box, image, record, label_font, small_font, tiny_font)
    draw_layer_strip(draw, width, height, records, record, scale, label_font, small_font)
    return np.asarray(frame)


def render_layer_animation(
    out_path: Path,
    image: np.ndarray,
    record: LayerRun,
    records: Sequence[LayerRun],
    frames_per_layer: int,
    fps: int,
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, mode="I", duration=1.0 / fps, loop=0) as writer:
        for i in range(frames_per_layer):
            writer.append_data(render_frame(image, record, records, i, frames_per_layer, dpi))


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_html_report(
    outdir: Path,
    image_path: Path,
    weights_path: Path,
    records: Sequence[LayerRun],
    cfg: Dict[str, Any],
) -> Path:
    source_dir = outdir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    image_copy = source_dir / image_path.name
    shutil.copy2(image_path, image_copy)

    layers_payload = []
    for rec in records:
        layers_payload.append(
            {
                "index": rec.index,
                "module": rec.module,
                "section": rec.section,
                "from": str(rec.from_),
                "params": rec.params,
                "inputShape": rec.input_shape,
                "outputShape": rec.output_shape,
                "gradcamSource": rec.gradcam_source,
                "gif": f"assets/layer_{rec.index:03d}.gif",
                "notes": rec.notes,
            }
        )
    payload = {
        "layers": layers_payload,
        "image": relpath(image_copy, outdir),
        "weights": str(weights_path),
        "cfg": cfg,
    }

    html_path = outdir / "index.html"
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"># 详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO26真实逐层运算与Grad-CAM</title>
  <style>
    :root {{
      --ink: #202124; --muted: #68707a; --faint: #d8dde3;
      --paper: #fbfbf8; --white: #ffffff; --blue: #0072B2;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background: var(--paper);
      font-family: Inter, "Microsoft YaHei", "Segoe UI", sans-serif; line-height: 1.45; }}
    header {{ padding: 24px 32px 18px; background: var(--white); border-bottom: 1px solid var(--faint); }}
    h1 {{ margin: 0; font-size: clamp(24px, 3vw, 38px); letter-spacing: 0; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 10px 18px; margin-top: 10px; color: var(--muted); font-size: 13px; }}
    .layout {{ display: grid; grid-template-columns: minmax(260px, 360px) minmax(0, 1fr); min-height: calc(100vh - 100px); }}
    aside {{ background: var(--white); border-right: 1px solid var(--faint); padding: 18px; overflow: auto; max-height: calc(100vh - 100px); }}
    main {{ padding: 22px; overflow: auto; }}
    .notice {{ margin-bottom: 14px; padding: 10px 12px; border-left: 3px solid var(--blue); background: #fff; color: var(--muted); font-size: 13px; }}
    .layer-list {{ display: grid; gap: 8px; }}    # 详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽
    .layer-button {{ width: 100%; border: 1px solid var(--faint); background: #fff; color: var(--ink); padding: 10px 11px; border-radius: 8px; cursor: pointer; text-align: left; }}
    .layer-button:hover {{ border-color: rgba(0, 114, 178, .45); box-shadow: 0 4px 18px rgba(32, 33, 36, .07); }}
    .layer-button.active {{ border-color: var(--blue); box-shadow: inset 3px 0 0 var(--blue), 0 5px 18px rgba(0, 114, 178, .11); }}
    .row1 {{ display: flex; justify-content: space-between; gap: 10px; font-weight: 680; font-size: 13px; }}
    .row2 {{ margin-top: 4px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .stage, .notes {{ background: var(--white); border: 1px solid var(--faint); border-radius: 8px; padding: 14px; }}
    .stage img {{ display: block; width: 100%; height: auto; border: 1px solid var(--faint); background: #fff; }}
    .caption {{ margin-top: 12px; display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: start; }}
    .caption h2 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
    .caption p {{ margin: 4px 0 0; color: var(--muted); font-size: 13px; }}
    .controls {{ display: flex; gap: 8px; }}
    .controls button {{ border: 1px solid var(--faint); background: var(--white); color: var(--ink); border-radius: 8px; padding: 8px 11px; cursor: pointer; }}
    .notes {{ margin-top: 14px; }}
    .notes h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .notes ol {{ margin: 0; padding-left: 22px; color: var(--muted); font-size: 13px; }}
    .notes li + li {{ margin-top: 6px; }}
    code {{ font-family: Consolas, "Liberation Mono", monospace; }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      aside {{ max-height: none; border-right: 0; border-bottom: 1px solid var(--faint); }}
      main {{ padding: 14px; }}
      .caption {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>YOLO26真实逐层运算与Grad-CAM</h1>
    <div class="meta">
      <span>layers: <code>{len(records)}</code></span>
      <span>image: <code>{html.escape(image_path.name)}</code></span>
      <span>weights: <code>{html.escape(str(weights_path))}</code></span>
    </div>
  </header>
  <div class="layout">
    <aside>
      <div class="notice">点击任意层，查看该层真实输入、真实输出、特征提取图和Grad-CAM热力图。</div>
      <div class="layer-list" id="layerList"></div>
    </aside>
    <main>
      <section class="stage">
        <img id="layerImage" alt="Layer animation">
        <div class="caption">
          <div>
            <h2 id="layerTitle"></h2>
            <p id="layerText"></p>
          </div>
          <div class="controls">
            <button id="prevBtn" type="button">Prev</button>
            <button id="nextBtn" type="button">Next</button>
          </div>
        </div>
      </section>
      <section class="notes">
        <h3>注释</h3>
        <ol id="layerNotes"></ol>
      </section>
    </main>
  </div>
  <script>
    const report = {json.dumps(payload, ensure_ascii=False)};
    const listEl = document.getElementById('layerList');
    const imgEl = document.getElementById('layerImage');
    const titleEl = document.getElementById('layerTitle');
    const textEl = document.getElementById('layerText');
    const notesEl = document.getElementById('layerNotes');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    let activeIndex = 0;

    function label(layer) {{
      return `L${{String(layer.index).padStart(2, '0')}} - ${{layer.module}}`;
    }}
    function selectLayer(index) {{
      activeIndex = Math.max(0, Math.min(report.layers.length - 1, index));
      const layer = report.layers[activeIndex];
      imgEl.src = layer.gif + '?layer=' + activeIndex;
      titleEl.textContent = label(layer);
      textEl.textContent = `input=${{layer.inputShape}} | output=${{layer.outputShape}} | Grad-CAM=${{layer.gradcamSource}}`;
      notesEl.innerHTML = '';
      (layer.notes || []).forEach(note => {{
        const li = document.createElement('li');
        li.textContent = note;
        notesEl.appendChild(li);
      }});
      document.querySelectorAll('.layer-button').forEach((btn, i) => btn.classList.toggle('active', i === activeIndex));
    }}
    function buildList() {{
      report.layers.forEach((layer, index) => {{
        const btn = document.createElement('button');
        btn.className = 'layer-button';
        btn.type = 'button';
        btn.innerHTML = `<div class="row1"><span>${{label(layer)}}</span><span>${{layer.section}}</span></div>
          <div class="row2">from=${{layer.from}} - output=${{layer.outputShape}}</div>`;
        btn.addEventListener('click', () => selectLayer(index));
        listEl.appendChild(btn);
      }});
    }}
    prevBtn.addEventListener('click', () => selectLayer(activeIndex - 1));
    nextBtn.addEventListener('click', () => selectLayer(activeIndex + 1));
    document.addEventListener('keydown', event => {{
      if (event.key === 'ArrowLeft') selectLayer(activeIndex - 1);
      if (event.key === 'ArrowRight') selectLayer(activeIndex + 1);
    }});
    buildList();
    selectLayer(0);
  </script>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    weights_path = args.weights.resolve()
    outdir = args.outdir.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    asset_dir = outdir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    image = letterbox_rgb(load_rgb_image(image_path), args.imgsz)
    records, cfg = capture_true_layer_runs(weights_path, image, args.device)
    if args.max_layers > 0:
        records = records[: args.max_layers]

    for i, record in enumerate(records):
        log(f"Rendering layer {i + 1}/{len(records)}: {record.module}")
        render_layer_animation(
            asset_dir / f"layer_{record.index:03d}.gif",
            image,
            record,
            records,
            args.frames_per_layer,
            args.fps,
            args.dpi,
        )

    html_path = write_html_report(outdir, image_path, weights_path, records, cfg)
    log(f"Done in {time.time() - started:.1f}s.")
    log(f"Open report: {html_path}")
    if args.open:
        webbrowser.open(html_path.as_uri())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"[yolo26-viz] ERROR: {exc}", flush=True)
        if "torch" in str(exc).lower() or "ultralytics" in str(exc).lower():
            print("[yolo26-viz] Install dependencies: pip install torch ultralytics", flush=True)
        raise SystemExit(1)
