from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

# =============================================================================
# Common configuration: edit this block, then run:
#
#     python plot_loss_landscape.py
#
# Command-line arguments are still supported and will override this block.
# =============================================================================

WEIGHTS_PATH = ROOT / r"yolo26n.pt"
DATA_PATH = ROOT / r"datasets\Aijiaojioashou\data.yaml"
OUTPUT_ROOT = ROOT / "runs" / "loss_landscape"
RUN_NAME = "code_config_loss_landscape"

# auto: infer original/ydy from weight path; original: YOLO loss; ydy: fused YDYloss.
LOSS_PROFILE = "auto"

# Plot quality/speed settings.
STEPS = 9  # 9 is quick; use 21 or 31 for paper-grade figures.
RADIUS = 0.05
MAX_BATCHES = 16  # 0 means full split.
BATCH = 16
IMG_SIZE = 640
SPLIT = "val"
DEVICE = "auto"  # auto, cpu, 0, cuda:0, etc.
WORKERS = 0
SEED = 0
DIRECTION_NORM = "filter"  # filter, weight, unit.
INCLUDE_BN_BIAS = False
DPI = 640

DEFAULT_OUT = OUTPUT_ROOT
DEFAULT_YDY_ENV = {
    "YOLO26_USE_COMPOSITE_CLS": "1",
    "YOLO26_YDY_WEIGHT": "0.008",
    "YOLO26_YDY_FINAL_WEIGHT": "0.035",
    "YOLO26_YDY_SEP_WEIGHT": "0.030",
    "YOLO26_YDY_SEP_FINAL_WEIGHT": "0.100",
    "YOLO26_YDY_LIFT_WEIGHT": "0.018",
    "YOLO26_YDY_LIFT_FINAL_WEIGHT": "0.055",
    "YOLO26_YDY_MARGIN": "0.050",
    "YOLO26_YDY_MARGIN_FINAL": "0.120",
    "YOLO26_YDY_POS_FLOOR": "0.140",
    "YOLO26_YDY_POS_FLOOR_FINAL": "0.320",
    "YOLO26_YDY_BG_WEIGHT": "0.018",
    "YOLO26_YDY_BG_FINAL_WEIGHT": "0.045",
    "YOLO26_YDY_BG_THRESH": "0.200",
    "YOLO26_YDY_BG_MAX": "256",
}


@dataclass
class ResolvedInputs:
    weights: Path
    data: Path
    outdir: Path
    loss_profile: str


@dataclass
class LossPoint:
    x: float
    y: float
    total_loss: float
    box_loss: float | None
    cls_loss: float | None
    dfl_loss: float | None
    batches: int
    images: int
    seconds: float


def log(message: str) -> None:
    print(f"[loss-landscape] {message}", flush=True)


def ensure_repo_import() -> None:
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def looks_like_data_yaml(path: Path) -> bool:
    if path.suffix.lower() not in {".yaml", ".yml"} or not path.exists():
        return False
    data = load_yaml(path)
    return any(k in data for k in ("train", "val", "test")) and "names" in data


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def infer_data_from_weights(weights: Path) -> Path | None:
    candidates = []
    if weights.parent.name.lower() == "weights":
        candidates.append(weights.parent.parent / "args.yaml")
    candidates.append(weights.parent / "args.yaml")

    args_yaml = first_existing(candidates)
    if args_yaml:
        args_data = load_yaml(args_yaml).get("data")
        if args_data:
            data_path = Path(str(args_data))
            if not data_path.is_absolute():
                data_path = (args_yaml.parent / data_path).resolve()
            if data_path.exists():
                return data_path

    default_dataset = ROOT / "datasets" / "Aijiaojioashou" / "data.yaml"
    if default_dataset.exists():
        return default_dataset

    for candidate in sorted((ROOT / "datasets").glob("*/data.yaml")):
        if candidate.exists():
            return candidate
    return None


def infer_weights_from_data(data: Path) -> Path | None:
    candidates = [
        ROOT / "yolo26n.pt",
        ROOT / "best.pt",
        ROOT / "runs" / "detect" / "Aijiaojioashou_yolo26n_yaml_100e_original_strongaug_control" / "weights" / "best.pt",
    ]
    return first_existing(candidates)


def resolve_path(raw: str) -> Path:
    path = Path(raw.strip('"'))
    if path.exists():
        return path.resolve()
    candidate = (ROOT / path).resolve()
    return candidate


def parse_free_paths(paths: list[str]) -> tuple[Path | None, Path | None]:
    weights = None
    data = None
    for raw in paths:
        path = resolve_path(raw)
        suffix = path.suffix.lower()
        if path.is_dir():
            if (path / "data.yaml").exists():
                data = path / "data.yaml"
            elif (path / "weights" / "best.pt").exists():
                weights = path / "weights" / "best.pt"
            elif (path / "best.pt").exists():
                weights = path / "best.pt"
            continue
        if suffix in {".pt", ".pth"}:
            weights = path
        elif suffix in {".yaml", ".yml"}:
            if looks_like_data_yaml(path):
                data = path
            else:
                weights = path
        else:
            raise ValueError(f"Unsupported input path: {raw}")
    return weights, data


def infer_loss_profile(request: str, weights: Path) -> str:
    if request != "auto":
        return request
    text = str(weights).lower()
    return "ydy" if any(key in text for key in ("ydy", "ctcal", "cad")) else "original"


def resolve_inputs(args: argparse.Namespace) -> ResolvedInputs:
    weights, data = parse_free_paths(args.paths)
    if weights is None and data is None and args.weights is None and args.data is None:
        weights = resolve_path(str(WEIGHTS_PATH)) if str(WEIGHTS_PATH).strip() else None
        data = resolve_path(str(DATA_PATH)) if str(DATA_PATH).strip() else None

    if args.weights:
        weights = resolve_path(str(args.weights))
    if args.data:
        data = resolve_path(str(args.data))
        if data.is_dir():
            data = data / "data.yaml"

    if weights is None and data is not None:
        weights = infer_weights_from_data(data)
    if data is None and weights is not None:
        data = infer_data_from_weights(weights)

    if weights is None:
        raise FileNotFoundError("No weights were found. Pass a .pt path, or pass a data.yaml and keep yolo26n.pt in the repo root.")
    if data is None:
        raise FileNotFoundError("No data.yaml was found. Pass a data.yaml/dataset directory, or keep args.yaml next to the weight run.")
    if not weights.exists():
        raise FileNotFoundError(f"Weights do not exist: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"Data yaml does not exist: {data}")

    loss_profile = infer_loss_profile(args.loss_profile, weights)
    name = args.name or f"{weights.stem}_{loss_profile}_{int(time.time())}"
    outdir = (resolve_path(str(args.out)) / name).resolve()
    return ResolvedInputs(weights=weights.resolve(), data=data.resolve(), outdir=outdir, loss_profile=loss_profile)


def apply_ydy_loss_patch() -> None:
    for key, value in DEFAULT_YDY_ENV.items():
        os.environ.setdefault(key, value)
    import ultralytics.nn.tasks as tasks
    import ultralytics.utils.loss_YDY as loss_ydy

    tasks.E2ELoss = loss_ydy.E2ELoss
    tasks.v8DetectionLoss = loss_ydy.v8DetectionLoss
    tasks.v8SegmentationLoss = loss_ydy.v8SegmentationLoss
    tasks.v8OBBLoss = loss_ydy.v8OBBLoss
    tasks.PoseLoss26 = loss_ydy.PoseLoss26


def trainer_device_arg(device: str) -> str:
    return "" if device.lower() == "auto" else device


def build_loss_context(inputs: ResolvedInputs, args: argparse.Namespace) -> dict[str, Any]:
    ensure_repo_import()
    if inputs.loss_profile == "ydy":
        apply_ydy_loss_patch()

    try:
        import torch
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.utils.checks import check_imgsz
    except ModuleNotFoundError as exc:
        missing = exc.name or "dependency"
        raise RuntimeError(
            f"Missing dependency: {missing}\n"
            "Run this script with a PyTorch-enabled Ultralytics environment, for example:\n"
            r'  C:\Users\D\.conda\envs\yolov12\python.exe plot_loss_landscape.py path\to\best.pt'
        ) from exc

    overrides = {
        "model": str(inputs.weights),
        "data": str(inputs.data),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": trainer_device_arg(args.device),
        "workers": int(args.workers),
        "project": str(inputs.outdir / "_trainer_tmp"),
        "name": "probe",
        "exist_ok": True,
        "save": False,
        "plots": False,
        "val": False,
        "amp": False,
        "deterministic": True,
        "seed": int(args.seed),
        "rect": args.split != "train",
        "cache": False,
        "multi_scale": 0.0,
        "mosaic": 0.0 if args.split != "train" else 1.0,
        "close_mosaic": 0,
    }
    trainer = DetectionTrainer(overrides=overrides)
    trainer.setup_model()
    trainer.model.to(trainer.device)
    trainer.set_model_attributes()

    stride = max(int(trainer.model.stride.max() if hasattr(trainer.model, "stride") else 32), 32)
    trainer.args.imgsz = check_imgsz(trainer.args.imgsz, stride=stride, floor=stride, max_dim=1)
    trainer.stride = stride

    split_key = args.split
    if split_key not in trainer.data:
        if split_key == "test" and "val" in trainer.data:
            split_key = "val"
        else:
            raise KeyError(f"Dataset split '{args.split}' is not available in {inputs.data}")
    mode = "train" if split_key == "train" else "val"
    loader = trainer.get_dataloader(trainer.data[split_key], batch_size=int(args.batch), rank=-1, mode=mode)

    model = trainer.model
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    if hasattr(model.criterion, "set_progress"):
        model.criterion.set_progress(float(args.loss_progress))

    return {"torch": torch, "trainer": trainer, "loader": loader, "model": model, "split_key": split_key}


def ignored_parameter(name: str, parameter: Any, include_bn_bias: bool) -> bool:
    if include_bn_bias:
        return False
    lowered = name.lower()
    if parameter.ndim <= 1 or name.endswith(".bias"):
        return True
    return any(token in lowered for token in (".bn", "batchnorm", "norm", ".dfl."))


def normalize_direction(direction: Any, weight: Any, mode: str) -> Any:
    if mode == "unit":
        return direction / direction.norm().clamp_min(1e-12)
    if mode == "weight":
        return direction / direction.norm().clamp_min(1e-12) * weight.detach().norm().clamp_min(1e-12)
    if weight.ndim <= 1:
        return direction / direction.norm().clamp_min(1e-12) * weight.detach().norm().clamp_min(1e-12)

    rows = weight.shape[0]
    d_view = direction.view(rows, -1)
    w_view = weight.detach().view(rows, -1)
    d_norm = d_view.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w_norm = w_view.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (d_view / d_norm * w_norm).view_as(weight)


def build_directions(model: Any, torch: Any, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    base: dict[str, Any] = {}
    dir_x: dict[str, Any] = {}
    dir_y: dict[str, Any] = {}
    used = 0
    skipped = 0
    for name, parameter in model.named_parameters():
        if not parameter.dtype.is_floating_point:
            continue
        base[name] = parameter.detach().clone()
        if ignored_parameter(name, parameter, bool(args.include_bn_bias)):
            dir_x[name] = torch.zeros_like(parameter)
            dir_y[name] = torch.zeros_like(parameter)
            skipped += parameter.numel()
            continue
        dx = normalize_direction(torch.randn_like(parameter), parameter, args.direction_norm)
        dy = normalize_direction(torch.randn_like(parameter), parameter, args.direction_norm)
        dir_x[name] = dx
        dir_y[name] = dy
        used += parameter.numel()
    log(f"Perturbable parameters: {used:,}; frozen for landscape directions: {skipped:,}")
    return base, dir_x, dir_y


def apply_coordinates(model: Any, base: dict[str, Any], dir_x: dict[str, Any], dir_y: dict[str, Any], x: float, y: float) -> None:
    for name, parameter in model.named_parameters():
        if name not in base:
            continue
        parameter.data.copy_(base[name] + float(x) * dir_x[name] + float(y) * dir_y[name])


def restore_base(model: Any, base: dict[str, Any]) -> None:
    for name, parameter in model.named_parameters():
        if name in base:
            parameter.data.copy_(base[name])


def evaluate_loss(context: dict[str, Any], max_batches: int) -> tuple[float, list[float | None], int, int]:
    torch = context["torch"]
    trainer = context["trainer"]
    loader = context["loader"]
    model = context["model"]
    total = 0.0
    components: list[float] | None = None
    batches = 0
    images = 0

    with torch.no_grad():
        for batch in loader:
            if max_batches > 0 and batches >= max_batches:
                break
            batch = trainer.preprocess_batch(batch)
            preds = model(batch["img"])
            loss, items = model.loss(batch, preds)
            batch_size = int(batch["img"].shape[0])
            total += float(loss.detach().sum().cpu())
            item_values = items.detach().float().cpu().reshape(-1).tolist()
            if components is None:
                components = [0.0 for _ in item_values]
            for i, value in enumerate(item_values):
                components[i] += float(value) * batch_size
            batches += 1
            images += batch_size

    if images <= 0:
        raise RuntimeError("No images were evaluated. Check the dataset split and labels.")
    averaged = [(value / images) for value in components] if components else []
    while len(averaged) < 3:
        averaged.append(None)
    return total / images, averaged[:3], batches, images


def write_csv(points: list[LossPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(points[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def plot_landscape(points: list[LossPoint], inputs: ResolvedInputs, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.edgecolor": "#2F3437",
            "axes.labelcolor": "#202124",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": "#202124",
            "ytick.color": "#202124",
            "grid.color": "#D6DADF",
            "grid.linewidth": 0.7,
            "figure.facecolor": "#FAFAF7",
            "savefig.facecolor": "#FAFAF7",
        }
    )

    xs = sorted({p.x for p in points})
    ys = sorted({p.y for p in points})
    z = np.full((len(ys), len(xs)), np.nan, dtype=float)
    for p in points:
        z[ys.index(p.y), xs.index(p.x)] = p.total_loss
    x_grid, y_grid = np.meshgrid(xs, ys)

    min_index = np.nanargmin(z)
    min_y, min_x = np.unravel_index(min_index, z.shape)
    center_loss = next((p.total_loss for p in points if abs(p.x) < 1e-12 and abs(p.y) < 1e-12), None)

    fig, ax = plt.subplots(figsize=(8.4, 6.4), constrained_layout=True)
    contour = ax.contourf(x_grid, y_grid, z, levels=18, cmap="viridis")
    lines = ax.contour(x_grid, y_grid, z, levels=10, colors="#1C1F23", linewidths=0.45, alpha=0.55)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.3f")
    ax.scatter([0], [0], s=55, marker="x", color="#D55E00", linewidths=2.0, label="trained weights")
    ax.scatter([xs[min_x]], [ys[min_y]], s=46, color="#FFFFFF", edgecolor="#1C1F23", linewidth=1.2, label="lowest sampled loss")
    ax.set_xlabel("Direction u")
    ax.set_ylabel("Direction v")
    ax.set_title("Parameter-space loss landscape")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    cbar = fig.colorbar(contour, ax=ax, shrink=0.92)
    cbar.set_label("Detection loss per image")
    fig.savefig(inputs.outdir / "loss_landscape_contour.png", dpi=int(args.dpi))
    plt.close(fig)

    fig = plt.figure(figsize=(9.0, 6.6), constrained_layout=True)
    ax3d = fig.add_subplot(111, projection="3d")
    surface = ax3d.plot_surface(x_grid, y_grid, z, cmap=cm.viridis, linewidth=0.1, antialiased=True, alpha=0.95)
    ax3d.scatter([0], [0], [center_loss if center_loss is not None else z[min_y, min_x]], color="#D55E00", s=42)
    ax3d.set_xlabel("Direction u", labelpad=8)
    ax3d.set_ylabel("Direction v", labelpad=8)
    ax3d.set_zlabel("Loss", labelpad=8)
    ax3d.set_title("3D loss surface")
    ax3d.view_init(elev=28, azim=-132)
    fig.colorbar(surface, ax=ax3d, shrink=0.62, pad=0.08, label="Loss")
    fig.savefig(inputs.outdir / "loss_landscape_surface.png", dpi=int(args.dpi))
    plt.close(fig)

    row_y0 = min(range(len(ys)), key=lambda i: abs(ys[i]))
    col_x0 = min(range(len(xs)), key=lambda i: abs(xs[i]))
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.plot(xs, z[row_y0, :], color="#0072B2", linewidth=2.4, label="v = 0")
    ax.plot(ys, z[:, col_x0], color="#D55E00", linewidth=2.4, label="u = 0")
    ax.axvline(0, color="#202124", linewidth=0.8, alpha=0.55)
    ax.set_xlabel("Coordinate")
    ax.set_ylabel("Detection loss per image")
    ax.set_title("Central cross-sections")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(inputs.outdir / "loss_landscape_profile.png", dpi=int(args.dpi))
    plt.close(fig)

    fig = plt.figure(figsize=(12.0, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_a = fig.add_subplot(gs[:, :2])
    ax_b = fig.add_subplot(gs[0, 2], projection="3d")
    ax_c = fig.add_subplot(gs[1, 2])
    contour = ax_a.contourf(x_grid, y_grid, z, levels=18, cmap="viridis")
    ax_a.contour(x_grid, y_grid, z, levels=10, colors="#1C1F23", linewidths=0.35, alpha=0.5)
    ax_a.scatter([0], [0], s=52, marker="x", color="#D55E00", linewidths=2.0)
    ax_a.scatter([xs[min_x]], [ys[min_y]], s=42, color="#FFFFFF", edgecolor="#1C1F23", linewidth=1.1)
    ax_a.set_title("a  Contour map")
    ax_a.set_xlabel("Direction u")
    ax_a.set_ylabel("Direction v")
    fig.colorbar(contour, ax=ax_a, shrink=0.84, label="Loss")
    ax_b.plot_surface(x_grid, y_grid, z, cmap=cm.viridis, linewidth=0.1, antialiased=True)
    ax_b.set_title("b  Surface")
    ax_b.set_xlabel("u")
    ax_b.set_ylabel("v")
    ax_b.set_zlabel("Loss")
    ax_b.view_init(elev=28, azim=-132)
    ax_c.plot(xs, z[row_y0, :], color="#0072B2", linewidth=2.1, label="v = 0")
    ax_c.plot(ys, z[:, col_x0], color="#D55E00", linewidth=2.1, label="u = 0")
    ax_c.set_title("c  Cross-sections")
    ax_c.set_xlabel("Coordinate")
    ax_c.set_ylabel("Loss")
    ax_c.grid(True, alpha=0.35)
    ax_c.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{inputs.weights.name} | {inputs.loss_profile} loss | {args.steps} x {args.steps} samples", y=1.02)
    fig.savefig(inputs.outdir / "loss_landscape_report.png", dpi=int(args.dpi))
    plt.close(fig)


def save_metadata(inputs: ResolvedInputs, args: argparse.Namespace, points: list[LossPoint], seconds: float) -> None:
    best = min(points, key=lambda p: p.total_loss)
    center = next((p for p in points if abs(p.x) < 1e-12 and abs(p.y) < 1e-12), None)
    metadata = {
        "weights": str(inputs.weights),
        "data": str(inputs.data),
        "loss_profile": inputs.loss_profile,
        "outdir": str(inputs.outdir),
        "steps": args.steps,
        "radius": args.radius,
        "max_batches": args.max_batches,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "split": args.split,
        "device": args.device,
        "direction_norm": args.direction_norm,
        "include_bn_bias": bool(args.include_bn_bias),
        "best_sample": asdict(best),
        "center_sample": asdict(center) if center else None,
        "elapsed_seconds": seconds,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
    }
    with (inputs.outdir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    inputs = resolve_inputs(args)
    inputs.outdir.mkdir(parents=True, exist_ok=True)
    log(f"Weights: {inputs.weights}")
    log(f"Data: {inputs.data}")
    log(f"Loss profile: {inputs.loss_profile}")
    log(f"Output: {inputs.outdir}")

    if args.dry_run:
        return

    context = build_loss_context(inputs, args)
    base, dir_x, dir_y = build_directions(context["model"], context["torch"], args)
    import numpy as np

    coords = np.linspace(-float(args.radius), float(args.radius), int(args.steps))
    if not any(abs(float(v)) < 1e-12 for v in coords):
        coords[len(coords) // 2] = 0.0

    points: list[LossPoint] = []
    try:
        total_points = len(coords) * len(coords)
        index = 0
        for y in coords:
            x_iter = coords if index % 2 == 0 else coords[::-1]
            for x in x_iter:
                point_start = time.time()
                apply_coordinates(context["model"], base, dir_x, dir_y, float(x), float(y))
                total, components, batches, images = evaluate_loss(context, int(args.max_batches))
                point = LossPoint(
                    x=round(float(x), 8),
                    y=round(float(y), 8),
                    total_loss=total,
                    box_loss=components[0],
                    cls_loss=components[1],
                    dfl_loss=components[2],
                    batches=batches,
                    images=images,
                    seconds=time.time() - point_start,
                )
                points.append(point)
                index += 1
                log(f"{index:>3}/{total_points} u={point.x:+.3f} v={point.y:+.3f} loss={point.total_loss:.6f}")
    finally:
        restore_base(context["model"], base)

    write_csv(points, inputs.outdir / "loss_landscape.csv")
    plot_landscape(points, inputs, args)
    save_metadata(inputs, args, points, time.time() - started)
    log(f"Done. Report image: {inputs.outdir / 'loss_landscape_report.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a true 2D parameter-space loss landscape for local Ultralytics YOLO26 weights. "
            "Edit WEIGHTS_PATH and DATA_PATH at the top of this file, then run it directly. "
            "Command-line paths are optional and override the code configuration."
        )
    )
    parser.add_argument("paths", nargs="*", help="Free paths: .pt weights, data.yaml, or dataset/run directory.")
    parser.add_argument("--weights", "-w", type=Path, help="Weight path. Overrides positional detection.")
    parser.add_argument("--data", "-d", type=Path, help="data.yaml or dataset directory. Overrides positional detection.")
    parser.add_argument("--out", "-o", type=Path, default=DEFAULT_OUT, help="Output root directory.")
    parser.add_argument("--name", default=RUN_NAME, help="Output subdirectory name.")
    parser.add_argument("--loss-profile", choices=("auto", "original", "ydy"), default=LOSS_PROFILE)
    parser.add_argument("--loss-progress", type=float, default=1.0, help="YDY schedule progress used during loss evaluation.")
    parser.add_argument("--steps", type=int, default=STEPS, help="Grid samples per direction. Use odd values so 0 is sampled.")
    parser.add_argument("--radius", type=float, default=RADIUS, help="Perturbation radius along normalized directions.")
    parser.add_argument("--max-batches", type=int, default=MAX_BATCHES, help="Batches evaluated per grid point. 0 means full split.")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--split", choices=("train", "val", "test"), default=SPLIT)
    parser.add_argument("--device", default=DEVICE, help="auto, cpu, 0, cuda:0, etc.")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--direction-norm", choices=("filter", "weight", "unit"), default=DIRECTION_NORM)
    parser.add_argument(
        "--include-bn-bias",
        action="store_true",
        default=INCLUDE_BN_BIAS,
        help="Also perturb bias, norm and DFL parameters.",
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--dry-run", action="store_true", help="Only resolve paths and print the selected profile.")
    args = parser.parse_args()
    if args.steps < 3:
        parser.error("--steps must be at least 3")
    if args.batch < 1:
        parser.error("--batch must be at least 1")
    if args.radius <= 0:
        parser.error("--radius must be positive")
    return args


def main() -> None:
    try:
        run(parse_args())
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
