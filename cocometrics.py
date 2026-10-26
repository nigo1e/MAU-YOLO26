


import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import yaml
import tempfile
import shutil
import csv
import os
WEIGHTS = "yolo26n.pt"
DATA_YAML = "ultralytics\cfg\datasets\coco128.yaml"


CONF_THRES = 0.001
IOU_THRES = 0.6
MAX_DET = 300
DEVICE = ""
KEEP_TEMP_JSON = False
OUTPUT_DIR = "./results"

try:
    from ultralytics import YOLO
except ImportError:
    print("请安装 ultralytics: pip install ultralytics")
    exit(1)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("请安装 pycocotools: pip install pycocotools")
    exit(1)


def get_image_size(img_path):
    with Image.open(img_path) as img:
        return img.size


def parse_yaml(yaml_path):
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML文件不存在: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    path = data.get('path', '')
    if path:
        path = Path(path)
    else:
        path = yaml_path.parent
    val = data.get('val', '')
    if not val:
        raise ValueError("YAML文件中必须包含 'val' 字段")
    img_dir = path / val if path else Path(val)
    if not img_dir.exists():
        img_dir = Path(val)
        if not img_dir.exists():
            raise FileNotFoundError(f"验证图像目录不存在: {img_dir}")
    if 'images' in str(img_dir):
        label_dir = Path(str(img_dir).replace('images', 'labels'))
    else:
        label_dir = path / 'labels' / img_dir.name
    if not label_dir.exists():
        label_dir = path / 'labels' / val
    if not label_dir.exists():
        raise FileNotFoundError(f"无法定位标签目录，请确保标签在 {label_dir} 或手动调整代码")
    names = data.get('names', [])
    if not names:
        raise ValueError("YAML文件中必须包含 'names' 字段")
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]

    return img_dir, label_dir, names


def yolo_to_coco(img_dir, label_dir, names, output_json):
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(img_dir.glob(ext))
    if not image_files:
        image_files = list(img_dir.rglob('*'))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    if not image_files:
        raise FileNotFoundError(f"在 {img_dir} 中未找到任何图像")
    images = []
    annotations = []
    ann_id = 1
    category_to_id = {name: idx+1 for idx, name in enumerate(names)}

    for img_id, img_path in enumerate(image_files, start=1):
        width, height = get_image_size(img_path)
        try:
            file_name = str(img_path.relative_to(img_dir.parent))
        except ValueError:
            file_name = img_path.name

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })


        label_path = label_dir / img_path.stem
        if label_path.with_suffix('.txt').exists():
            label_file = label_path.with_suffix('.txt')
        else:

            continue

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                if class_id >= len(names):
                    print(f"警告: 标签中的类别ID {class_id} 超出 names 列表长度，跳过")
                    continue

                x_center, y_center, w, h = map(float, parts[1:5])
                x = (x_center - w/2) * width
                y = (y_center - h/2) * height
                bbox_width = w * width
                bbox_height = h * height
                bbox = [x, y, bbox_width, bbox_height]
                area = bbox_width * bbox_height

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id + 1,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1

    categories = [{"id": i+1, "name": name, "supercategory": "object"} for i, name in enumerate(names)]

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"转换完成: {len(images)} 张图像, {len(annotations)} 个标注 -> {output_json}")
    return output_json


def convert_to_coco_detection(predictions, image_id_map, conf_thres=0.001):
    coco_results = []
    for result in predictions:
        img_path = Path(result.path)
        file_name = img_path.name
        if file_name not in image_id_map:
            try:
                image_id = int(img_path.stem)
            except:
                raise KeyError(f"无法从文件名 {file_name} 映射到 image_id")
        else:
            image_id = image_id_map[file_name]

        boxes = result.boxes
        if boxes is None:
            continue
        for box, conf, cls in zip(boxes.xyxy.cpu().numpy(),
                                   boxes.conf.cpu().numpy(),
                                   boxes.cls.cpu().numpy()):
            if conf < conf_thres:
                continue
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            bbox = [float(x1), float(y1), float(width), float(height)]
            category_id = int(cls) + 1
            coco_results.append({
                "image_id": int(image_id),
                "category_id": category_id,
                "bbox": bbox,
                "score": float(conf)
            })
    return coco_results


def evaluate_coco(gt_json_path, dt_json_path, output_dir=None, iou_type="bbox"):
    cocoGt = COCO(gt_json_path)
    cocoDt = cocoGt.loadRes(dt_json_path)
    cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    stats_names = [
        'AP (IoU=0.50:0.95, area=all)',
        'AP (IoU=0.50, area=all)',
        'AP (IoU=0.75, area=all)',
        'AP (IoU=0.50:0.95, area=small)',
        'AP (IoU=0.50:0.95, area=medium)',
        'AP (IoU=0.50:0.95, area=large)',
        'AR (IoU=0.50:0.95, area=all, maxDets=1)',
        'AR (IoU=0.50:0.95, area=all, maxDets=10)',
        'AR (IoU=0.50:0.95, area=all, maxDets=100)',
        'AR (IoU=0.50:0.95, area=small, maxDets=100)',
        'AR (IoU=0.50:0.95, area=medium, maxDets=100)',
        'AR (IoU=0.50:0.95, area=large, maxDets=100)'
    ]
    stats_values = cocoEval.stats

    print("\n详细指标：")
    for name, value in zip(stats_names, stats_values):
        print(f"{name:60} = {value:.3f}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        overall_csv = output_dir / "overall_metrics.csv"
        with open(overall_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for name, value in zip(stats_names, stats_values):
                writer.writerow([name, f"{value:.3f}"])
        print(f"整体指标已保存至: {overall_csv}")



    cat_ids = cocoGt.getCatIds()
    print("\n每个类别的 AP:")
    precision = cocoEval.eval['precision']
    per_class_data = []
    for idx, cat_id in enumerate(cat_ids):
        cat_name = cocoGt.loadCats(cat_id)[0]['name']
        ap = np.mean(precision[:, :, idx, 0, -1])
        print(f"  {cat_id:3d} ({cat_name:20}): AP = {ap:.3f}")
        per_class_data.append([cat_id, cat_name, ap])

    if output_dir:
        per_class_csv = output_dir / "per_class_ap.csv"
        with open(per_class_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['category_id', 'category_name', 'AP'])
            for row in per_class_data:
                writer.writerow([row[0], row[1], f"{row[2]:.3f}"])
        print(f"各类别 AP 已保存至: {per_class_csv}")


def main():

    print(f"解析 YAML 文件: {DATA_YAML}")
    img_dir, label_dir, class_names = parse_yaml(DATA_YAML)
    print(f"图像目录: {img_dir}")
    print(f"标签目录: {label_dir}")
    print(f"类别数: {len(class_names)}")

    if KEEP_TEMP_JSON:
        temp_dir = Path.cwd()
        coco_gt_path = temp_dir / "coco_gt_temp.json"
    else:
        temp_dir = Path(tempfile.mkdtemp())
        coco_gt_path = temp_dir / "coco_gt.json"

    print("正在将 YOLO 标签转换为 COCO 格式...")
    yolo_to_coco(img_dir, label_dir, class_names, coco_gt_path)

    with open(coco_gt_path, 'r') as f:
        coco_data = json.load(f)
    image_id_map = {}
    for img_info in coco_data['images']:

        file_name = Path(img_info['file_name']).name
        image_id_map[file_name] = img_info['id']

    device = DEVICE if DEVICE else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"加载模型: {WEIGHTS} 到设备 {device}")
    model = YOLO(WEIGHTS)

    image_paths = list(img_dir.glob('*'))
    image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    if not image_paths:
        print("错误: 未找到任何图像文件。")
        return
    print(f"找到 {len(image_paths)} 张图像用于评估")

    print("开始预测...")
    all_results = []
    batch_size = 16
    for i in tqdm(range(0, len(image_paths), batch_size), desc="预测进度"):
        batch_paths = [str(p) for p in image_paths[i:i+batch_size]]
        results = model(
            batch_paths,
            conf=CONF_THRES,
            iou=IOU_THRES,
            max_det=MAX_DET,
            device=device,
            verbose=False
        )
        batch_coco = convert_to_coco_detection(results, image_id_map, conf_thres=CONF_THRES)
        all_results.extend(batch_coco)

    print(f"共生成 {len(all_results)} 个检测结果")

    dt_json_path = Path("coco_dt.json")
    with open(dt_json_path, 'w') as f:
        json.dump(all_results, f)
    print(f"预测结果已保存至: {dt_json_path}")



    evaluate_coco(str(coco_gt_path), str(dt_json_path), output_dir=OUTPUT_DIR)

    if not KEEP_TEMP_JSON:
        shutil.rmtree(temp_dir)
        print("临时 COCO 标注文件已删除")
    else:
        print(f"临时 COCO 标注文件保留在: {coco_gt_path}")


if __name__ == "__main__":
    main()
