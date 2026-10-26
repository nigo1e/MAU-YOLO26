


import os
import random
import numpy as np
import cv2
import yaml
from typing import List, Tuple, Optional

INPUT_ROOT = 'datasets\coco128'
OUTPUT_ROOT = './augmented_coco128'
AUGMENTATIONS = ['mixup', 'copypaste', 'random_erasing']
IMG_SIZE = 640
MIXUP_PROB = 0.5
COPYPASTE_PROB = 0.5
ERASE_PROB = 0.5
MAX_OBJECTS_COPY = 3
RANDOM_SEED = 42

def mixup(img1: np.ndarray, labels1: np.ndarray,
          img2: np.ndarray, labels2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lam = np.random.beta(1.5, 1.5)
    img = (lam * img1 + (1 - lam) * img2).astype(np.uint8)
    if labels1.size == 0 and labels2.size == 0:
        labels = np.zeros((0, 5), dtype=np.float32)
    elif labels1.size == 0:
        labels = labels2
    elif labels2.size == 0:
        labels = labels1
    else:
        labels = np.vstack((labels1, labels2))
    return img, labels
def copy_paste(img_dst: np.ndarray, labels_dst: np.ndarray,
               img_src: np.ndarray, labels_src: np.ndarray,
               img_size: int, max_objects: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    if labels_src.size == 0:
        return img_dst, labels_dst

    num_objects = min(len(labels_src), max_objects)
    selected_idx = np.random.choice(len(labels_src), num_objects, replace=False)

    img_out = img_dst.copy()
    labels_out = labels_dst.copy() if labels_dst.size > 0 else np.zeros((0, 5), dtype=np.float32)

    for idx in selected_idx:
        cls, x_norm, y_norm, w_norm, h_norm = labels_src[idx]
        x_center = int(x_norm * img_size)
        y_center = int(y_norm * img_size)
        box_w = int(w_norm * img_size)
        box_h = int(h_norm * img_size)
        x1 = max(0, x_center - box_w // 2)
        y1 = max(0, y_center - box_h // 2)
        x2 = min(img_size, x1 + box_w)
        y2 = min(img_size, y1 + box_h)
        if x2 <= x1 or y2 <= y1:
            continue
        obj_patch = img_src[y1:y2, x1:x2].copy()
        paste_w = x2 - x1
        paste_h = y2 - y1
        if paste_w <= 0 or paste_h <= 0:
            continue
        new_x1 = random.randint(0, img_size - paste_w)
        new_y1 = random.randint(0, img_size - paste_h)
        new_x2 = new_x1 + paste_w
        new_y2 = new_y1 + paste_h
        img_out[new_y1:new_y2, new_x1:new_x2] = obj_patch
        new_x_center = (new_x1 + new_x2) / 2 / img_size
        new_y_center = (new_y1 + new_y2) / 2 / img_size
        new_w = paste_w / img_size
        new_h = paste_h / img_size
        new_label = np.array([[cls, new_x_center, new_y_center, new_w, new_h]], dtype=np.float32)
        labels_out = np.vstack((labels_out, new_label)) if labels_out.size else new_label
    return img_out, labels_out
def random_erasing(img: np.ndarray, labels: np.ndarray,
                   img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    img_h, img_w = img.shape[:2]
    s_min = 0.02
    s_max = 0.4
    r_min = 0.3
    r_max = 3.0
    area = img_h * img_w
    target_area = random.uniform(s_min, s_max) * area
    aspect_ratio = random.uniform(r_min, r_max)
    erase_w = int(round(np.sqrt(target_area * aspect_ratio)))
    erase_h = int(round(np.sqrt(target_area / aspect_ratio)))
    if erase_w < img_w and erase_h < img_h:
        x = random.randint(0, img_w - erase_w)
        y = random.randint(0, img_h - erase_h)
        img[y:y+erase_h, x:x+erase_w] = np.random.randint(0, 256, (erase_h, erase_w, 3), dtype=np.uint8)

    return img, labels
def load_image_and_labels(img_path: str, label_path: str, img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    labels = []
    if os.path.isfile(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, w_norm, h_norm = map(float, parts)
                    labels.append([cls, x, y, w_norm, h_norm])
    return img, np.array(labels, dtype=np.float32)
def save_image_and_labels(img: np.ndarray, labels: np.ndarray,
                          img_save_path: str, label_save_path: str):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_save_path, img_bgr)
    with open(label_save_path, 'w', encoding='utf-8') as f:
        for label in labels:
            cls, x, y, w, h = label
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
def generate_augmented_dataset(
    input_root: str,
    output_root: str,
    augmentations: List[str],
    img_size: int = 640,
    mixup_prob: float = 0.5,
    copypaste_prob: float = 0.5,
    erase_prob: float = 0.5,
    max_objects_copy: int = 3,
    seed: Optional[int] = None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    img_dir_in = os.path.join(input_root, 'images')
    label_dir_in = os.path.join(input_root, 'labels')
    if not os.path.isdir(img_dir_in) or not os.path.isdir(label_dir_in):
        raise ValueError("输入数据集必须包含 images/ 和 labels/ 子目录")
    img_dir_out = os.path.join(output_root, 'images')
    label_dir_out = os.path.join(output_root, 'labels')
    os.makedirs(img_dir_out, exist_ok=True)
    os.makedirs(label_dir_out, exist_ok=True)
    img_files = [f for f in os.listdir(img_dir_in) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_files.sort()
    print(f"找到 {len(img_files)} 张图像")
    yaml_in = os.path.join(input_root, 'data.yaml')
    if os.path.isfile(yaml_in):
        with open(yaml_in, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        data['path'] = output_root
        with open(os.path.join(output_root, 'data.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        print("已复制 data.yaml")
    else:
        print("警告: 未找到 data.yaml，输出目录中不会包含该文件")

    for idx, fname in enumerate(img_files):
        print(f"处理 [{idx+1}/{len(img_files)}]: {fname}")

        img_path = os.path.join(img_dir_in, fname)
        label_path = os.path.join(label_dir_in, fname.replace('.jpg', '.txt')
                                  .replace('.jpeg', '.txt').replace('.png', '.txt'))
        img, labels = load_image_and_labels(img_path, label_path, img_size)
        for aug in augmentations:
            if aug == 'mixup' and random.random() < mixup_prob:
                other_idx = random.randint(0, len(img_files) - 1)
                while other_idx == idx and len(img_files) > 1:
                    other_idx = random.randint(0, len(img_files) - 1)
                other_fname = img_files[other_idx]
                other_img_path = os.path.join(img_dir_in, other_fname)
                other_label_path = os.path.join(label_dir_in, other_fname.replace('.jpg', '.txt')
                                                .replace('.jpeg', '.txt').replace('.png', '.txt'))
                img2, labels2 = load_image_and_labels(other_img_path, other_label_path, img_size)
                img, labels = mixup(img, labels, img2, labels2)

            elif aug == 'copypaste' and random.random() < copypaste_prob:
                other_idx = random.randint(0, len(img_files) - 1)
                while other_idx == idx and len(img_files) > 1:
                    other_idx = random.randint(0, len(img_files) - 1)
                other_fname = img_files[other_idx]
                other_img_path = os.path.join(img_dir_in, other_fname)
                other_label_path = os.path.join(label_dir_in, other_fname.replace('.jpg', '.txt')
                                                .replace('.jpeg', '.txt').replace('.png', '.txt'))
                img2, labels2 = load_image_and_labels(other_img_path, other_label_path, img_size)
                img, labels = copy_paste(img, labels, img2, labels2, img_size, max_objects_copy)

            elif aug == 'random_erasing' and random.random() < erase_prob:
                img, labels = random_erasing(img, labels, img_size)
        out_img_path = os.path.join(img_dir_out, fname)
        base, ext = os.path.splitext(fname)
        out_label_path = os.path.join(label_dir_out, base + '.txt')
        save_image_and_labels(img, labels, out_img_path, out_label_path)

    print("增强数据集生成完成！")
if __name__ == "__main__":
    generate_augmented_dataset(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        augmentations=AUGMENTATIONS,
        img_size=IMG_SIZE,
        mixup_prob=MIXUP_PROB,
        copypaste_prob=COPYPASTE_PROB,
        erase_prob=ERASE_PROB,
        max_objects_copy=MAX_OBJECTS_COPY,
        seed=RANDOM_SEED
    )
