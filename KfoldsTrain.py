import os
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import yaml



class SimpleYOLOKFoldCV:
    def __init__(self, data_yaml_path, model_config=None, weights_path=None, k_folds=5, output_dir='kfold_results'):

        self.data_yaml_path = Path(data_yaml_path).resolve()

        if model_config:
            self.model_config = Path(model_config).resolve()
            self.weights_path = Path(weights_path) if weights_path else None
        elif weights_path:
            self.model_config = None
            self.weights_path = Path(weights_path).resolve()
        else:
            raise ValueError("必须提供 model_config 或 weights_path 中的一个")

        self.k_folds = k_folds
        self.output_dir = Path(output_dir).resolve()

        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"数据配置文件不存在: {data_yaml_path}")

        if self.model_config and not self.model_config.exists():
            raise FileNotFoundError(f"模型配置文件不存在: {model_config}")

        if self.weights_path and not self.weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

        with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ 初始化完成:")
        print(f"   - 数据配置: {self.data_yaml_path}")
        if self.model_config:
            print(f"   - 模型配置: {self.model_config}")
        if self.weights_path:
            print(f"   - 预训练权重: {self.weights_path}")
        print(f"   - 折数: {self.k_folds}")
        print(f"   - 输出目录: {self.output_dir}")

    def get_image_labels(self):

        train_path = self.data_config.get('train', '')
        if not train_path:
            raise ValueError("数据配置文件中未找到 'train' 路径")

        print(f"🔍 解析训练路径: {train_path}")

        image_paths = []

        if isinstance(train_path, list):
            for path in train_path:
                abs_path = self.data_yaml_path.parent / Path(path)
                if abs_path.exists():
                    if abs_path.is_file():
                        with open(abs_path, 'r', encoding='utf-8') as f:
                            image_paths.extend([line.strip() for line in f.readlines() if line.strip()])
                    elif abs_path.is_dir():
                        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                        for ext in image_extensions:
                            image_paths.extend([str(p) for p in abs_path.rglob(f"*{ext}")])
        else:
            abs_path = self.data_yaml_path.parent / Path(train_path)
            if abs_path.exists():
                if abs_path.is_file():
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        image_paths = [line.strip() for line in f.readlines() if line.strip()]
                elif abs_path.is_dir():
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                    for ext in image_extensions:
                        image_paths.extend([str(p) for p in abs_path.rglob(f"*{ext}")])

        image_paths = list(set(image_paths))
        print(f"📁 找到 {len(image_paths)} 张训练图像")

        if len(image_paths) == 0:
            print("❌ 没有找到任何图像文件")
            return [], []

        image_labels = []
        valid_image_paths = []

        for img_path in image_paths:
            label_path = self._get_label_path(img_path)
            if label_path and label_path.exists():
                main_class = self._get_main_class(label_path)
                image_labels.append(main_class)
                valid_image_paths.append(img_path)

        print(f"📊 有效图像: {len(valid_image_paths)} 张 (有对应标签)")
        return valid_image_paths, image_labels

    def _get_label_path(self, image_path):

        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = self.data_yaml_path.parent / img_path


        possible_paths = [
            img_path.parent.parent / 'labels' / img_path.parent.name / f"{img_path.stem}.txt",
            img_path.parent / 'labels' / f"{img_path.stem}.txt",
            img_path.with_suffix('.txt'),
            self.data_yaml_path.parent / 'labels' / img_path.parent.name / f"{img_path.stem}.txt",
        ]

        for label_path in possible_paths:
            if label_path.exists():
                return label_path
        return None

    def _get_main_class(self, label_path):

        try:
            with open(label_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            if not lines:
                return 0

            classes = []
            for line in lines:
                parts = line.split()
                if parts:
                    try:
                        class_id = int(parts[0])
                        classes.append(class_id)
                    except (ValueError, IndexError):
                        continue

            return max(set(classes), key=classes.count) if classes else 0
        except Exception as e:
            print(f"❌ 读取标签文件错误 {label_path}: {e}")
            return 0

    def create_fold_data(self, image_paths, fold_indices):

        fold_data_files = {}

        for fold in range(self.k_folds):
            print(f"🔄 创建第 {fold + 1} 折数据...")


            train_indices = []
            val_indices = []

            for i, (train_idx, val_idx) in enumerate(fold_indices):
                if i == fold:
                    val_indices.extend(val_idx)
                else:
                    train_indices.extend(train_idx)


            fold_dir = self.output_dir / f'fold_{fold}'
            fold_dir.mkdir(exist_ok=True)


            train_file = fold_dir / 'train.txt'
            with open(train_file, 'w', encoding='utf-8') as f:
                for idx in train_indices:

                    abs_path = self._ensure_absolute_path(image_paths[idx])
                    f.write(f"{abs_path}\n")


            val_file = fold_dir / 'val.txt'
            with open(val_file, 'w', encoding='utf-8') as f:
                for idx in val_indices:

                    abs_path = self._ensure_absolute_path(image_paths[idx])
                    f.write(f"{abs_path}\n")


            fold_data_config = {

                'path': str(self.data_yaml_path.parent.resolve()),

                'train': str(train_file.resolve()),
                'val': str(val_file.resolve()),

                'names': self.data_config['names']
            }


            for key, value in self.data_config.items():
                if key not in ['path', 'train', 'val', 'test', 'nc', 'names']:
                    fold_data_config[key] = value

            fold_data_yaml = fold_dir / 'data.yaml'
            with open(fold_data_yaml, 'w', encoding='utf-8') as f:
                yaml.dump(fold_data_config, f, default_flow_style=False, allow_unicode=True)

            fold_data_files[fold] = str(fold_data_yaml)

            print(f"   ✅ 训练集: {len(train_indices)} 张")
            print(f"   ✅ 验证集: {len(val_indices)} 张")
            print(f"   ✅ 数据配置: {fold_data_yaml}")


            self._validate_fold_data(fold_dir, train_file, val_file, fold_data_yaml)

        return fold_data_files

    def _ensure_absolute_path(self, image_path):

        img_path = Path(image_path)
        if not img_path.is_absolute():

            abs_path = self.data_yaml_path.parent / img_path
            if abs_path.exists():
                return str(abs_path.resolve())

            elif img_path.exists():
                return str(img_path.resolve())
            else:

                return str(img_path)
        return str(img_path.resolve())

    def _validate_fold_data(self, fold_dir, train_file, val_file, data_yaml):

        print(f"   🔍 验证生成的文件...")


        if train_file.exists():
            with open(train_file, 'r') as f:
                train_lines = f.readlines()
            print(f"     训练文件: {len(train_lines)} 行")


            if train_lines:
                sample_path = train_lines[0].strip()
                full_path = Path(sample_path)
                print(f"     样本图像: {sample_path}")
                print(f"     完整路径: {full_path}")
                print(f"     图像存在: {full_path.exists()}")


        if val_file.exists():
            with open(val_file, 'r') as f:
                val_lines = f.readlines()
            print(f"     验证文件: {len(val_lines)} 行")

            if val_lines:
                sample_path = val_lines[0].strip()
                full_path = Path(sample_path)
                print(f"     样本图像: {sample_path}")
                print(f"     完整路径: {full_path}")
                print(f"     图像存在: {full_path.exists()}")


        if data_yaml.exists():
            with open(data_yaml, 'r') as f:
                data_content = f.read()
            print(f"     数据配置内容:")
            for line in data_content.split('\n')[:8]:
                print(f"       {line}")

    def train_fold(self, fold, data_yaml, output_dir, epochs=100, imgsz=640, batch_size=16):

        print(f"🚀 开始训练第 {fold + 1} 折...")

        try:
            from ultralytics import YOLO


            if self.model_config:

                print(f"📋 使用模型配置: {self.model_config}")
                model = YOLO(str(self.model_config))
            else:

                print(f"⚖️ 使用预训练权重: {self.weights_path}")
                model = YOLO(str(self.weights_path))


            data_yaml_abs = Path(data_yaml).resolve()
            print(f"📁 使用数据配置文件: {data_yaml_abs}")

            if not data_yaml_abs.exists():
                raise FileNotFoundError(f"数据配置文件不存在: {data_yaml_abs}")


            with open(data_yaml_abs, 'r') as f:
                data_config = yaml.safe_load(f)

            print(f"🔍 数据配置验证:")
            print(f"   - path: {data_config.get('path')}")
            print(f"   - train: {data_config.get('train')}")
            print(f"   - val: {data_config.get('val')}")


            train_file = Path(data_config['train'])
            val_file = Path(data_config['val'])

            print(f"   - 训练文件存在: {train_file.exists()}")
            print(f"   - 验证文件存在: {val_file.exists()}")

            if not train_file.exists():
                raise FileNotFoundError(f"训练文件不存在: {train_file}")
            if not val_file.exists():
                raise FileNotFoundError(f"验证文件不存在: {val_file}")


            results = model.train(
                data=str(data_yaml_abs),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                project=str(self.output_dir),
                name=f'fold_{fold}',
                exist_ok=True,
                patience=20,
                lr0=0.01,
                save=True,
                verbose=False
            )

            return {
                'status': 'success',
                'metrics': results.results_dict if hasattr(results, 'results_dict') else {},
                'best_model': str(results.best) if hasattr(results, 'best') else None
            }

        except Exception as e:
            print(f"❌ 第 {fold + 1} 折训练失败: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}

    def run(self, epochs=100, imgsz=640, batch_size=16):

        print("🎯 开始 k 折交叉验证...")


        image_paths, image_labels = self.get_image_labels()

        if len(image_paths) == 0:
            print("❌ 没有找到有效的图像文件")
            return {}

        if len(image_paths) < self.k_folds:
            print(f"⚠️  图像数量 ({len(image_paths)}) 少于折数 ({self.k_folds})，自动调整折数为 {len(image_paths)}")
            self.k_folds = min(self.k_folds, len(image_paths))


        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        fold_indices = list(skf.split(image_paths, image_labels))


        fold_data_files = self.create_fold_data(image_paths, fold_indices)


        results = {}
        for fold in range(self.k_folds):
            print(f"\n{'=' * 60}")
            print(f"📊 第 {fold + 1}/{self.k_folds} 折")
            print(f"{'=' * 60}")

            data_yaml = fold_data_files[fold]
            result = self.train_fold(fold, data_yaml, self.output_dir, epochs, imgsz, batch_size)
            results[fold] = result

            if result['status'] == 'success':
                print(f"✅ 第 {fold + 1} 折训练完成")
            else:
                print(f"❌ 第 {fold + 1} 折训练失败")


        self.save_results(results)
        return results

    def save_results(self, results):

        results_file = self.output_dir / 'cross_validation_results.json'


        serializable_results = {}
        for fold, result in results.items():
            serializable_results[fold] = {
                'status': result['status'],
                'error': result.get('error'),
                'metrics': result.get('metrics', {}),
                'best_model': str(result.get('best_model'))
            }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)


        self.calculate_and_save_metrics(results)

        print(f"\n🎉 交叉验证完成!")
        print(f"📁 结果保存至: {self.output_dir}")

    def calculate_and_save_metrics(self, results):

        metrics_summary = {}

        successful_folds = [r for r in results.values() if r['status'] == 'success' and 'metrics' in r]

        if not successful_folds:
            print("⚠️  没有成功的训练折，无法计算指标")
            return


        metric_names = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'precision', 'recall']

        for metric_name in metric_names:
            values = []
            for fold_result in successful_folds:

                for key, value in fold_result['metrics'].items():
                    if metric_name in key:
                        values.append(float(value))
                        break

            if values:
                metrics_summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }

        metrics_file = self.output_dir / 'metrics_summary.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

        print(f"📊 指标摘要:")
        for metric, stats in metrics_summary.items():
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='YOLO K折交叉验证')
    parser.add_argument('--data', type=str, required=True, help='datasets/coco128/coco128.yaml')
    parser.add_argument('--model', type=str, help='yolov13')
    parser.add_argument('--weights', type=str, help='预训练权重路径 (如: yolov8s.pt)')
    parser.add_argument('--k-folds', type=int, default=5, help='折数')
    parser.add_argument('--epochs', type=int, default=4, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--output-dir', type=str, default='kfold_results', help='输出目录')

    args = parser.parse_args()

    if not args.model and not args.weights:
        print("❌ 错误: 必须提供 --model 或 --weights 参数")
        return 1

    print("🚀 YOLO K折交叉验证")
    print("=" * 50)

    try:
        cv = SimpleYOLOKFoldCV(
            data_yaml_path=args.data,
            model_config=args.model,
            weights_path=args.weights,
            k_folds=args.k_folds,
            output_dir=args.output_dir
        )

        results = cv.run(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch_size
        )

        print("\n✅ 所有任务完成!")

    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:

        print("🔧 示例用法:")
        print("1. 使用模型配置文件 (从头训练):")
        print("   python kfold_cv.py --data datasets/coco128/data.yaml --model models/yolov8n.yaml")
        print()
        print("2. 使用预训练权重 (迁移学习):")
        print("   python kfold_cv.py --data datasets/coco128/data.yaml --weights yolov8n.pt")


        DATA_YAML = "datasets/coco128/data.yaml"

        if Path(DATA_YAML).exists():
            print(f"\n🏃 尝试使用默认路径运行...")

            possible_models = [
                "models/yolov8n.yaml",
                "yolov8n.yaml",
                "yolov8n.pt",
                "yolov8s.pt"
            ]

            model_to_use = None
            weights_to_use = None

            for model_path in possible_models:
                if Path(model_path).exists():
                    if model_path.endswith('.yaml'):
                        model_to_use = model_path
                    else:
                        weights_to_use = model_path
                    break

            if model_to_use or weights_to_use:
                cv = SimpleYOLOKFoldCV(
                    data_yaml_path=DATA_YAML,
                    model_config=model_to_use,
                    weights_path=weights_to_use,
                    k_folds=2,
                    output_dir="my_kfold_results"
                )
                results = cv.run(epochs=3, imgsz=0, batch_size=0)
            else:
                print("❌ 未找到模型配置文件或预训练权重")
                print("请通过命令行参数指定正确的路径")
        else:
            print("❌ 未找到数据配置文件")
            print("请通过命令行参数指定正确的路径")
