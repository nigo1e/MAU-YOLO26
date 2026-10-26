# MAU-YOLO26

## Installation

Create and activate a Python environment, then install PyTorch and its dependencies for your system:

```bash
conda create -n mau-yolo26 python=3.10
conda activate mau-yolo26
pip install torch torchvision
```

Run commands from the repository root so Python imports this project's local `ultralytics` package.

## Training

Use the MAU-YOLO26 model configuration with a dataset YAML file:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/MAU-YOLO26.yaml")
model.train(
    data="path/to/your/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)
```

Available model configurations are in `ultralytics/cfg/models/26/`, including MAU-YOLO26 and its ablation variants.

## Dataset

1. [UAV Construction](https://drive.google.com/file/d/1VFUE-pzlfZlJDC0YrRiaX9zulYHYYu5X/view?usp=sharing)  

## Acknowledgement

This project is based on [Ultralytics](https://github.com/ultralytics/ultralytics). Thanks to the Ultralytics team and contributors for their work.

## License

The Ultralytics-derived project source is distributed under the GNU Affero General Public License v3.0; see [LICENSE](LICENSE). Third-party components retain their respective terms, listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Model checkpoints and the linked dataset may have separate ownership and redistribution terms. Confirm that you have permission to redistribute each asset before publishing or rehosting it.
