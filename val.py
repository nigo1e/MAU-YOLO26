
 
from ultralytics import YOLO

model = YOLO('best.pt')

 
# Train the model
results = model.val(data="coco128.yaml",  imgsz=640)
