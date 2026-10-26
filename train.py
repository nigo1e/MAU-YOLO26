
 
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26-OverLoCK-Backbone.yaml")  # build a new model from scratch
# model = YOLO("ultralytics/cfg/models/26/yolo26-EfficientViM-Backbone.yaml")
 
# Train the model
results = model.train(data="coco128.yaml",
    epochs=30,
    imgsz=640,
    optimizer='MuSGD',
 )

# m3fd-rgbir
