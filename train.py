from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x-seg.yaml")  # build a new model from YAML
model = YOLO("yolo11x-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x-seg.yaml").load("yolo11x.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(
    data="coco8-seg.yaml", 
    epochs=10000, 
    imgsz=320,
)