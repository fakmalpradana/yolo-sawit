from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x-seg.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("path/img.jpg")  # predict on an image