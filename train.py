from ultralytics import YOLO
from roboflow import Roboflow

import subprocess as sp

# Load dataset
rf = Roboflow(api_key="MAmVfKJcQVrC1zBogRHo")
project = rf.workspace("project-geoai").project("palm-area-detections")
version = project.version(1)
dataset = version.download("yolov11")

# Run YOLO training
sp.call(
    [
        "yolo", 
        "task=segment",
        "mode=train", 
        "model=yolo11s-seg.pt",
        "data={dataset.location}/data.yaml", 
        "epochs=10", 
        "imgsz=640"
    ]
)