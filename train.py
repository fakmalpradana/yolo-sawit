import os

# Disable OpenMP threading to avoid duplicate initialization
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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
        "model=yolo11x-seg.pt",
        f"data={dataset.location}/data.yaml", 
        "epochs=10", 
        "imgsz=320",
        "scale=1.0"
    ]
)