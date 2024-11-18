import os
import cv2
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
from ultralytics import YOLO
import model_compression_toolkit as mct
from matplotlib import pyplot as plt
from tqdm import tqdm

# Load the YOLO model
yolo_model = YOLO("/home/edr/Documents/Divers/plume_detection/yolov8n.pt")

# Train the YOLO model
yolo_model.train(
    data=f"/home/edr/Documents/Divers/plume_detection/data.yaml",
    epochs=300,
    imgsz=640,
    plots=True
)

# Validate the YOLO model
yolo_model.val(
    data=f"/home/edr/Documents/Divers/plume_detection/data.yaml",
    weights=f"/home/edr/Documents/Divers/plume_detection/runs/detect/train/weights/best.pt"
)

