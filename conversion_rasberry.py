import os
import cv2
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt
from tqdm import tqdm

# Load the YOLO model
yolo_model = YOLO("/home/edouard/workspace/Plume_Detection/yolov8n.pt")

# Train the YOLO model
yolo_model.train(
    data=f"/home/edouard/workspace/Plume_Detection/notebooks/data.yaml",
    epochs=300,
    imgsz=640,
    plots=True,
    device="0",
)

# Validate the YOLO model
yolo_model.val(
    data=f"/home/edouard/workspace/Plume_Detection/notebooks/data.yaml",
    weights=f"/home/edouard/workspace/Plume_Detection/runs/detect/train/weights/best.pt"
)

