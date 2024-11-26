from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
#model = YOLO("/home/edr/workspace/Plume_Detection/YOLO_Models/plume.pt")

# Export the model
#model.export(format="imx", data = "/home/edr/workspace/Plume_Detection/data.yaml")  # exports with PTQ quantization by default

# Load the exported model
imx_model = YOLO("/home/edr/workspace/Plume_Detection/YOLO_Models/plume_imx_model")

