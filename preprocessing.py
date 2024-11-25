from ultralytics import YOLO

#model = YOLO('/home/edr/workspace/Plume_Detection/YOLO_Models/plume_yolo_v8n.pt')

# Export the model
#model.export(format="imx")  # exports with PTQ quantization by default



# Load the exported model
imx_model = YOLO("/home/edr/workspace/Plume_Detection/YOLO_Models/plume_yolo_v8n_imx_model")

# Validate the YOLO model
imx_model.val(
    data=f"/home/edr/workspace/Plume_Detection/notebooks/data.yaml",
)


