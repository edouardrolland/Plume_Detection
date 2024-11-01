import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model from the .pt file
model_path = '/home/edr/Downloads/4-YOLO-V8-Plume-Summit-Detections.pt'
model = YOLO(model_path)

# Open the input video
video_path = 'plume.mp4'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

# Get video parameters to create the output video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create the output video
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Open a text file to store inference speeds
with open('inference_speeds.txt', 'w') as speed_file:
    # Loop through each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Start time for inference
        start_time = time.time()

        # Perform detection on the frame
        results = model(frame)  # YOLOv8 returns results directly for each frame

        # End time for inference
        end_time = time.time()
        inference_time = end_time - start_time  # Time taken for inference

        # Write the inference time to the text file
        speed_file.write(f"Inference time: {inference_time:.6f} seconds\n")

        # Print the inference time on the console
        print(f"Inference time: {inference_time:.6f} seconds")

        # Iterate through the results for each detected object
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # xyxy: bounding box coordinates
            conf = result.conf[0]  # Detection confidence
            cls = int(result.cls[0])  # Detected class
            label = f'{model.names[cls]} {conf:.2f}'

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the annotated frame to the output video
        out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
