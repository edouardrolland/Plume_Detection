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

# Load the YOLO model
float_model = YOLO("plume_binary.pt")


def load_dataset(data_dir, format="yolo", sets = ['train']):
    """
    Loads images and their labels from a dataset organized into subfolders (train, valid, test).

    :param data_dir: Path to the folder containing the dataset.
    :param format: Format of the labels, either "yolo" or "coco".
    :return: A dictionary containing data for each set.
    """
    dataset = {}
     # You can include 'valid' and 'test' if they exist in your dataset.

    for data_set in sets:
        set_dir = Path(data_dir) / data_set
        images_dir = set_dir / "images"
        labels_dir = set_dir / "labels"

        images = []
        labels = []

        if not images_dir.exists() or not labels_dir.exists():
            print(f"Warning: Missing directory for {data_set}. Skipping.")
            continue

        for image_file in images_dir.glob("*.jpg"):
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Error loading image: {image_file}")
                continue

            # Load labels
            if format == "yolo":
                label_file = labels_dir / f"{image_file.stem}.txt"
                if label_file.exists():
                    with open(label_file, "r") as f:
                        labels.append(f.read())
            elif format == "coco":
                label_file = labels_dir / "annotations.json"
                if label_file.exists():
                    with open(label_file, "r") as f:
                        labels.append(json.load(f))
            else:
                raise ValueError("Unsupported label format: Use 'yolo' or 'coco'.")

            images.append(image)

        dataset[data_set] = {"images": images, "labels": labels}
        print(f"Loaded {len(images)} images and {len(labels)} labels for {data_set}.")

    return dataset


def draw_bounding_boxes(image, label, img_width, img_height):
    """
    Draws bounding boxes on the image based on YOLO format labels.

    :param image: The image to draw bounding boxes on.
    :param label: The label in YOLO format.
    :param img_width: Width of the image.
    :param img_height: Height of the image.
    :return: Image with bounding boxes drawn.
    """
    for line in label.strip().split("\n"):
        parts = line.split()
        if len(parts) != 5:
            continue

        # Parse YOLO format (class_id, x_center, y_center, width, height)
        class_id, x_center, y_center, width, height = map(float, parts)

        # Convert normalized coordinates to absolute coordinates
        x_center_abs = int(x_center * img_width)
        y_center_abs = int(y_center * img_height)
        width_abs = int(width * img_width)
        height_abs = int(height * img_height)

        # Calculate top-left and bottom-right corners
        x1 = int(x_center_abs - width_abs / 2)
        y1 = int(y_center_abs - height_abs / 2)
        x2 = int(x_center_abs + width_abs / 2)
        y2 = int(y_center_abs + height_abs / 2)

        # Draw the rectangle
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        image = cv2.putText(image, f"Class {int(class_id)}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image


def show_example_with_boxes(dataset):
    """
    Displays an example image from each set with bounding boxes.

    :param dataset: Dictionary containing 'train', 'valid', 'test' datasets with images and labels.
    """
    for set_name, data in dataset.items():
        if not data['images']:
            print(f"No images in {set_name} set.")
            continue

        # Take the first image and label as an example
        image = data['images'][0].copy()
        label = data['labels'][0] if data['labels'] else None

        if label is None:
            print(f"No labels for the first image in {set_name} set.")
            continue

        # Get image dimensions
        img_height, img_width, _ = image.shape

        # Draw bounding boxes
        image_with_boxes = draw_bounding_boxes(image, label, img_width, img_height)

        # Convert image from BGR (OpenCV default) to RGB for Matplotlib
        image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 8))
        plt.imshow(image_rgb)
        plt.title(f"{set_name.capitalize()} Set Example with Bounding Boxes")
        plt.axis("off")
        plt.show()


class YoloDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset wrapper for YOLO images and labels.
    """
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert image to Tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
        return image_tensor, label


# Load dataset
data_dir = "/home/edr/Documents/Divers/plume_detection/data_plume_original"
dataset = load_dataset(data_dir, format="yolo")

# Wrap the training data in PyTorch Dataset
train_images = dataset["train"]["images"]
train_labels = dataset["train"]["labels"]
train_dataset = YoloDataset(train_images, train_labels)

# Create a DataLoader
batch_size = 16
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Representative dataset generator
def representative_dataset_gen():
    for images, _ in dataloader:
        yield [images]


# Post-training quantization
target_platform_cap = mct.get_target_platform_capabilities('pytorch', 'default')

quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
    in_module=float_model,
    representative_data_gen=representative_dataset_gen,
    target_platform_capabilities=target_platform_cap
)


dataset_valid = load_dataset(data_dir, format="yolo", sets=['valid'])

val_dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=16, pin_memory=True)

from tqdm import tqdm


def evaluate(model, testloader):
    """
    Evaluate a model using a test loader.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # correct += (predicted == labels).sum().item()
    val_acc = (100 * correct / total)
    print('Accuracy: %.2f%%' % val_acc)
    return val_acc

evaluate(float_model, val_dataloader)

evaluate(quantized_model, val_dataloader)

mct.exporter.pytorch_export_model(quantized_model, save_model_path='qmodel.onnx', repr_dataset=representative_dataset_gen)