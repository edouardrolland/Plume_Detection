import os
import pandas as pd
import model_compression_toolkit as mct
from ultralytics import YOLO
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

# Load your model in inference mode only
float_model = YOLO('/home/edr/Documents/Divers/plume_detection/plume.pt')
float_model.model.eval()  # Ensure the model stays in eval mode

# Define the paths for each split in the Hugging Face dataset
splits = {
    'train': 'hf://datasets/edouard-rolland/volcanic-plumes/data/train-00000-of-00001.parquet',
    'validation': 'hf://datasets/edouard-rolland/volcanic-plumes/data/validation-00000-of-00001.parquet',
    'test': 'hf://datasets/edouard-rolland/volcanic-plumes/data/test-00000-of-00001.parquet'
}

# Load each split into a DataFrame
train_df = pd.read_parquet(splits['train'])
validation_df = pd.read_parquet(splits['validation'])
test_df = pd.read_parquet(splits['test'])

# Define the Dataset class
class VolcanicPlumeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image, label = self.dataframe.iloc[idx]  # Use .iloc to get the row by position
        if self.transform:
            image = self.transform(image)  # Apply transformation here
        return image, label


# Define the transformation for the dataset (adjust as needed for your model)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize PyTorch Dataset
train_dataset = VolcanicPlumeDataset(train_df, transform=transform)

# Custom collate function to handle variable label sizes
def custom_collate_fn(batch):
    images, labels = zip(*batch)
    # Stack images as they are all resized to the same shape
    images = torch.stack(images, dim=0)
    
    # Pad labels to make them uniform in length within the batch
    labels = [torch.tensor(label['id']) for label in labels]  # Extract 'id' field as an example
    labels = pad_sequence(labels, batch_first=True, padding_value=0)  # Pad with zeros
    
    return images, labels

# Initialize DataLoader with the custom collate function
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

# Define the representative dataset generator
batch_size = 16
n_iter = 10

def representative_dataset_gen():
    dataloader_iter = iter(train_loader)
    for _ in range(n_iter):
        try:
            yield [next(dataloader_iter)[0]]
        except StopIteration:
            break

# Get a TargetPlatformCapabilities object that models the hardware platform for the quantized model inference.
target_platform_cap = mct.get_target_platform_capabilities('pytorch', 'default')

# Run Post-Training Quantization in inference mode only
quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
        in_module=float_model.model,  # Pass the actual model inside YOLO
        representative_data_gen=representative_dataset_gen,
        target_platform_capabilities=target_platform_cap
)
