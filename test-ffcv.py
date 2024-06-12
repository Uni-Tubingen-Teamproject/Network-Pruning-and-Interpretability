import time
import torch
import ffcv
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms import ToTensor, ToDevice, NormalizeImage
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from torchvision import transforms

# Paths to datasets
ffcv_dataset_path = '/mnt/lustre/datasets/ImageNet-ffcv'
local_imagenet_path = '/scratch_local/datasets/ImageNet2012'

# Define FFCV data pipeline
image_pipeline = [
    SimpleRGBImageDecoder(),
    ToTensor(),
    ToDevice(torch.device('cuda:0'), non_blocking=True),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
label_pipeline = [IntDecoder(), ToTensor(), ToDevice(torch.device('cuda:0'), non_blocking=True)]

# FFCV DataLoader
ffcv_loader = Loader(ffcv_dataset_path, batch_size=120, num_workers=8, order=OrderOption.SEQUENTIAL,
                     pipelines={'image': image_pipeline, 'label': label_pipeline})

# Local ImageNet DataLoader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
local_dataset = ImageNet(root=local_imagenet_path, split='val', transform=transform)
local_loader = DataLoader(local_dataset, batch_size=120, shuffle=False, num_workers=8)

# Timing FFCV DataLoader
start_time = time.time()
for images, labels in ffcv_loader:
    pass  # Simulate processing
ffcv_time = time.time() - start_time

# Timing Local ImageNet DataLoader
start_time = time.time()
for images, labels in local_loader:
    pass  # Simulate processing
local_time = time.time() - start_time

print(f"FFCV DataLoader time: {ffcv_time:.2f} seconds")
print(f"Local ImageNet DataLoader time: {local_time:.2f} seconds")

