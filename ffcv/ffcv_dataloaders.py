"""
Implements dataloaders that use ffcv.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch

from ffcv.fields.basics import IntDecoder
from ffcv.pipeline.operation import Operation
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.loader import Loader, OrderOption

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256


def create_train_loader(train_dataset, num_workers, batch_size,
                        distributed, in_memory, device, image_size=IMG_SIZE):
    train_path = Path(train_dataset)
    assert train_path.is_file()

    decoder = RandomResizedCropRGBImageDecoder((image_size, image_size))

    image_pipeline: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(device), non_blocking=True)
    ]

    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    loader = Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed)

    return loader


def create_test_loader(test_dataset, num_workers, batch_size,
                       distributed, in_memory, device, image_size=IMG_SIZE):
    test_path = Path(test_dataset)
    assert test_path.is_file()

    decoder = CenterCropRGBImageDecoder(
        (image_size, image_size), DEFAULT_CROP_RATIO)

    image_pipeline: List[Operation] = [
        decoder,
        ToTensor(),
        ToDevice(torch.device(device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(device), non_blocking=True)
    ]

    loader = Loader(test_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    os_cache=in_memory,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed)

    return loader
