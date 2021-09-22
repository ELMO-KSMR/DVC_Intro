import numpy as np
import sys
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

# pathname = os.path.dirname(sys.argv[0])
# path = os.path.abspath(pathname)
# print(path)

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = "model.h5"

base_dir = Path(__file__).resolve().parent.parent

train_data_dir = os.path.join(base_dir, "data", "train")
validation_data_dir = os.path.join(base_dir, "data", "validation")
cats_train_path = os.path.join(base_dir, train_data_dir, "cats")
nb_train_samples = 2 * len(
    [
        name
        for name in os.listdir(cats_train_path)
        if os.path.isfile(os.path.join(cats_train_path, name))
    ]
)
nb_validation_samples = 800
epochs = 10
batch_size = 10


def save_bottleneck_features():
    transform_data = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_data = datasets.ImageFolder(train_data_dir, transform_data)
    validation_data = datasets.ImageFolder(validation_data_dir, transform_data)

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=False
    )
    validation_dataloader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False
    )

    model = models.vgg16(pretrained=True)
    print(train_data, validation_data)
    print(model)


save_bottleneck_features()
