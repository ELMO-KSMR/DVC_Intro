"""This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io. But in this file it is trained usin pytorch

In our example we will be using data that can be downloaded at:
https://www.kaggle.com/tongpython/cat-and-dog

In our setup, it expects:
- a data/ folder
- train/ and validation/ subfolders inside data/
- cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-X in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 0-X in data/train/dogs
- put the dog pictures index 1000-1400 in data/validation/dogs

We have X training examples for each class, and 400 validation examples
for each class. In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
"""
from __future__ import print_function
import os
import sys
import numpy as np
from time import time
from PIL import Image as PILImage
import torch
import torchvision
import torch

# from torchsummary import summary
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models

from tqdm import tqdm
from torch.nn import CrossEntropyLoss

from os import listdir
from os.path import isfile, join
import pandas as pd


def get_number_of_files(path):
    file_list = [f for f in listdir(path) if isfile(join(path, f))]
    return len(file_list)


def get_file_index_list(path):
    return [
        f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    ]


classes = ["cats", "dogs"]


def get_model():

    model = models.vgg16(pretrained=True)
    # print("model: ", model)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[6].in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    for param in model.classifier[6].parameters():
        param.requires_grad = True

    return model


def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    train_losses,
    train_acc,
    cats_train_acc,
    dogs_train_acc,
    l1_param=0.0,
):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    correct_cats = 0
    correct_dogs = 0
    processed = 0
    processed_cats = 0
    processed_dogs = 0
    local_train_losses = []
    local_train_acc = []
    local_cats_train_acc = []
    local_dogs_train_acc = []
    for batch_idx, train_data in enumerate(pbar):
        # get samples
        data, target = train_data["X"].to(device), train_data["Y"].to(device)

        mask_cats = target == classes.index("cats")
        # print("mask_cats = ", mask_cats)
        target_cats = target[mask_cats]
        # print( " target_cats = ", target_cats)

        mask_dogs = target == classes.index("dogs")
        # print("mask_dogs = ", mask_dogs)
        target_dogs = target[mask_dogs]
        # print("target_dogs = ", target_dogs)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)
        # print( " y_pred = ", y_pred)

        # Calculate loss
        # loss = F.nll_loss(y_pred, target)
        criterion = CrossEntropyLoss()
        loss = criterion(y_pred, target)
        regularization_loss = 0.0

        for param in model.parameters():
            if param.dim() > 1:
                regularization_loss += param.norm(1)

        regularization_loss *= l1_param
        loss += regularization_loss

        local_train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability

        pred_cats = pred[mask_cats]
        # print( " pred_cats = ", pred_cats)
        pred_dogs = pred[mask_dogs]
        # print( " pred_dogs = ", pred_dogs)
        correct += pred.eq(target.view_as(pred)).sum().item()
        correct_cats += (
            pred_cats.eq(target_cats.view_as(pred_cats)).sum().item()
        )
        correct_dogs += (
            pred_dogs.eq(target_dogs.view_as(pred_dogs)).sum().item()
        )
        processed += len(data)
        processed_cats += len(data[mask_cats])
        processed_dogs += len(data[mask_dogs])
        # print(" correct_cats: ", correct_cats, " processed_cats: ",  processed_cats)
        # print(" correct_dogs: ", correct_dogs, " processed_dogs: ",  processed_dogs)

        pbar.set_description(
            desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f} Cats Accuracy={100*correct_cats/processed_cats:0.2f} Dogs Accuracy={100*correct_dogs/processed_dogs:0.2f}"
        )
        local_train_acc.append(100 * correct / processed)
        local_cats_train_acc.append(100 * correct_cats / processed_cats)
        local_dogs_train_acc.append(100 * correct_dogs / processed_dogs)
    train_acc.append(sum(local_train_acc) / len(local_train_acc))
    cats_train_acc.append(
        sum(local_cats_train_acc) / len(local_cats_train_acc)
    )
    dogs_train_acc.append(
        sum(local_dogs_train_acc) / len(local_dogs_train_acc)
    )
    train_losses.append(sum(local_train_losses) / len(local_train_losses))


def test(
    model,
    device,
    test_loader,
    test_losses,
    test_acc,
    cats_test_acc,
    dogs_test_acc,
):
    model.eval()
    test_loss = 0
    correct = 0
    correct_cats = 0
    correct_dogs = 0
    processed = 0
    processed_cats = 0
    processed_dogs = 0

    with torch.no_grad():
        for test_data in test_loader:
            data, target = test_data["X"].to(device), test_data["Y"].to(device)
            mask_cats = target == classes.index("cats")
            # print("test: mask_cats = ", mask_cats)
            target_cats = target[mask_cats]
            # print( " target_cats = ", target_cats)

            mask_dogs = target == classes.index("dogs")
            # print("test mask_dogs = ", mask_dogs)
            target_dogs = target[mask_dogs]
            # print("target_dogs = ", target_dogs)

            output = model(data)
            criterion = CrossEntropyLoss()
            test_loss += criterion(output, target)
            # loss = criterion(y_pred, target)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_cats += (
                pred[mask_cats]
                .eq(target[mask_cats].view_as(pred[mask_cats]))
                .sum()
                .item()
            )
            correct_dogs += (
                pred[mask_dogs]
                .eq(target[mask_dogs].view_as(pred[mask_dogs]))
                .sum()
                .item()
            )
            processed += len(data)
            processed_cats += len(data[mask_cats])
            processed_dogs += len(data[mask_dogs])

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) Cats Accuracy: {}/{} ({:.2f}%) Dogs Accuracy: {}/{} ({:.2f}%) \n".format(
            test_loss,
            correct,
            processed,
            100.0 * correct / processed,
            correct_cats,
            processed_cats,
            100.0 * correct_cats / processed_cats,
            correct_dogs,
            processed_dogs,
            100.0 * correct_dogs / processed_dogs,
        )
    )

    test_acc.append(100.0 * correct / processed)
    cats_test_acc.append(100.0 * correct_cats / processed_cats)
    dogs_test_acc.append(100.0 * correct_dogs / processed_dogs)


class TrainImageDataset(Dataset):
    def __init__(self, root_dir, total_records, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
                        total_records: Number of records
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.total_records = total_records
        self.transform = transform
        self.file_index_list = {}

        for class_id, cls in enumerate(classes):
            self.file_index_list[class_id] = get_file_index_list(
                self.root_dir + "/" + cls
            )

    def __len__(self):
        return self.total_records

    def __getitem__(self, idx):
        factor = self.total_records / len(classes)
        class_id = int(idx / factor)
        file_id = int(idx - class_id * factor)
        indx_list = self.file_index_list[class_id]
        valid = False

        while not valid:
            img = None

            try:
                img = PILImage.open(
                    self.root_dir
                    + "/"
                    + classes[class_id]
                    + "/"
                    + indx_list[file_id]
                )
            except:
                # print("Train Excetpion caught while opening file:" + classes[class_id] + "/" + indx_list[file_id])
                pass

            if img is not None:
                if self.transform is not None:
                    img = self.transform(img)
                    # print(" img shape: ", img.shape)

            if (
                img is None
                or img.shape[0] != 3
                or img.shape[1] != 224
                or img.shape[2] != 224
            ):
                # print(classes[class_id] + "/" + indx_list[file_id] + " is not good")
                file_id = (file_id + 1) % 2000
            else:
                valid = True

        return {"X": img, "Y": class_id}


class TestImageDataset(Dataset):
    def __init__(self, root_dir, total_records, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.total_records = total_records
        self.file_index_list = {}

        for class_id, cls in enumerate(classes):
            self.file_index_list[class_id] = get_file_index_list(
                self.root_dir + "/" + cls
            )

    def __len__(self):
        return self.total_records

    def __getitem__(self, idx):
        factor = self.total_records / len(classes)
        class_id = int(idx / factor)
        file_id = int(idx - class_id * factor)
        # print("idx=", idx, " class_id=",  class_id, " file_id: ",file_id)
        # print("file_index_list: ",file_id)
        indx_list = self.file_index_list[class_id]
        valid = False
        while not valid:
            img = None

            try:
                img = PILImage.open(
                    self.root_dir
                    + "/"
                    + classes[class_id]
                    + "/"
                    + indx_list[file_id]
                )
            except:
                # print("Test Excetpion caught while opening file:" + classes[class_id] + "/" + indx_list[file_id])
                pass

            if img is not None:
                if self.transform is not None:
                    img = self.transform(img)
                    # print(" img shape: ", img.shape)

            if (
                img is None
                or img.shape[0] != 3
                or img.shape[1] != 224
                or img.shape[2] != 224
            ):
                # print(classes[class_id] + "/" + indx_list[file_id] + " is not good")
                file_id = (file_id + 1) % 500
            else:
                valid = True

        return {"X": img, "Y": class_id}


def get_train_transform():
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
        ]
    )
    # transforms.Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010)))])
    return transform


def get_test_transform():
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
        ]
    )
    # transforms.Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010)))])
    return transform


print("Generating Model")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# device = "cpu"
model = get_model()
model = model.to(device)
# summary(model, input_size=(3, 224, 224))

print("Populating Data")
# Defining all the hyper parameters
BATCH_SIZE = 10
EPOCHS = 10
# total_train_records = 1000
# total_test_records = 800
total_train_records = get_number_of_files(
    "data/train/cats"
) + get_number_of_files("data/train/dogs")
print(" total_train_records: ", total_train_records)
total_test_records = get_number_of_files(
    "data/validation/cats"
) + get_number_of_files("data/validation/dogs")
print(" total_test_records: ", total_test_records)

train_transform = get_train_transform()
test_transform = get_test_transform()


train_image_dataset = TrainImageDataset(
    root_dir="data/train",
    total_records=total_train_records,
    transform=train_transform,
)
test_image_dataset = TestImageDataset(
    root_dir="data/validation",
    total_records=total_test_records,
    transform=test_transform,
)

train_dl = DataLoader(
    train_image_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)
sample_train = next(iter(train_dl))
# print(sample_train['X'].shape, sample_train['Y'].shape)

test_dl = DataLoader(
    test_image_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)
sample_test = next(iter(test_dl))
# print(sample_test['X'].shape, sample_test['Y'].shape)


print("Starting training")
train_losses = []
test_losses = []
train_acc = []
cats_train_acc = []
dogs_train_acc = []
test_acc = []
cats_test_acc = []
dogs_test_acc = []
PATH = "./checkpoint.pth"


optimizer = optim.SGD(model.classifier[6].parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=6, gamma=0.1)


best_test_accuracy = 0.0


for epoch in range(EPOCHS):
    print("EPOCH:", (epoch + 1))
    train(
        model,
        device,
        train_dl,
        optimizer,
        epoch,
        train_losses,
        train_acc,
        cats_train_acc,
        dogs_train_acc,
    )
    test(
        model,
        device,
        test_dl,
        test_losses,
        test_acc,
        cats_test_acc,
        dogs_test_acc,
    )
    t_acc = test_acc[-1]
    if t_acc > best_test_accuracy:
        best_test_accuracy = t_acc
        torch.save(model.state_dict(), PATH)
        # model.to('cpu')
    scheduler.step()

metrics_records = {
    "train_acc": [train_acc[-1]],
    "cats_train_acc": [cats_train_acc[-1]],
    "dogs_train_acc": [dogs_train_acc[-1]],
    "test_acc": [test_acc[-1]],
    "cats_test_acc": [cats_test_acc[-1]],
    "dogs_test_acc": [dogs_test_acc[-1]],
}
df = pd.DataFrame(metrics_records)
df.to_csv("metrics.csv")
