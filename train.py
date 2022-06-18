from collections import defaultdict
from math import floor
from turtle import forward

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCDetection

from model import YOLO

TARGET_SIZE = (448, 448)
NAME_TO_ID = {
    'horse': 0,
    'bird': 1,
    'boat': 2,
    'pottedplant': 3,
    'bicycle': 4,
    'cow': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'train': 9,
    'chair': 10,
    'motorbike': 11,
    'tvmonitor': 12,
    'person': 13,
    'aeroplane': 14,
    'sofa': 15,
    'dog': 16,
    'sheep': 17,
    'diningtable': 18,
    'bottle': 19,
}

class VOCTargetTransform(nn.Module):
    def __init__(self, n_features=7, n_bboxes=2, n_classes=20):
        super().__init__()
        self.S = n_features
        self.B = n_bboxes
        self.C = n_classes

    def forward(self, x):
        size = x["annotation"]["size"]
        w, h = float(size["width"]), float(size["height"])
        objects = x["annotation"]["object"]
        cell_size = 1 / self.S
        data = defaultdict(list)
        target = torch.zeros(self.S, self.S, 5 * self.B + self.C)
        for obj in objects:
            label = int(NAME_TO_ID[obj["name"]])
            bndbox = obj["bndbox"]
            box_w = (float(bndbox["xmax"]) - float(bndbox["xmin"])) / (2 * w)
            box_h = (float(bndbox["ymax"]) - float(bndbox["ymin"])) / (2 * h)
            center_x = (float(bndbox["xmax"]) + float(bndbox["xmin"])) / (2 * w)
            center_y = (float(bndbox["ymax"]) + float(bndbox["ymin"])) / (2 * h)
            i = floor(center_x / cell_size)
            j = floor(center_y / cell_size)

            data[(i, j)].append(
                (label, torch.tensor([center_x - i * cell_size, center_y - j * cell_size, box_w, box_h, 1]))
            )

        for (i, j), elements in data.items():
            for box_idx, (label, element) in zip(range(self.B), elements):
                target[..., i, j, 5*box_idx:5*box_idx + 5] = element
                target[..., i, j, 5*self.B + label] = 1
        return target

def area(box):
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])

def IoU(box1, box2):
    lower = torch.max(box1[..., :2], box2[..., :2])
    upper = torch.min(box1[..., 2:], box2[..., 2:])
    intersection = (upper - lower).clamp(min=0).prod(-1)
    union = area(box1) + area(box2) - intersection
    return intersection / union


class YOLOLoss(nn.Module):
    def __init__(self, n_features=7, n_bboxes=2, n_classes=20,
                lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = n_features
        self.B = n_bboxes
        self.C = n_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj


class YOLOModule(pl.LightningModule):
    def __init__(self, batch_size: int = 16) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(TARGET_SIZE),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.dataset = VOCDetection(
            root="data/VOC/", transform=self.transform,
            target_transform=VOCTargetTransform(),
        )
        self.model = YOLO()


    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=8)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        coord_mask = y[..., 4] > 0

        loss = F.cross_entropy(y_pred.transpose(1, 2), y)
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    module = YOLOModule()
    for idx, (x, y) in zip(range(3), module.train_dataloader()):
        print(x.shape, y.shape)
        print(module(x).shape)
