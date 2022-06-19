from collections import defaultdict
from math import floor

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI
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
    return intersection / (union + 1e-6)


class YOLOLoss(nn.Module):
    def __init__(self, n_features=7, n_bboxes=2, n_classes=20,
                lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = n_features
        self.B = n_bboxes
        self.C = n_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds, targets):
        coord_mask = (targets[..., 4] == 1)
        noobj_mask = (targets[..., 4] == 0)

        coord_preds = preds[coord_mask][..., :5*self.B].reshape(-1, self.B, 5) # (n_coord, B, 5)
        coord_targets = targets[coord_mask][..., :5*self.B].reshape(-1, self.B, 5)  # (n_coord, B, 5)
        preds_xy = torch.cat(
            (
                coord_preds[..., :2] / self.S - 0.5 * coord_preds[..., 2:4],
                coord_preds[..., :2] / self.S + 0.5 * coord_preds[..., 2:4],
            ), -1
        )
        targets_xy = torch.cat(
            (
                coord_targets[..., :2] / self.S - 0.5 * coord_targets[..., 2:4],
                coord_targets[..., :2] / self.S + 0.5 * coord_targets[..., 2:4],
            ), -1
        )
        iou = IoU(*torch.broadcast_tensors(targets_xy.unsqueeze(-3), preds_xy.unsqueeze(-2)))
        max_iou, max_idxs = iou.max(-2)
        coord_response_preds = torch.gather(coord_preds, 1, max_idxs.unsqueeze(-1).expand_as(coord_preds))

        # Part 1
        loss_coords = F.mse_loss(coord_response_preds[..., :2], coord_targets[..., :2], reduction="sum")

        # Part 2
        loss_dims = F.mse_loss(torch.sqrt(coord_response_preds[..., 2:4]), torch.sqrt(coord_targets[..., 2:4]), reduction="sum")

        # Part 3
        loss_obj = F.mse_loss(coord_response_preds[..., 4], max_iou, reduction="sum")

        # Part 4
        noobj_preds = preds[noobj_mask]
        loss_noobj = noobj_preds[..., 4:self.B*5:5].square().sum()

        # Part 5
        label_preds = preds[coord_mask][..., 5*self.B:]
        label_targets = targets[coord_mask][..., 5*self.B:]
        loss_labels = F.mse_loss(label_preds, label_targets, reduction="sum")

        return (self.lambda_coord * loss_coords, self.lambda_coord * loss_dims, loss_obj, self.lambda_noobj * loss_noobj, loss_labels)


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
        self.loss = YOLOLoss()


    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=8)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        (loss_coords, loss_dims, loss_obj, loss_noobj, loss_labels) = self.loss(y_pred, y)
        self.log("train_loss_coords", loss_coords.item())
        self.log("train_loss_dims", loss_dims.item())
        self.log("train_loss_obj", loss_obj.item())
        self.log("train_loss_noobj", loss_noobj.item())
        self.log("train_loss_labels", loss_labels.item())
        loss = loss_coords + loss_dims + loss_obj + loss_noobj + loss_labels
        self.log("train_loss", loss.item())
        return loss / x.size(0)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


if __name__ == "__main__":
    # module = YOLOModule()
    # for idx, (x, y) in zip(range(3), module.train_dataloader()):
    #     print(x.shape, y.shape)
    #     y_pred = module(x)
    #     print(y_pred.shape, y[y[..., 4] == 1].shape)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="model_checkpoints/",
        filename="model-{epoch:02d}-{train_loss:.2f}",
        save_top_k=-1,
        mode="min",
    )

    cli = LightningCLI(
        YOLOModule,
        trainer_defaults={'gpus': 1, 'callbacks': [checkpoint_callback]},
        seed_everything_default=1234,
        save_config_overwrite=True
    )
