from collections import defaultdict
from math import floor

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCDetection


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

class ConvBlock(nn.Module):
    def __init__(self, in_chanels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_chanels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class InitialConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2, 2),
            
            ConvBlock(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),

            ConvBlock(192, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3),
            ConvBlock(256, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            nn.MaxPool2d(2, 2),

            nn.Sequential(
                *[
                    nn.Sequential(ConvBlock(512, 256, kernel_size=1), ConvBlock(256, 512, kernel_size=3, padding=1))
                    for _ in range(4)
                ]
            ),
            ConvBlock(512, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=1, padding=1),
            nn.MaxPool2d(2, 2),

            nn.Sequential(
                *[
                    nn.Sequential(ConvBlock(1024, 512, kernel_size=1), ConvBlock(512, 1024, kernel_size=3, padding=1))
                    for _ in range(2)
                ]
            )
        )

    def forward(self, x):
        return self.conv(x)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = InitialConv()
        self.fc = nn.Sequential(nn.AvgPool2d(7), nn.Flatten(), nn.Linear(1024, 1000))

    def forward(self, x):
        return self.fc(self.conv(x))

class YOLO(nn.Module):
    def __init__(self, n_features=7, n_bboxes=2, n_classes=20):
        super().__init__()
        self.S = n_features
        self.B = n_bboxes
        self.C = n_classes
        self.initial_conv = InitialConv()
        self.final_conv = nn.Sequential(
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.S * self.S * 1024, 496), nn.LeakyReLU(0.1, inplace=True), nn.Dropout(0.5),
            nn.Linear(496, self.S * self.S * (5 * self.B + self.C)), nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.initial_conv(x)
        out = self.final_conv(out)
        out = self.fc(out)
        return out.view(-1, self.S, self.S, 5 * self.B + self.C)


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

def get_precision_recall(y_pred, y, B, S):
    threshold = 0.5
    positives = y[..., B * 5:].sum()
    detections = 0
    tp = 0

    for sample_pred, sample_y in zip(y_pred, y):
        for i in range(sample_pred.shape[0]):
            for j in range(sample_pred.shape[1]):
                box_preds = sample_pred[i][j][:B * 5].reshape(B, 5)
                confident = box_preds[..., -1] > 0.5
                detections += confident.sum()
                _, pred_class = sample_pred[i][j][B * 5:].max(-1)
                if sample_y[i][j][B * 5 + pred_class] != 0:
                    for confident_idx in confident.nonzero():
                        box_conf_xy = torch.cat(
                            (
                                box_preds[confident_idx][..., :2] / S - 0.5 * box_preds[confident_idx][..., 2:4],
                                box_preds[confident_idx][..., :2] / S + 0.5 * box_preds[confident_idx][..., 2:4],
                            ), -1,
                        )
                        box_y = sample_y[i][j][:B*5].reshape(B, 5)
                        box_y_xy = torch.cat(
                            (
                                box_y[..., :2] / S - 0.5 * box_y[..., 2:4],
                                box_y[..., :2] / S + 0.5 * box_y[..., 2:4],
                            ), -1,
                        )
                        iou = IoU(box_y_xy, box_conf_xy)
                        if iou.max() > threshold:
                            tp += 1

    precision = tp / detections if detections != 0 else torch.tensor(0)
    recall = tp / positives if positives != 0 else torch.tensor(0)
    if torch.isnan(precision) or torch.isnan(recall):
        print("Dupa")
    return precision, recall


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

        self.train_dataset = VOCDetection(
            root="data/VOC/", transform=self.transform,
            target_transform=VOCTargetTransform(), image_set="train",
        )
        self.val_dataset = VOCDetection(
            root="data/VOC/", transform=self.transform,
            target_transform=VOCTargetTransform(), image_set="val",
        )
        self.model = YOLO()
        self.loss = YOLOLoss()


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def forward(self, x):
        return self.model(x)

    def get_precision_recall(self, y_pred, y):
        return get_precision_recall(y_pred, y, self.model.B, self.model.S)


    def step(self, batch, batch_idx, label):
        x, y = batch
        y_pred = self(x)

        (loss_coords, loss_dims, loss_obj, loss_noobj, loss_labels) = self.loss(y_pred, y)
        precision, recall = self.get_precision_recall(y_pred, y)
        self.log(f"{label}_precision", precision.item())
        self.log(f"{label}_recall", recall.item())
        self.log(f"{label}_loss_coords", loss_coords.item())
        self.log(f"{label}_loss_dims", loss_dims.item())
        self.log(f"{label}_loss_obj", loss_obj.item())
        self.log(f"{label}_loss_noobj", loss_noobj.item())
        self.log(f"{label}_loss_labels", loss_labels.item())
        loss = loss_coords + loss_dims + loss_obj + loss_noobj + loss_labels
        self.log(f"{label}_loss", loss.item())
        return loss / x.size(0)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


if __name__ == "__main__":
    module = YOLOModule()
    # x, y = module.train_dataset[0]
    # x, y = x.unsqueeze(0), y.unsqueeze(0)
    # pred = module(x)

    # B = 2
    # threshold = 0.5
    # fp = 0
    # tp = 0

    # for sample_pred, sample_y in zip(pred, y):
    #     for i in range(sample_pred.shape[0]):
    #         for j in range(sample_pred.shape[1]):
    #             box_preds = sample_pred[i][j][:B*5].reshape(B, 5)
    #             confident = box_preds[..., -1] > 0.5
    #             class_prob, pred_class = sample_pred[i][j][B*5:].max(-1)
    #             true_classes = (sample_y[i][j][B*5:] != 0).nonzero()

    #             if sample_y[i][j][B*5 + pred_class] == 0:
    #                 fp += confident.sum()
    #             else:
    #                 for confident_idx in confident.nonzero():
    #                     box_conf_xy = torch.cat((box_preds[confident_idx][..., :2] / 7 - 0.5 * box_preds[confident_idx][..., 2:4], box_preds[confident_idx][..., :2] / 7 + 0.5 * box_preds[confident_idx][..., 2:4]), -1)
    #                     box_y = sample_y[i][j][:B*5].reshape(B, 5)
    #                     box_y_xy = torch.cat((box_y[..., :2] / 7 - 0.5 * box_y[..., 2:4], box_y[..., :2] / 7 + 0.5 * box_y[..., 2:4]), -1)
    #                     iou = IoU(box_y_xy, box_conf_xy)
    #                     if iou.max() > threshold:
    #                         tp += 1
    #                     else:
    #                         fp += 1

                

    # preds_coords = pred[..., :5*2].reshape(*pred.shape[:-1], 2, 5)
    # target_coords = y[..., :5*2].reshape(*y.shape[:-1], 2, 5)
    # preds_confidences = preds_coords[..., -1]
    # preds_mask = (preds_confidences > 0.5)

    # confident_preds = preds_coords[preds_mask]
    # corresponding_targets = target_coords[preds_mask]
    # confident_preds_xy = torch.cat((confident_preds[..., :2] / 7 - 0.5 * confident_preds[..., 2:4], confident_preds[..., :2] / 7 + 0.5 * confident_preds[..., 2:4]), -1)
    # corresponding_targets_xy = torch.cat((corresponding_targets[..., :2] / 7 - 0.5 * corresponding_targets[..., 2:4], corresponding_targets[..., :2] / 7 + 0.5 * corresponding_targets[..., 2:4]), -1)
    # iou = IoU(*torch.broadcast_tensors(corresponding_targets_xy.unsqueeze(-3), confident_preds_xy.unsqueeze(-2)))[0]

    # confidences = pred[..., 5:(5 * (2 + 1)):5]
    # pred[coord_mask][..., :5*2].reshape(*pred.shape[:-1], 2, 5)

    # coord_mask = (y[..., 4] == 1)
    # noobj_mask = (y[..., 4] == 0)
    # coord_preds = pred[coord_mask][..., :5*2].reshape(-1, 2, 5)
    # coord_targets = y[coord_mask][..., :5*2].reshape(-1, 2, 5)

    # prediction_mask = (coord_preds[..., -1] > 0.5)



