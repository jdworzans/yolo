from model import get_precision_recall
import torch


def test_precision_recall():
    y = torch.tensor(
        [
            0.5, 0.5, 0.1, 0.1, 1,
            0.0, 0.0, 0.0, 0.0, 0.0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
    ).reshape(1, 1, 1, 30)
    y_pred = torch.tensor(
        [
            0.5, 0.5, 0.1, 0.1, 1,
            0.0, 0.0, 0.0, 0.0, 0.0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
    ).reshape(1, 1, 1, 30)
    precision, recall = get_precision_recall(y_pred, y, 2, 1)
    assert precision == 1
    assert recall == 1

if __name__ == "__main__":
    test_precision_recall()
