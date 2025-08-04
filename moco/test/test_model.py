import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model.resnet import ResNet18Model
from utils.misc import load_config


def test_model():
    config = load_config("moco/config/moco_config.yaml")
    model = ResNet18Model(config)
    B, C, H, W = 2, 3, 512, 512
    x = torch.randn(B, C, H, W)
    out = model(x)
    assert out.shape == (B, 128)


if __name__ == "__main__":
    print("All tests passed!")
