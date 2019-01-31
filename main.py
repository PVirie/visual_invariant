import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataset import FashionMNIST


if __name__ == "__main__":
    print("main")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
