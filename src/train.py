import torch
import argparse
import yaml
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomRotation, RandomHorizontalFlip
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import dataframe_image as dfi

def setup():
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    args = parser.parse_args()
    with open(args.config,"r") as file:
        config = yaml.safe_load(file)
    


if __name__ == '__main__':
    main()