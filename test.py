import argparse
import os
import torch
import cv2
import numpy as np
from model.FDADN import FDADN

m = FDADN()
a = torch.ones(2,3,64,64)
print(m(a).shape)