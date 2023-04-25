import pandas as pd
from torch.utils import data
import numpy as np
import torch
import os
import io
import torchvision.transforms as transforms
from PIL import ImageFile
from PIL import Image
# from imblearn.over_sampling import RandomOverSampler
import torchvision.utils as vutils
import cv2
from tqdm import tqdm

img_name = np.squeeze(pd.read_csv('test_img.csv'))
reolution_ft = 768

for i in tqdm(range(len(img_name))):
    path = os.path.join("./dataset/test", img_name[i] + '.jpeg')

    img = cv2.imread(path)
    min_size = img.shape[0] if img.shape[0]<img.shape[1] else img.shape[1] 
    scale = reolution_ft/min_size


    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite('./dataset/test_resize/' + img_name[i] + '.jpeg', resized_img)
