import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
import os
from torchvision import transforms
import cv2

def getData(mode):
    if mode == 'train' or mode == "train_resize":
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode, transform=None):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transform = transforms.Compose(transform + [transforms.ToTensor(), transforms.Lambda(lambda x: x/255)])
        # self.transform = transforms.Compose(transform )
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = os.path.join(self.root + self.mode + "/" + self.img_name[index] + ".jpeg")
        image = Image.open(path)
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        upper = (height - min_dim) // 2
        right = left + min_dim
        lower = upper + min_dim
        # Crop the image
        cropped_image = image.crop((left, upper, right, lower))
        
        # Downsample the image to 512x512
        resized_image = cropped_image.resize((512, 512), Image.BICUBIC)

        # image = cv2.imread(self.root + "new_" + self.mode + "/" + self.img_name[index] + ".jpeg")
        # height, width = image.shape[:2]
        # min_dim = min(width, height)
        # left = (width - min_dim) // 2
        # upper = (height - min_dim) // 2
        # right = left + min_dim
        # lower = upper + min_dim
        # # Crop the image
        # cropped_image = image[upper:lower, left:right]

        # resized_image = cv2.resize(cropped_image, (512, 512))
        


        # cv2.imwrite('My Image.png', resized_image)
        # print(img)
        # resized_image.save("1.png")
        img = self.transform(resized_image)
        label = self.label[index]
        return img, label
