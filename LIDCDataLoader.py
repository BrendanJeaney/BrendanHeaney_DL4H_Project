import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize
import matplotlib.pyplot as plt
# import fmatch
import random
from glob import glob
import numpy as np
import cv2



class LIDCDataLoader(Dataset):
    def __init__(self, lidc_dir, img_size=128, transform=None, target_transform=None, max_samples=200, return_targets=False):
        self.samples = []
        self.img_size = img_size
        self.return_targets_flag = return_targets

        for patient_dir in glob(os.path.join(lidc_dir, 'LIDC-IDRI-*')):
            for nodule_dir in glob(os.path.join(patient_dir, 'nodule-*')):
                # print(nodule_dir)
                images_dir = os.path.join(nodule_dir, "images")
                if not os.path.exists(images_dir):
                    continue
                image_files = os.listdir(images_dir)
                # print(image_files)
                for fname in image_files:
                    img_path = os.path.join(images_dir, fname)
                    mask_paths = []
                    for mask_dir in glob(os.path.join(nodule_dir, "mask-*")):
                        mpath = os.path.join(mask_dir, fname)
                        if os.path.exists(mpath):
                            mask_paths.append(mpath)
                    if mask_paths:
                        self.samples.append((img_path, mask_paths))

        # print(self.samples)
        random.shuffle(self.samples)
        self.samples = self.samples[:max_samples]
        self.transform = Compose([Resize((img_size, img_size)), ToTensor()])



    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_path, mask_paths = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))/255.0

        # masked = np.zeros((self.img_size, self.img_size))
        combined_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        for mpath in mask_paths:
            m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (self.img_size, self.img_size))
            combined_mask = np.maximum(combined_mask, m)


        # combined_mask = combined_mask > 127
        # print(type(combined_mask))
        combined_mask = (combined_mask > 127).astype(np.float32)

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(combined_mask, dtype=torch.float32).unsqueeze(0)

        if self.return_targets_flag:
            ys, xs = np.where(combined_mask > 0.3)
            if len(xs) == 0 or len(ys) == 0:
                box = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
                cls = torch.tensor([0.0], dtype=torch.float32)
                return img_tensor, mask_tensor, box, cls
            else:
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32) / self.img_size
                cls = torch.tensor([1.0], dtype=torch.float32)
                return img_tensor, mask_tensor, box, cls
            
        else:
            return img_tensor, mask_tensor

        # return torch.tensor(img, dtype=torch.float32).unsqueeze(0), torch.tensor(combined_mask, dtype=torch.float32).unsqueeze(0)

