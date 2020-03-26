import random
import re
from glob import glob

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from config import IMG_DIR

mask_path = ''
image_path = ''
image_type = ''
mask_type = ''

def _mask_to_img(mask_file):
    img_file = re.sub(mask_path.format(IMG_DIR),
                      image_path.format(IMG_DIR), mask_file)
    img_file = re.sub('\.' + mask_type + '$', '.' + image_type, img_file)
    return img_file


def _img_to_mask(img_file):
    mask_file = re.sub(image_path.format(IMG_DIR),
                     mask_path.format(IMG_DIR), img_file)
    mask_file = re.sub('\.' + image_type + '$', '.' + mask_type, mask_file)
    return mask_file


def get_img_files(masks, imgs, mtype, itype):
    global mask_path
    mask_path = masks
    global image_path
    image_path = imgs
    global mask_type
    mask_type = mtype
    global image_type
    image_type = itype
    mask_files = sorted(glob((mask_path+'/*.'+mask_type).format(IMG_DIR)))
    return np.array([_mask_to_img(f) for f in mask_files])


class MaskDataset(Dataset):
    def __init__(self, img_files, transform, mask_transform=None, mask_axis=0):
        self.img_files = img_files
        self.mask_files = [_img_to_mask(f) for f in img_files]
        self.transform = transform
        if mask_transform is None:
            self.mask_transform = transform
        else:
            self.mask_transform = mask_transform
        self.mask_axis = mask_axis

    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_files[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask[:, :, self.mask_axis]

        seed = random.randint(0, 2 ** 32)

        # Apply transform to img
        random.seed(seed)
        img = Image.fromarray(img)
        img = self.transform(img)

        # Apply same transform to mask
        random.seed(seed)
        mask = Image.fromarray(mask)
        mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.img_files)
