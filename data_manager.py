import random
import cv2
from PIL import Image
from torch.utils.data import Dataset


class MaskDataset(Dataset):
    def __init__(self, img_files, transform, mask_transform=None, mask_axis=0):
        self.img_files = img_files
        self.transform = transform

    def __getitem__(self, idx):
        img = self.img_files
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seed = random.randint(0, 2 ** 32)
        # Apply transform to img
        random.seed(seed)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_files)
