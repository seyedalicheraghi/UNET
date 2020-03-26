import cv2
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from dataset_modified import MaskDataset, get_img_files
from nets.MobileNetV2_unet import MobileNetV2_unet
import glob


def get_data_loaders(val_files):
    val_transform = Compose([Resize((224, 224)), ToTensor(), ])
    val_loader = DataLoader(MaskDataset(val_files, val_transform), batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=4)
    return val_loader

precision_list = dict()
recall_list = dict()
split_list = dict()
merge_list = dict()
recall = []
precision = []
p_cnt = 0

predictions = list()
masks = list()
cnt = 0

list_of_models = glob.glob("/home/skeri/tentorch/unet/trained_models/trained_three/trained/*.pth")
models_numbers = []

for items in list_of_models:
    p1 = items.split('/')
    model_no = p1[len(p1) - 1].split('-')
    models_numbers.append(int(model_no[0]))
sorted_items = sorted(models_numbers)
sorted_items.append(sorted_items.pop(0))

list_of_models = []
for items in sorted_items:
    list_of_models.append(
        '/home/skeri/tentorch/unet/trained_models/trained_three/trained/' + str(items) + '-best.pth')
precision_list = []
recall_list = []
error_list = []
iou_list = []
IMG_SIZE = 224
for loading_model in list_of_models:
    img_size = (IMG_SIZE, IMG_SIZE)
    n_shown = 0
    image_files = get_img_files()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_loader = get_data_loaders(image_files)
    model = MobileNetV2_unet()
    model.load_state_dict(torch.load(loading_model))
    model.to(device)
    model.eval()
    with torch.no_grad():
        for I, M in data_loader:
            cnt += 1
            I = I.to(device)
            M = M.to(device)
            P = model(I)
            masks.append(M)
            predictions.append(P)

        TP = 0
        FP = 0
        FN = 0
        for i in range(0, cnt):
            M = masks[i].cpu().numpy().reshape(*img_size)*255
            M = M // 255
            sign_present = len(np.where(M > 0)[0]) > 0
            # print(sign_present)
            P = predictions[i].cpu().numpy().reshape(int(IMG_SIZE / 2), int(IMG_SIZE / 2)) * 255
            P = cv2.resize(P.astype(np.uint8), img_size)
            P[P > 0] = 1
            P[P < 1] = 0

            # AND between prediction and ground truth
            TP += len(np.where(P * M > 0)[0])
            # white pixels in P that don't have corresponding white pixels in ground truth
            FP += len(np.where(P - M > 0)[0])
            # print(FP)
            # white pixels in M that don't have correspondence in P
            FN += len(np.where(M - P > 0)[0])
            if TP > 0:
                recall.append(TP / (TP + FN))
                precision.append(TP / (TP + FP))
            else:
                recall.append(0)
                precision.append(0)
print(len(precision))
print(len(recall))