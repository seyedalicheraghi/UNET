import logging
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from dataset_modified import MaskDataset, get_img_files
from nets.MobileNetV2_unet import MobileNetV2_unet

import glob
from pylab import close, colorbar, figure, gray, hist, imshow, plot, savefig, show
import scipy
from math import sin, cos, tan, atan, pi
import statistics

IMG_SIZE = 224


def get_data_loaders(val_files):
    val_transform = Compose([Resize((224, 224)), ToTensor(), ])
    val_loader = DataLoader(MaskDataset(val_files, val_transform), batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=4)
    return val_loader


# References:
#       Arganda - Carreras I, Turaga SC, Berger DR, et al.(2015) Crowdsourcing the creation of image
#       segmentation algorithms for connectomics.Front.Neuroanat. 9: 142. DOI: 10.3389/ fnana .2015 .00142
# def precision_recall(im_true, im_test):
#     error, precision, recall = adapted_rand_error(im_true, im_test)
#
#     # Intersection over union
#     intersection = np.logical_and(im_true, im_test)
#     union = np.logical_or(im_true, im_test)
#     iou_score = np.sum(intersection) / np.sum(union)
#
#     # print(f"\n## Method: unet")
#     # print(f"error: {error}")
#     # print(f"precision: {precision}")
#     # print(f"recall: {recall}")
#     #
#     # fig, axes = plt.subplots(1, 3, figsize=(12, 6), constrained_layout=True)
#     # ax = axes.ravel()
#     # ax[0].imshow(im_true)
#     # ax[0].set_title('Hand Segmentation')
#     # ax[0].set_axis_off()
#     #
#     # ax[1].imshow(im_test)
#     # ax[1].set_title('Predicted Segmentation')
#     # ax[1].set_axis_off()
#     #
#     # ax[2].imshow(image)
#     # ax[2].set_title('Predicted Segmentation')
#     # ax[2].set_axis_off()
#     # plt.show()
#     return precision, recall, error, iou_score


def Gio_precision_recall(M, P):
    M = 1.0 * (M > 127)
    P = 1.0 * (P > 127)
    TP = len(np.where(P * M > 0)[0])  # AND between prediction (P) and ground truth (M)
    FP = len(np.where(cv2.subtract(P, M) > 0)[0])  # white pixels in P that don't have corresponding white pixels in M
    FN = len(np.where(cv2.subtract(M, P) > 0)[0])  # white pixels in M that don't have correspondence in P
    return TP, FP, FN


def calc_range3(f, h0, vlo, dv, gamma):  # this works!
    Z = h0 * (f * cos(gamma) - vlo * sin(gamma)) / dv
    return Z


def calc_range2(h0, gamma, alpha, delta):
    return h0 / (tan(gamma + alpha + delta) - tan(gamma + alpha))


def angle_from_center(f, p):  # pixel p is 0 at center of 1D image
    return atan(p / f)


def distanceMeasuring(apparent_height, bottom_row, gamma):
    # gamma = gamma * pi / 180
    h0 = .20  # meters
    f = 1602
    #f = (112 / 1440) * f
    #h, w = 149.34, 112
    h, w = 1920, 1440
    alpha = angle_from_center(f, h / 2 - bottom_row)
    delta = angle_from_center(f, h / 2 - (bottom_row - apparent_height)) - angle_from_center(f, h / 2 - bottom_row)
    Z_Cal3 = calc_range3(f, h0, h / 2 - bottom_row, apparent_height, gamma)
    Z_Cal2 = calc_range2(h0, gamma, alpha, delta)
    return Z_Cal2


def evaluate():
    model_path = "/home/skeri/tentorch/unet/trained_models/ali_best/trained/"
    mask_type = 'png'
    image_type = 'jpg'
    dataset_path = '/home/skeri/Squared_Unrolled_Exit_Signs_by_Distance_and_Their_UNET_Masks/'
    list_of_models = glob.glob(model_path + "*.pth")
    models_numbers = []

    for items in list_of_models:
        p1 = items.split('/')
        model_no = p1[len(p1) - 1].split('-')
        models_numbers.append(int(model_no[0]))

    sorted_items = sorted(models_numbers)
    sorted_items.append(sorted_items.pop(0))

    list_of_models = []
    for items in sorted_items:
        list_of_models.append(model_path + str(items) + '-best.pth')
    # precision_list = []
    # recall_list = []
    # error_list = []
    # iou_list = []

    # print('  FP  ', '  TP  ', '  FN  ', '      Recall  ', '  Precision  ', '  Model  ')

    for loading_model in list_of_models:
        print(loading_model)
        list_of_distances = []
        lm = loading_model.split('/')
        for meters in range(2, 7):
            distances = []
            img_size = (IMG_SIZE, IMG_SIZE)

            image_files = get_img_files(dataset_path + str(meters) + '/masks',
                                        dataset_path + str(meters) + '/imgs',
                                        mask_type, image_type)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            data_loader = get_data_loaders(image_files)
            model = MobileNetV2_unet()
            model.load_state_dict(torch.load(loading_model))
            model.to(device)
            model.eval()

            # TP = 0
            # FP = 0
            # FN = 0
            counter = 0

            with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)

                    for image, label, output in zip(inputs, labels, outputs):
                        image = image.cpu().numpy().transpose((1, 2, 0)) * 255
                        image = cv2.resize(image.astype(np.uint8), img_size)

                        label = label.cpu().numpy().reshape(*img_size) * 255
                        label = cv2.resize(label.astype(np.uint8), img_size)

                        output = output.cpu().numpy().reshape(int(IMG_SIZE / 2), int(IMG_SIZE / 2)) * 255

                        # print(output.shape)
                        output = cv2.resize(output.astype(np.uint8), img_size)
                        output[output>0] = 255

                        #

                        # print('\r', FP, TP, FN, end='')
                        # # time.sleep(0.1)
                        # r1, r2, r3 = Gio_precision_recall(label, output)
                        # TP = r1 + TP
                        # FP = r2 + FP
                        # FN = r3 + FN

                        if len(np.where(output > 0)[0]) != 0:

                            output1 = cv2.connectedComponentsWithStats(output, 4, cv2.CV_32S)
                            num_labels = output1[0]
                            labels = output1[1]
                            while num_labels>2:
                                small_component = 0
                                component_size = labels[labels == 0].shape[0]
                                for i in range(1, num_labels):
                                    if component_size > labels[labels == i].shape[0]:
                                        small_component = i
                                        component_size = labels[labels == i].shape[0]
                                labels[labels==small_component]=0
                                num_labels-=1
                            output[labels==0]=0
                            # print(labels)
                            dim = (1440, 1440)
                            # resize image
                            output = cv2.resize(output, dim, interpolation=cv2.INTER_AREA)
                            # fig, axes = plt.subplots(1, 3, figsize=(12, 6), constrained_layout=True)
                            # ax = axes.ravel()
                            # ax[0].imshow(label)
                            # ax[0].set_title('Hand Segmentation')
                            # ax[0].set_axis_off()
                            minY = np.amin(np.where(output > 0)[0])
                            minX = np.amin(np.where(output > 0)[1])
                            maxY = np.amax(np.where(output > 0)[0])
                            maxX = np.amax(np.where(output > 0)[1])
                            output1 = output
                            output1[output > 128] = 255
                            output1[output <= 128] = 0

                            nz = np.where(output1 != 0)
                            if len(nz[0]) != 0 and len(nz[1] != 0):
                                height = []
                                for iteration in range(np.amin(nz[1]), np.amax(nz[1])):
                                    height.append(np.where(output1[:, iteration] != 0))

                                av_height = []
                                for iterations in height:
                                    if (len(iterations[0])) != 0:
                                        mx = np.amax(iterations)
                                        mn = np.amin(iterations)
                                        av_height.append(mx - mn)

                                # output = cv2.rectangle(output, (minX, minY), (maxX, maxY), (255, 255, 255), -1)

                                sp0 = image_files.item(counter).split('/')
                                sp1 = sp0[len(sp0) - 1].split('_')
                                # print(image_files.item(counter), sp1[1])
                                #distances.append(distanceMeasuring(output, maxY - minY, maxY, float(sp1[1])))
                                distace_par = distanceMeasuring(statistics.median(av_height),
                                                                   maxY,
                                                                   float(sp1[1]))
                                distances.append(distace_par)
                                # print(image_files.item(counter), distace_par, statistics.mean(av_height), maxY, float(sp1[1]))
                            # ax[1].imshow(output)
                            # ax[1].set_title('Predicted Segmentation')
                            # ax[1].set_axis_off()
                            # ax[2].imshow(image)
                            # ax[2].set_title('Predicted Segmentation')
                            # ax[2].set_axis_off()
                            # plt.show()
                        # else:
                        #     distances.append(0)
                        #     # fig, axes = plt.subplots(1, 3, figsize=(12, 6), constrained_layout=True)
                        #     # ax = axes.ravel()
                        #     # ax[0].imshow(label)
                        #     # ax[0].set_title('Hand Segmentation')
                        #     # ax[0].set_axis_off()
                        #     # ax[1].imshow(output)
                        #     # ax[1].set_title('Predicted Segmentation')
                        #     # ax[1].set_axis_off()
                        #     # ax[2].imshow(image)
                        #     # ax[2].set_title('Predicted Segmentation')
                        #     # ax[2].set_axis_off()
                        #     # plt.show()
                        counter += 1
            print(distances, meters)

            list_of_distances.append(distances)
        #     print(list_of_distances)
        # print("*********************")
        # print(list_of_distances)
        counter = 2
        for items in list_of_distances:
            print("Distance of ", counter, " for loading model ", loading_model)
            if len(items)>2:
                print("Standard Deviation of sample is % s "
                      % (statistics.stdev(items)))
                print("Median of sample is % s "
                      % (statistics.median(items)))
                print("Mean of sample is % s "
                      % (statistics.mean(items)))
                print("Max of sample is % s "
                      % (np.amax(items)))
                print("Min of sample is % s "
                      % (np.amin(items)))
            axes = plt.gca()
            axes.set_xlim([0, 160])
            axes.set_ylim([1, 9])
            axes.plot(items, label=str(counter)+" M")
            counter += 1

            #     if (TP + FN) != 0 and (TP + FP) != 0:
#         recall_list.append(TP / (TP + FN))
#         precision_list.append(TP / (TP + FP))
#         print("  -->  ", round(TP / (TP + FN), 2), "    ", round(TP / (TP + FP), 2), "----->", loading_model)
#     else:
#         recall_list.append(0)
#         precision_list.append(0)
#         print("  -->  ", 0, "    ", 0, "----->", loading_model)
        #
        # plt.plot(list_of_distances, label='Precision')
        # plt.plot(recall_list, label='Recall')
        # plt.plot(error_list, label='Error')
        # plt.plot(iou_list, label='IOU')
        # plt.legend()
        #
        plt.legend()
        plt.savefig('Squared_Unrolled_Parches_From_Top_Left_Z_Cal2_' + lm[len(lm)-1] +'.png')
        plt.cla()
        # plt.show()


if __name__ == '__main__':

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(logging.FileHandler(filename="outputs/{}.log".format('train_unet')))

    evaluate()
