import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import statistics
import os.path
import math
import glob

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from data_manager import MaskDataset
from nets.MobileNetV2_unet import MobileNetV2_unet
from scipy import ndimage
from math import tan, atan

# h0 = .21 # meters
focal_length = 1602


def get_data_loaders(val_files):
    val_transform = Compose([Resize((224, 224)), ToTensor(), ])
    val_loader = DataLoader(MaskDataset(val_files, val_transform), batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=0)
    return val_loader


def Gio_precision_recall(M, P):
    TP = len(np.where(P * M > 0)[0])  # AND between prediction (P) and ground truth (M)
    FP = len(np.where(cv2.subtract(P, M) > 0)[0])  # white pixels in P that don't have corresponding white pixels in M
    FN = len(np.where(cv2.subtract(M, P) > 0)[0])  # white pixels in M that don't have correspondence in P
    return TP, FP, FN


def calc_range2(h0, gamma, alpha, delta):
    return h0 / (tan(gamma + alpha + delta) - tan(gamma + alpha))


def angle_from_center(f, p):  # pixel p is 0 at center of 1D image
    return atan(p / f)


def distanceMeasuring(apparent_height, bottom, gamma, h0):
    height, width = 1920, 1440
    alpha = angle_from_center(focal_length, height / 2 - bottom)
    delta = angle_from_center(
        focal_length, height / 2 - (bottom - apparent_height)) - angle_from_center(focal_length, height / 2 - bottom)
    Z_Cal2 = calc_range2(h0, gamma, alpha, delta)
    return Z_Cal2


def UNET_segmentation(image, output_size):
    with torch.no_grad():
        inputs = image.to(device)
        outputs = model(inputs)
        output = outputs.cpu().numpy().reshape(112, 112) * 255
        output = cv2.resize(output.astype(np.uint8), (output_size, output_size))
        return output


def roll_pitch_yaw(images, COREML):
    # read the mask file with the same name as image file
    split_name_1 = images.split('/')
    split_name_2 = split_name_1[len(split_name_1) - 1].split('.')
    # read the CoreML txt file
    for items in range(0, len(COREML)):
        if COREML[items].strip() == split_name_2[0].strip():
            sp = COREML[items + 5].split('<Float>')
            sp1 = sp[1].split(',')
            sp2 = sp1[0].split('(')
            sp3 = sp1[2].split(')')
            pitch = float(sp2[1])
            yaw = float(sp1[1].strip())
            roll = float(sp3[0].strip())
            return roll, pitch, yaw


def find_connected_components(output, dim):
    output1 = cv2.connectedComponentsWithStats(output, 4, cv2.CV_32S)
    num_labels = output1[0]
    labels = output1[1]
    while num_labels > 2:
        small_component = 0
        component_size = labels[labels == 0].shape[0]
        for i in range(1, num_labels):
            if component_size > labels[labels == i].shape[0]:
                small_component = i
                component_size = labels[labels == i].shape[0]
        labels[labels == small_component] = 0
        num_labels -= 1
    output[labels == 0] = 0
    # resize image
    output = cv2.resize(output, (dim, dim), interpolation=cv2.INTER_AREA)
    return output


def imageShow(image_to_show):
    cv2.imshow('image', image_to_show)
    cv2.waitKey(500)


def main(list_of_models, dataset_path):
    h0 = 200 / 1000
    file_to_write = open(str(h0) + ".txt", "a")
    print(h0)
    for loading_model in list_of_models:
        # List of distances contains estimated distances for various distance categories
        list_of_distances = []
        recall1 = []
        precision1 = []
        recall2 = []
        precision2 = []
        unrolled_path = "/home/skeri/Saved_Data/Urolled_Dataset/Exit Signs by Distance/"
        zoom_path = "/home/skeri/Saved_Data/Zoom_BMP/Exit Signs by Distance/"
        # We want to estimate distances for images captured 2 to 8 meters away from the camera
        for meters in range(2, 9):
            TP1 = TP2 = FP1 = FP2 = FN1 = FN2 = 0
            # Get images within image dataset
            image_directory = dataset_path + str(meters)
            images_folders = os.listdir(image_directory)
            # Distances only contain distances within one category. It means it only contain 2 meters or 3 meters etc.
            distances = []
            # Find folders within dataset folder. for example it takes folder 'a' within '2' meters folder within
            # 'Exit Signs by Distance' folder
            for folders in images_folders:
                # Following lines read ARKIT data which is saved in txt file and save them within coreml_data variable
                ARKIT_directory = glob.glob(image_directory + "/" + folders + "/*.txt")[0]
                file_ARIKIT = open(ARKIT_directory, "r")
                coreml_data = []
                unrolled_path_imgs = ''
                unrolled_path_masks = ''
                unrolled_path_predicted_masks = ''
                zoom_path_imgs = ''
                zoom_path_masks = ''
                for x in file_ARIKIT:
                    coreml_data.append(x)
                file_ARIKIT.close()
                # Get list of images within img folder. We don't get the mask folder here
                list_of_images = glob.glob(image_directory + "/" + folders + "/imgs/*.png")
                # Define the UNET model we need for prediction
                model.load_state_dict(torch.load(loading_model))
                # Start to read each image within the list of images we already have
                for images in list_of_images:
                    # Get Pitch, Roll, and Yaw for the picture we want to predict and then unroll it.
                    # For unrolling we also execute rotation; hence we do both rotation and unrolling.
                    # To see the result uncomment imageShow
                    roll, pitch, yaw = roll_pitch_yaw(images, coreml_data)
                    unrolled_image = ndimage.rotate(cv2.imread(images), (roll * 180) / math.pi)
                    s1 = images.split('/')
                    s2 = s1[len(s1) - 1].split('.')
                    mask_path = ''
                    for i in range(0, len(s1) - 2):
                        mask_path += s1[i] + "/"
                    print(mask_path)
                    mask_path += "masks/" + s2[0] + ".bmp"
                    unrolled_path_imgs = unrolled_path + s1[len(s1) - 4] + '/' + s1[len(s1) - 3] + \
                                         "/imgs/" + s2[0] + ".png"
                    unrolled_path_masks = unrolled_path + s1[len(s1) - 4] + '/' + s1[len(s1) - 3] + \
                                         "/masks/" + s2[0] + ".png"
                    unrolled_path_predicted_masks = unrolled_path + s1[len(s1) - 4] + '/' + s1[len(s1) - 3] + \
                                         "/masks predicted/" + s2[0] + ".png"
                    zoom_path_predicted_masks = zoom_path + s1[len(s1) - 4] + '/' + s1[len(s1) - 3] + \
                                         "/masks predicted/" + s2[0] + ".png"
                    zoom_path_masks = zoom_path + s1[len(s1) - 4] + '/' + s1[len(s1) - 3] + \
                                         "/masks/" + s2[0] + ".png"
                    zoom_path_imgs = zoom_path + s1[len(s1) - 4] + '/' + s1[len(s1) - 3] + \
                                         "/imgs/" + s2[0] + ".png"
                    unrolled_mask = ndimage.rotate(cv2.imread(mask_path), (roll * 180) / math.pi)
                    # print(unrolled_path_masks, unrolled_path_imgs, unrolled_path_predicted_masks)
                    print(zoom_path_masks, zoom_path_imgs, zoom_path_predicted_masks)
                    # imageShow(unrolled_image)
                    name = images.split("/")
                    # cv2.imwrite("/home/skeri/im1/" + name[len(name) - 3] + name[len(name) - 1], unrolled_image)
                    # Crop top left corner of the unrolled image
                    cropped_from_original = unrolled_image[0:1440, 0:1440]
                    cropped_mask_original = unrolled_mask[0:1440, 0:1440]
                    # Write images and masks
                    cv2.imwrite(unrolled_path_imgs, cropped_from_original)
                    cv2.imwrite(unrolled_path_masks, cropped_mask_original)
                    # cv2.imwrite("/home/skeri/im2/" + name[len(name) - 3] + name[len(name) - 1], cropped_from_original)
                    # Prepare cropped unrolled image for prediction. This step includes converting 'numpy.ndarray' image
                    # to 'torch.utils.data.dataloader.DataLoader'
                    to_predict = next(iter(get_data_loaders(cropped_from_original)))
                    # Predict the data and then resize the output to 1440x1440 image size
                    output = UNET_segmentation(to_predict, 1440)
                    cv2.imwrite(unrolled_path_predicted_masks, output)

                    TP0, FP0, FN0 = Gio_precision_recall(cropped_mask_original, output)
                    # It labels different detected components of the predicted image. Here we try to only consider the
                    # biggest connected components and remove small ones. It helps us to only consider closest sign.
                    # To see the result uncomment imageShow
                    output = find_connected_components(output, 1440)
                    # imageShow(output)
                    # cv2.imwrite("/home/skeri/im3/" + name[len(name) - 3] + name[len(name) - 1], output)
                    # Next step to Zoom into detected area. To do that we find the center of biggest detected connected
                    # component and then use it to crop 448x448 from the unrolled rotated cropped image we already have
                    # This step help the algorithm to have more precise segmentation
                    # Previous steps could be replace with any detection algorithm.
                    # At the moment (3/31/2020) we use MobileNetV2 then UNET then MobileNetV2 then UNET.
                    # We can remove the first UNET and maybe the second MobileNetV2.
                    # To see the detected image uncomment the imageShow
                    if np.where(output > 0)[0].size == 0:
                        print(images + "\n")
                    else:
                        minY = np.amin(np.where(output > 0)[0])
                        minX = np.amin(np.where(output > 0)[1])
                        maxY = np.amax(np.where(output > 0)[0])
                        maxX = np.amax(np.where(output > 0)[1])
                        if int(((maxX + minX) / 2) - 224) > 0:
                            top_corner_X = int(((maxX + minX) / 2) - 224)
                        else:
                            top_corner_X = 0
                        if int(((maxY + minY) / 2) - 224) > 0:
                            top_corner_Y = int(((maxY + minY) / 2) - 224)
                        else:
                            top_corner_Y = 0
                        detected_image = cropped_from_original[
                                         top_corner_Y:top_corner_Y + 448,
                                         top_corner_X:top_corner_X + 448]

                        Mask_image = cropped_mask_original[
                                                  top_corner_Y:top_corner_Y + 448,
                                                  top_corner_X:top_corner_X + 448]
                        cv2.imwrite(zoom_path_imgs, detected_image)
                        cv2.imwrite(zoom_path_masks, Mask_image)
                        # imageShow(detected_image)
                        # imageShow(Mask_image)
                        # cv2.imwrite("/home/skeri/im4/" + name[len(name) - 3] + name[len(name) - 1], detected_image)
                        # Second round of UNET segmentation. It tries to segment cropped 448 x 448 images from last step
                        to_predict = next(iter(get_data_loaders(detected_image)))
                        output = UNET_segmentation(to_predict, 448)
                        cv2.imwrite(zoom_path_predicted_masks, output)
                        cv2.imshow('image', output)
                        cv2.waitKey(1000)
                        exit(0)
                        TP3, FP3, FN3 = Gio_precision_recall(Mask_image, output)
                        TP1 += TP0
                        FP1 += FP0
                        FN1 += FN0
                        TP2 += TP3
                        FP2 += FP3
                        FN2 += FN3
                        # cv2.imwrite("/home/skeri/im5/" + name[len(name) - 3] + name[len(name) - 1], output)
                        # Labels different detected components of the predicted image. Here we try to only consider the
                        # biggest connected components and remove small ones. It helps us to only consider closest sign.
                        output = find_connected_components(output, 448)
                        # The following section find the mean or median height of the predicted component
                        rows = np.where(output > 0)[0]  # height
                        columns = np.where(output > 0)[1]  # width
                        minX = np.amin(np.where(output > 0)[1])
                        maxX = np.amax(np.where(output > 0)[1])
                        av_height = []
                        bottom_height = []
                        for items in range(minX, maxX + 1):
                            flag = False
                            for i in range(0, len(rows)):
                                if columns[i] == items:
                                    if not flag:
                                        flag = True
                                        min_Y = rows[i]
                                        max_Y = rows[i]
                                    if min_Y > rows[i]:
                                        min_Y = rows[i]
                                    if max_Y < rows[i]:
                                        max_Y = rows[i]
                            if flag:
                                av_height.append((max_Y - min_Y) + 1)
                                bottom_height.append(max_Y)
                        # Estimate the distance based on James Algorithm
                        distace_par = distanceMeasuring(statistics.median(av_height),
                                                        statistics.median(bottom_height), pitch, h0)
                        # print(str(name[len(name) - 4] + "/" + name[len(name) - 3] +
                        #           "/" + name[len(name) - 1]) + " " + str(distace_par) +
                        #       " " + str(statistics.median(av_height)) +
                        #       " " + str(TP2) + " " + str(FP2) + " " + str(FN2) +
                        #       " " + str(statistics.median(bottom_height)) +
                        #       " " + str(TP2 / (TP2 + FP2)) + " " + str(TP2 / (TP2 + FN2)) +
                        #       " " + str(TP1) + " " + str(FP1) + " " + str(FN1) +
                        #       " " + str(statistics.median(bottom_height)) +
                        #       " " + str(TP1 / (TP1 + FP1)) + " " + str(TP1 / (TP1 + FN1)) + " " + "\n")
                        file_to_write.write(str(name[len(name) - 4] +  # Meter
                                                "/" + name[len(name) - 3] +  # Location
                                                "/" + name[len(name) - 1]) +  # Image Name
                                            " " + str(distace_par) +  # Estimated Distance
                                            " " + str(statistics.median(av_height)) +  # Height of Detected Sign
                                            " " + str(statistics.median(bottom_height)) +  # Bottom Row of Detected Sign
                                            " " + str(TP0) +  # TP for First UNET Single Image
                                            " " + str(FP0) +  # FP for First UNET Single Image
                                            " " + str(FN0) +  # FN for First UNET Single Image
                                            " " + str(TP0 / (TP0 + FP0)) +  # Precision
                                            " " + str(TP0 / (TP0 + FN0)) +  # Recall
                                            " " + str(TP3) +  # TP for Second UNET Single Image
                                            " " + str(FP3) +  # FP for Second UNET Single Image
                                            " " + str(FN3) +  # FN for Second UNET Single Image
                                            " " + str(TP3 / (TP3 + FP3)) +  # Precision
                                            " " + str(TP3 / (TP3 + FN3)) +  # Recall
                                            "\n")
                        # Add estimated distance for each particular distance to the array for plotting
                        distances.append(distace_par)
            # Add each distance data to the array for final plotting
            list_of_distances.append(distances)
            recall1.append(TP1 / (TP1 + FN1))  # Recall
            precision1.append(TP1 / (TP1 + FP1))  # Precision
            recall2.append(TP2 / (TP2 + FN2))  # Recall
            precision2.append(TP2 / (TP2 + FP2))  # Precision
    file_to_write.write(str(precision1) + " " + str(precision2) + " " + str(recall1) + " " + str(recall2) + "\n")
    axes = plt.gca()
    axes.plot(precision1)
    axes.plot(precision1, label="Precision UNET#1", color='red')
    axes.plot(precision2)
    axes.plot(precision2, label="Precision UNET#2", color='yellow')
    axes.plot(recall1)
    axes.plot(recall1, label="Recall UNET#1", color='blue')
    axes.plot(recall2)
    axes.plot(recall2, label="Recall UNET#2", color='black')
    plt.legend()
    plt.savefig(str(h0) + "PR" + '.png')
    plt.cla()
    # print(precision1, precision2, recall1, recall2)
    counter = 2
    for items in list_of_distances:
        axes = plt.gca()
        axes.plot(items)
        axes.plot(items, label=str(counter) + " M")
        plt.legend()
        counter += 1
    plt.legend()
    plt.savefig(str(h0) + '.png')
    file_to_write.close()



if __name__ == '__main__':
    # Path to trained models. In case we want to test multiple models we put them under one directory
    model_path = "/home/skeri/tentorch/results/trained_models/ali_best/trained/"
    path_to_models = glob.glob(model_path + "*.pth")
    # Path to image dataset which follows Giovannies pattern (Each folder has ARKIT data as txt)
    database_path = "/home/skeri/Dropbox (ski.org)/Sign_Detection/datasets/Exit Signs by Distance (8bit BMP masks)/"
    # Define the UNET model we need for prediction
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2_unet()
    model.to(device)
    model.eval()
    main(path_to_models, database_path)
