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
from math import sin, cos, tan, atan


def get_data_loaders(val_files):
    val_transform = Compose([Resize((224, 224)), ToTensor(), ])
    val_loader = DataLoader(MaskDataset(val_files, val_transform), batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=0)
    return val_loader


def Gio_precision_recall(M, P):
    M = 1.0 * (M > 127)
    P = 1.0 * (P > 127)
    TP = len(np.where(P * M > 0)[0])  # AND between prediction (P) and ground truth (M)
    FP = len(np.where(cv2.subtract(P, M) > 0)[0])  # white pixels in P that don't have corresponding white pixels in M
    FN = len(np.where(cv2.subtract(M, P) > 0)[0])  # white pixels in M that don't have correspondence in P
    return TP, FP, FN


def calc_range3(focal, h0, vlo, dv, gamma):  # this works!
    Z = h0 * (focal * cos(gamma) - vlo * sin(gamma)) / dv
    return Z


def calc_range2(h0, gamma, alpha, delta):
    return h0 / (tan(gamma + alpha + delta) - tan(gamma + alpha))


def angle_from_center(f, p):  # pixel p is 0 at center of 1D image
    return atan(p / f)


def distanceMeasuring(apparent_height, bottom, gamma):
    # gamma = gamma * pi / 180
    h0 = .20  # meters
    focal_length = 1602
    # focal_length = (112 / 1440) * focal_length
    # h, w = 149.34, 112
    height, width = 1920, 1440
    alpha = angle_from_center(focal_length, height / 2 - bottom)
    delta = angle_from_center(
        focal_length, height / 2 - (bottom - apparent_height)) - angle_from_center(focal_length, height / 2 - bottom)
    # Z_Cal3 = calc_range3(focal_length, h0, h / 2 - bottom_row, apparent_height, gamma)
    Z_Cal2 = calc_range2(h0, gamma, alpha, delta)
    return Z_Cal2


def UNET_segmentation(image, output_size):
    with torch.no_grad():
        inputs = image.to(device)
        outputs = model(inputs)
        output = outputs.cpu().numpy().reshape(112, 112) * 255
        output = cv2.resize(output.astype(np.uint8), (output_size, output_size))
        output[output >= 128] = 255
        output[output < 128] = 0
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


def plot_distances(list_to_plot, selected_model):
    counter = 2
    for items in list_to_plot:
        print("Distance of ", counter, " for loading model ", selected_model)
        if len(items) > 2:
            print("Standard Deviation of sample is % s " % (statistics.stdev(items)))
            print("Median of sample is % s " % (statistics.median(items)))
            print("Mean of sample is % s " % (statistics.mean(items)))
            print("Max of sample is % s " % (np.amax(items)))
            print("Min of sample is % s " % (np.amin(items)))
        axes = plt.gca()
        axes.set_xlim([0, 160])
        axes.set_ylim([1, 9])
        axes.plot(items, label=str(counter) + " M")
        counter += 1
    plt.legend()
    # plt.savefig('plot1.png')
    # plt.cla()
    plt.show()


def main(list_of_models, dataset_path):
    for loading_model in list_of_models:
        # List of distances contains estimated distances for various distance categories
        list_of_distances = []
        # We want to estimate distances for images captured 2 to 8 meters away from the camera
        for meters in range(2, 8):
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
                f = open(ARKIT_directory, "r")
                coreml_data = []
                for x in f:
                    coreml_data.append(x)
                # Get list of images within img folder. We don't get the mask folder here
                list_of_images = glob.glob(image_directory + "/" + folders + "/imgs/*.jpg")
                # Define the UNET model we need for prediction
                model.load_state_dict(torch.load(loading_model))
                # Start to read each image within the list of images we already have
                for images in list_of_images:
                    # Get Pitch, Roll, and Yaw for the picture we want to predict and then unroll it.
                    # For unrolling we also execute rotation; hence we do both rotation and unrolling.
                    # To see the result uncomment imageShow
                    roll, pitch, yaw = roll_pitch_yaw(images, coreml_data)
                    unrolled_image = ndimage.rotate(cv2.imread(images), (roll * 180) / math.pi)

                    # imageShow(unrolled_image)

                    # Crop top left corner of the unrolled image
                    cropped_from_original = unrolled_image[0:1440, 0:1440]
                    # Prepare cropped unrolled image for prediction. This step includes converting 'numpy.ndarray' image
                    # to 'torch.utils.data.dataloader.DataLoader'
                    to_predict = next(iter(get_data_loaders(cropped_from_original)))
                    # Predict the data and then resize the output to 1440x1440 image size
                    output = UNET_segmentation(to_predict, 1440)
                    # It labels different detected components of the predicted image. Here we try to only consider the
                    # biggest connected components and remove small ones. It helps us to only consider closest sign.
                    # To see the result uncomment imageShow
                    output = find_connected_components(output, 1440)

                    # imageShow(output)

                    # Next step to Zoom into detected area. To do that we find the center of biggest detected connected
                    # component and then use it to crop 448x448 from the unrolled rotated cropped image we already have
                    # This step help the algorithm to have more precise segmentation
                    # Previous steps could be replace with any detection algorithm.
                    # At the moment (3/31/2020) we use MobileNetV2 then UNET then MobileNetV2 then UNET.
                    # We can remove the first UNET and maybe the second MobileNetV2.
                    # To see the detected image uncomment the imageShow

                    if np.where(output > 0)[0].size == 0:
                        print(images)
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

                        # imageShow(detected_image)

                        # Second round of UNET segmentation. It tries to segment cropped 448 x 448 images from last step
                        to_predict = next(iter(get_data_loaders(detected_image)))
                        output = UNET_segmentation(to_predict, 448)
                        # Labels different detected components of the predicted image. Here we try to only consider the
                        # biggest connected components and remove small ones. It helps us to only consider closest sign.
                        output = find_connected_components(output, 448)
                        # The following section find the mean or median height of the predicted component
                        nz = np.where(output != 0)
                        if len(nz[0]) != 0 and len(nz[1] != 0):
                            height = []
                            for iteration in range(np.amin(nz[1]), np.amax(nz[1])):
                                height.append(np.where(output[:, iteration] != 0))

                            av_height = []
                            for iterations in height:
                                if (len(iterations[0])) != 0:
                                    mx = np.amax(iterations)
                                    mn = np.amin(iterations)
                                    av_height.append(mx - mn)
                            # Estimate the distance based on James Algorithm
                            distace_par = distanceMeasuring(statistics.median(av_height), maxY, pitch)
                            # Add estimated distance for each particular distance to the array for plotting
                            distances.append(distace_par)
            # Add each distance data to the array for final plotting
            list_of_distances.append(distances)
        plot_distances(list_of_distances, loading_model)


if __name__ == '__main__':
    # Path to trained models. In case we want to test multiple models we put them under one directory
    model_path = "/home/skeri/tentorch/results/trained_models/ali_best/trained/"
    path_to_models = glob.glob(model_path + "*.pth")
    # Path to image dataset which follows Giovannies pattern (Each folder has ARKIT data as txt)
    database_path = "/home/skeri/Gio_orig_dataset/Exit Signs by Distance/"
    # Define the UNET model we need for prediction
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2_unet()
    model.to(device)
    model.eval()
    main(path_to_models, database_path)
