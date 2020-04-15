import cv2
import numpy as np
import os
import glob


database_path = "/home/skeri/Saved_Data/Urolled_Dataset/Exit Signs by Distance/"
new_database_path = "/home/skeri/Saved_Data/Unrolled_dataset_BMP/Exit Signs by Distance/"
for meters in range(2, 9):
    # Get images within image dataset
    image_directory = database_path + str(meters)
    bmp_image_directory = new_database_path + str(meters)
    images_folders = os.listdir(image_directory)

    for folders in images_folders:
        # Get list of images within img folder. We don't get the mask folder here
        list_of_images = glob.glob(image_directory + "/" + folders + "/imgs/*.png")

        save_bmp = bmp_image_directory + "/" + folders
        for images in list_of_images:
            s1 = images.split('/')
            s2 = s1[len(s1) - 1].split('.')
            mask_path = ''
            new_path = ''
            for i in range(0, len(s1) - 2):
                mask_path += s1[i] + "/"
            mask_path += "masks/" + s2[0] + ".png"
            new_path = save_bmp + "/masks/" + s2[0] + ".bmp"
            new_image_path = save_bmp + "/imgs/" + s2[0] + ".png"
            M = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            M = M[:, :, 1]
            M = M.astype(np.uint8)
            cv2.imwrite(new_path, M)
            cv2.imwrite(new_image_path, cv2.imread(images))