# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import cv2
# import os, shutil
# import matplotlib.image as mpiimg
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
# import imutils
# import time
# # %matplotlib inline
# plt.style.use('ggplot')

# ## Data set
# import zipfile
# if os.path.exists('archive.zip'):
#     z = zipfile.ZipFile('archive.zip', 'r')
#     z.extractall()

# # renaming the image folders
# folder = "C:\\Users\\kanth\\Brain Tumor\\brain_tumor_dataset//yes/"
# count = 1
# for i in os.listdir(folder):
#     src = folder + i
#     dst = folder + "Y_" + str(count) + ".jpg"
#     if os.path.exists(dst) or os.path.abspath(src) == os.path.abspath(dst):
#         count += 1
#         continue
#     os.rename(src, dst)
#     count += 1

# folder = "C:\\Users\\kanth\\Brain Tumor\\brain_tumor_dataset//no/"
# count = 1
# for i in os.listdir(folder):
#     src = folder + i
#     dst = folder + "N_" + str(count) + ".jpg"
#     if os.path.exists(dst) or os.path.abspath(src) == os.path.abspath(dst):
#         count += 1
#         continue
#     os.rename(src, dst)
#     count += 1


# #Exploaratory Data Analysis
# listyes = os.listdir("C:\\Users\\kanth\\Brain Tumor\\brain_tumor_dataset//yes/")
# num_yes = len(listyes)
# print("Number of images with tumor: ", num_yes)

# listno = os.listdir("C:\\Users\\kanth\\Brain Tumor\\brain_tumor_dataset//no/")
# num_no = len(listno)
# print("Number of images without tumor: ", num_no)

# #plot
# data = {'tumorous' : num_yes, 'non-tumorous': num_no}
# typex = data.keys()
# values = data.values()
# fig = plt.figure(figsize=(5,7))
# plt.bar(typex, values, color="red")
# plt.xlabel('Data')
# plt.ylabel("No Of Brain Tumor Images")
# plt.title("Count of Brain tumor")
# # plt.show()  # Commented out to prevent blocking

# #Data Augmentation
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import  Flatten, Dense, Dropout
# from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.optimizers import Adam,SGD
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.models import load_model

# def timing (sec_elapsed):
#     h = int(sec_elapsed / (60 * 60))
#     m = int((sec_elapsed % (60 * 60)) / 60)
#     s= sec_elapsed % 60
#     if h == 0 and m == 0:
#         return f'{s} seconds'
#     elif h == 0:
#         return f'{m}:{s}'
#     else:
#            return f'{h}:{m}:{s}'
    
# def augmented_data(file_dir, n_generated, save_to_dir):
#     data_gen = ImageDataGenerator(
#         rotation_range=10, 
#         width_shift_range=0.1, 
#         height_shift_range=0.1, 
#         shear_range=0.15, 
#         brightness_range=(0.3, 1.0), 
#         horizontal_flip=True, 
#         vertical_flip=True, 
#         fill_mode='nearest'
#     )
    
#     if not os.path.exists(save_to_dir):
#         os.makedirs(save_to_dir)
        
#     for filename in os.listdir(file_dir):
#         image = cv2.imread(os.path.join(file_dir, filename))
#         if image is not None:
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             image = image.reshape((1,) + image.shape)
#             save_prefix = 'aug_' + filename[:-4]
#             i = 0
#             for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir, save_prefix=save_prefix, save_format='jpg'):
#                 i += 1
#                 if i >= n_generated:
#                     break

# start_time = time.time()

# yes_path = 'brain_tumor_dataset/yes'
# no_path = 'brain_tumor_dataset/no'
# augmented_data_path = 'augmented_data'

# # Remove previously generated augmented data
# if os.path.exists(augmented_data_path):
#     shutil.rmtree(augmented_data_path)

# # Create new folders
# os.makedirs(augmented_data_path + '/yes', exist_ok=True)
# os.makedirs(augmented_data_path + '/no', exist_ok=True)

# # Augment majority class
# augmented_data(
#     file_dir=yes_path,
#     n_generated=6,
#     save_to_dir=augmented_data_path + '/yes'
# )

# # Augment minority class
# augmented_data(
#     file_dir=no_path,
#     n_generated=8,
#     save_to_dir=augmented_data_path + '/no'
# )

# end_time = time.time()
# execution_time = end_time - start_time

# print(f"Elapsed time: {timing(execution_time)}")


# def data_summary(main_path):
#     yes_path = os.path.join(main_path, 'yes')
#     no_path = os.path.join(main_path, 'no')

#     m_pos = len(os.listdir(yes_path))
#     m_neg = len(os.listdir(no_path))
#     m = m_pos + m_neg
    
#     pos_per = (m_pos * 100) / m
#     neg_per = (m_neg * 100) / m

#     print(f"Number of examples: {m}")
#     print(f"Percentage of positive examples: {pos_per:.2f}%")
#     print(f"Percentage of negative examples: {neg_per:.2f}%")

# # Call the function to see the summary
# data_summary('augmented_data')

# #Data Preprocessing
# def crop_brain_contour(image, plot=False):
#     # Convert the image to grayscale and blur it
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Threshold the image, then perform erosions + dilations to remove noise
#     thres = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
#     thres = cv2.erode(thres, None, iterations=2)
#     thres = cv2.dilate(thres, None, iterations=2)

#     # Find contours and grab the largest one
#     cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     c = max(cnts, key=cv2.contourArea)

#     # Find the extreme points
#     extLeft = tuple(c[c[:, :, 0].argmin()][0])
#     extRight = tuple(c[c[:, :, 0].argmax()][0])
#     extTop = tuple(c[c[:, :, 1].argmin()][0])
#     extBot = tuple(c[c[:, :, 1].argmax()][0])

#     # Crop the image
#     new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

#     if plot:
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.title('Original')
#         plt.subplot(1, 2, 2)
#         plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
#         plt.title('Cropped')
#         # plt.show()  # Commented out to prevent blocking

#     return new_image

# # Test the cropping function with an existing image
# # Using the file you provided/selected
# img_path = 'brain_tumor_dataset/no/N_101.jpg'
# img = cv2.imread(img_path)

# if img is not None:
#     print(f"Successfully loaded {img_path}")
#     cropped_img = crop_brain_contour(img, plot=True)
# else:
#     # Fallback to check directory if the specific name fails
#     files = os.listdir('brain_tumor_dataset/no/')
#     if files:
#         img_path = os.path.join('brain_tumor_dataset/no/', files[0])
#         img = cv2.imread(img_path)
#         crop_brain_contour(img, plot=True)
#     else:
#         print("No images found in the directory.")


# # Updated image loading function
# def load_data(dir_list, image_size):
#     X = []
#     y = []
#     image_width, image_height = image_size
    
#     for directory in dir_list:
#         if not os.path.exists(directory):
#             print(f'Directory not found: {directory}')
#             continue
#         for filename in os.listdir(directory):
#             image_path = os.path.join(directory, filename)
#             image = cv2.imread(image_path)
#             if image is None:
#                 continue
            
#             # Corrected function name from crop_brain_tumor to crop_brain_contour
#             image = crop_brain_contour(image, plot=False)
#             image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
#             image = image / 255.0
            
#             X.append(image)
#             # Check if 'yes' is in the path to assign label 1, else 0
#             if 'yes' in directory.lower():
#                 y.append(1)
#             else:
#                 y.append(0)
#         print(f'{directory} loaded.')

#     X = np.array(X)
#     y = np.array(y)
#     X, y = shuffle(X, y, random_state=42)
    
#     return X, y


# # Data splitting and processing
# base_dir = 'tumorous_and_non_tumorous'
# if not os.path.isdir(base_dir):
#     os.makedirs(base_dir)
#     print(f'Directory {base_dir} created.')

# # Create subdirectories for splitting the data
# train_dir = os.path.join(base_dir, 'train')
# test_dir = os.path.join(base_dir, 'test')
# valid_dir = os.path.join(base_dir, 'valid')

# for directory in [train_dir, test_dir, valid_dir]:
#     if not os.path.isdir(directory):
#         os.makedirs(directory, exist_ok=True)
#         print(f'Created: {directory}')
#     # Create tumorous and non_tumorous subdirectories
#     for label in ['tumorous', 'non_tumorous']:
#         label_dir = os.path.join(directory, label)
#         if not os.path.isdir(label_dir):
#             os.makedirs(label_dir, exist_ok=True)
#             print(f'  Created: {label_dir}')

# print("Subdirectories created successfully.")
# print(f"Train dir structure: {os.listdir(train_dir) if os.path.isdir(train_dir) else 'Not found'}")

# # Load and process data
# IMG_SIZE = (240, 240)
# print("Loading and processing data...")
# X, y = load_data(['augmented_data/yes', 'augmented_data/no'], IMG_SIZE)

# # Split
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# def save_images(X_set, y_set, base_folder):
#     for i in range(len(X_set)):
#         label = 'tumorous' if y_set[i] == 1 else 'non_tumorous'
#         folder = os.path.join(base_folder, label)
#         os.makedirs(folder, exist_ok=True)
#         img = (X_set[i] * 255).astype(np.uint8)
#         cv2.imwrite(os.path.join(folder, f"{label}_{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# print("Saving images to local folders...")
# print(f"Train directory: {train_dir}")
# print(f"Valid directory: {valid_dir}")
# print(f"Test directory: {test_dir}")
# save_images(X_train, y_train, train_dir)
# print(f"  Saved {len(X_train)} training images")
# save_images(X_val, y_val, valid_dir)
# print(f"  Saved {len(X_val)} validation images")
# save_images(X_test, y_test, test_dir)
# print(f"  Saved {len(X_test)} test images")

# # Final verification
# print("\nFinal folder structure:")
# for split in ['train', 'test', 'valid']:
#     split_path = os.path.join(base_dir, split)
#     if os.path.isdir(split_path):
#         print(f"{split}: {os.listdir(split_path)}")
#         for label in ['tumorous', 'non_tumorous']:
#             label_path = os.path.join(split_path, label)
#             if os.path.isdir(label_path):
#                 count = len(os.listdir(label_path))
#                 print(f"  {label}: {count} images")

# print("Done! Check the folder sidebar now.")





"""
Delete specific duplicate image files by exact filename.

This script ONLY deletes files whose names exactly match the list below.
Every other file in the folder is left untouched.

USAGE:
1. Set FOLDER_PATH to the folder containing your images.
2. Check/edit FILES_TO_DELETE if needed.
3. Run: python delete_duplicates.py
   - By default it runs in DRY-RUN mode (just prints what it would delete).
   - Set DRY_RUN = False once you're happy with the list, then run again to actually delete.
"""

import os
import re

# ----------------------------------------------------------------------
# 1. Set this to the folder where your images are stored
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(BASE_DIR, "brain_tumor_dataset", "no")

# ----------------------------------------------------------------------
# 2. Pattern for files to delete: names like "2 no.jpg", "no 1.jpg", "N1.JPG"
# ----------------------------------------------------------------------
FILE_PATTERN = re.compile(r"^(?:\d+\s+no|no\s+\d+|n\d+)(\.(jpg|jpeg|png))?$", re.IGNORECASE)

# ----------------------------------------------------------------------
# 3. Safety switch — keep True until you've verified the printed list
# ----------------------------------------------------------------------
DRY_RUN = True


def delete_duplicate_files(folder_path, pattern, dry_run=True):
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    deleted = []

    for name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, name)
        if not os.path.isfile(file_path):
            continue
        if pattern.match(name):
            if dry_run:
                print(f"[DRY RUN] Would delete: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
                    continue
            deleted.append(name)

    print("\n--- Summary ---")
    print(f"Matched files: {len(deleted)}")

    if dry_run:
        print("\nThis was a DRY RUN. No files were deleted.")
        print("Set DRY_RUN = False in the script to actually delete these files.")


if __name__ == "__main__":
    delete_duplicate_files(FOLDER_PATH, FILE_PATTERN, dry_run=DRY_RUN)


from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load trained model
model = load_model("model.h5")

print("Model loaded successfully")

# Prediction on test set
predictions = model.predict(test_generator)

# Convert probabilities to class labels
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(test_generator.class_indices.keys())
))