from random import random

from keras.utils.np_utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import splitfolders

"""
KEY: 
** : indicates optimize the method which we use to complete this 
task and take notes for the research paper (Delete markings later)

1. Split the data into training, testing, and validation
2. Load the masks and CBCT Scans
3. Explore the dataset 
4. *Preprocess the data*
5. *Augment the data*
"""

#1. Split the data into training, testing, and validation
input_folder = 'datasets/segmentation_dataset'
output_folder = 'datasets/segmentation_data'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.6,.2,.2), group_prefix=None)

#2. Load the masks and CBCT Scans
train_images = []
train_masks = []

for directory_path in sorted(glob.glob("datasets/segmentation_dataset/train_images/")):
    for img_path in sorted(glob.glob(os.path.join(directory_path, "+.jpg"))):
        print(img_path)
        img = cv2.imread(img_path, 1)
        train_images.append(img)

train_images = np.array(train_images)

for directory_path in sorted(glob.glob("datasets/segmentation_dataset/train_masks/")):
    for mask_path in (sorted(glob.glob(os.path.join(directory_path, "+.jpg"))) or sorted(glob.glob(os.path.join(directory_path, "+.jpeg")))):
      print(mask_path)
      mask = cv2.imread(mask_path)
      train_masks.append(mask)

train_masks = np.array(train_masks)

#3. Explore the data
img_num = random.randint(0, len(train_images) - 1)

img_for_plot = cv2.imread(train_images[img_num], 0)
mask_for_plot = cv2.imread(train_masks[img_num], 0)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_for_plot, cmap='gray')
plt.title('Image')
plt.subplot(122)
plt.imshow(mask_for_plot, cmap='gray')
plt.title('Mask')
plt.show()

print("Unique values in the mask are: ", np.unique(mask_for_plot))

#4. Preprocess the data
train_images = np.expand_dims(normalize(train_images, axis=1), 3)
labelencoder = LabelEncoder()
h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1, 1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded = train_masks_reshaped_encoded.reshape(h, w)

#TODO: More Preprocessing + Augmentation






