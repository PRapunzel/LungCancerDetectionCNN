# Import needed libraries
import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
from skimage.util import random_noise
from scipy.ndimage import median_filter
import warnings

sns.set_style('darkgrid')
warnings.filterwarnings('ignore')


# Read the training dataset into the dataframe
def loading_the_data(data_dir):
    # Generate data paths with labels
    filepaths = []
    labels = []

    # Get folder names
    folds = os.listdir(data_dir)

    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)

    # Concatenate data paths with labels into one DataFrame
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df


# Change label names to their original names
def change_label_names(df, column_name):
    index = {'lung_aca': 'Lung_adenocarcinoma', 'lung_n': 'Lung_benign_tissue',
             'lung_scc': 'Lung squamous_cell_carcinoma'}
    df[column_name] = df[column_name].replace(index)


# Preprocess images: Add Gaussian noise and apply a median filter
def preprocess_images(input_df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocessed_filepaths = []
    preprocessed_labels = []

    for idx, row in input_df.iterrows():
        img_path = row['filepaths']
        label = row['labels']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Add Gaussian noise
        noisy_img = random_noise(img, mode='gaussian', var=0.5)
        noisy_img = np.array(255 * noisy_img, dtype=np.uint8)

        # Apply median filter
        denoised_img = median_filter(noisy_img, size=3)

        # Save the preprocessed image
        new_img_path = os.path.join(output_dir, f"preprocessed_{os.path.basename(img_path)}")
        cv2.imwrite(new_img_path, cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR))

        preprocessed_filepaths.append(new_img_path)
        preprocessed_labels.append(label)

    preprocessed_df = pd.DataFrame({'filepaths': preprocessed_filepaths, 'labels': preprocessed_labels})
    return preprocessed_df


# Load the data
data_dir = 'lung_colon_image_set/lung_image_sets'
df = loading_the_data(data_dir)
change_label_names(df, 'labels')
print(df)

# Preprocess the images
preprocessed_data_dir = 'preprocessed_lung_image_set'
df_preprocessed = preprocess_images(df, preprocessed_data_dir)
print(df_preprocessed)

# df_preprocessed = 'preprocessed_lung_image_set'

# Check if the training data is balanced
data_balance = df_preprocessed.labels.value_counts()
