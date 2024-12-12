"""
created on Decembre 10 2024

@author: Mohamed Fakhfakh
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_data(image_folder, mask_endo_folder, mask_epi_folder, target_size=(256, 256)):
    images, masks_endo, masks_epi = [], [], []
    for filename in sorted(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, filename)
        mask_endo_path = os.path.join(mask_endo_folder, filename)
        mask_epi_path = os.path.join(mask_epi_folder, filename)

        if os.path.exists(mask_endo_path) and os.path.exists(mask_epi_path):
            img = load_img(img_path, target_size=target_size, color_mode='grayscale')
            mask_endo = load_img(mask_endo_path, target_size=target_size, color_mode='grayscale')
            mask_epi = load_img(mask_epi_path, target_size=target_size, color_mode='grayscale')

            images.append(img_to_array(img))
            masks_endo.append(img_to_array(mask_endo))
            masks_epi.append(img_to_array(mask_epi))

    return np.array(images), np.array(masks_endo), np.array(masks_epi)