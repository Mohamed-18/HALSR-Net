"""
created on Decembre 10 2024

@author: Mohamed Fakhfakh
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_loader import *
from asfsnet_model import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K

# Argument parser for CLI inputs
parser = argparse.ArgumentParser(description='Train and predict with ASFSNet model.')
parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing input images.')
parser.add_argument('--mask_endo_folder', type=str, required=True, help='Path to the folder containing endocardial masks.')
parser.add_argument('--mask_epi_folder', type=str, required=True, help='Path to the folder containing epicardial masks.')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs.')

args = parser.parse_args()

# Paths from arguments
image_folder = args.image_folder
mask_endo_folder = args.mask_endo_folder
mask_epi_folder = args.mask_epi_folder
output_dir = args.output_dir

print(f"Image folder: {image_folder}")
print(f"Endocardial masks folder: {mask_endo_folder}")
print(f"Epicardial masks folder: {mask_epi_folder}")

# Loss functions
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# Data loading and preprocessing
images, masks_endo, masks_epi = load_data(image_folder, mask_endo_folder, mask_epi_folder)

# Normalize images and masks
images = images / 255.0
masks_endo = masks_endo / 255.0
masks_epi = masks_epi / 255.0

# Create background mask
mask_background = np.logical_not(masks_endo.astype(bool) | masks_epi.astype(bool)).astype(int)
masks = np.concatenate([masks_endo, masks_epi, mask_background], axis=-1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.25, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Model initialization and compilation
model = ASFSNet_model(input_size=(256, 256, 1), num_classes=3)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=bce_dice_loss, metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
]

# Training
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=32, epochs=300, callbacks=callbacks)


# Evaluation
val_loss, val_accuracy = model.evaluate(X_test, y_test)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
