# Deep Learning-Based Image Segmentation

## HALSR-Net: Improving CNN Segmentation of Cardiac Left Ventricle MRI with Hybrid Attention and Latent Space Reconstruction

Accurate cardiac MRI segmentation is vital for detailed cardiac analysis, yet the manual process is labor-intensive and prone to variability. Despite advancements in MRI technology, there remains a significant need for automated methods that can reliably and efficiently segment cardiac structures. This repository introduces **ASFS-Net**, a novel multi-level segmentation architecture designed to improve the accuracy and reproducibility of cardiac segmentation from Cine-MRI acquisitions, focusing on the left ventricle (LV).

### Key Features

1. **Region of Interest Extraction**: Uses a regression model to predict the bounding box around the LV for precise localization.
2. **Semantic Segmentation**: Employs the ASFS-Net architecture incorporating:
   - **Hybrid Attention Pooling Module (HAPM)**: Combines attention and pooling mechanisms to enhance feature extraction and contextual information.
   - **Reconstruction Module**: Utilizes latent space features to refine segmentation accuracy.

### Performance
Experiments conducted on an in-house clinical dataset and two public datasets (**ACDC** and **LVQuant19**) demonstrate that ASFS-Net outperforms state-of-the-art architectures, achieving up to **98% accuracy** and **F1-score** for segmenting the LV cavity and myocardium. The proposed approach effectively addresses existing limitations, offering a robust solution for cardiac MRI segmentation and improving cardiac function analysis and patient care.

## Project Structure
- **data_loader.py**: Contains functions for loading and preprocessing the dataset.
- **model_utils.py**: Utility functions for attention mechanisms and ASPP.
- **model.py**: Defines the U-Net architecture with ASFS-Net extensions.
- **train.py**: Training script for the segmentation model.

## Requirements
- TensorFlow
- NumPy
- scikit-learn
- Matplotlib

## Usage
1. Place your images and masks in separate directories.
2. Update the paths in `train.py`.
3. Run `python train.py` to start training.

## Results
HALSR-Net achieves state-of-the-art performance with the integration of advanced attention mechanisms, ASPP, and reconstruction modules.

## Acknowledgments
This project was inspired by advanced segmentation frameworks in medical imaging and aims to improve automation in cardiac analysis workflows.
