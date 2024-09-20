# Crop_Weed_detection

# Overview
The Crop-Weed Detection project aims to accurately identify and differentiate between sesame crops and weeds using a Region-Based Convolutional Neural Network (RCNN). The project employs selective search for region proposals and Intersection over Union (IoU) filtering to refine object detection. A user-friendly Streamlit interface is provided for uploading images, applying the model, and visualizing the results with bounding boxes around detected crops and weeds.

# Features
Image Upload: Allows users to upload images for analysis.
Preprocessing: Processes images to prepare them for detection.
Region Proposals: Uses selective search to generate potential areas of interest in images.
RCNN Model: Detects crops and weeds in the proposed regions.
IoU Filtering: Refines detection results by filtering based on Intersection over Union.
Visualization: Displays images with bounding boxes around detected objects.

# Installation
 1.Prerequisites
 
2.Python

3.Required libraries

4.TensorFlow/Keras (for RCNN)

5.OpenCV

6.Streamlit

7.NumPy

8.pandas
