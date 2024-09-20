import streamlit as st
import pandas as pd
import numpy as np
import cv2
from region_proposals import iou_filter
import matplotlib.pyplot as plt
from PIL import Image

# Set the paths for the data and images
IMAGE_PATH = 'DATA/agri_data/'
LABEL_PATH = 'DATA/agri_label.csv'

# Load the data
df = pd.read_csv(LABEL_PATH)

# Function to display the image with bounding boxes
def display_image_with_bboxes(image_name, iou_threshold=0.5):
    # Full image path
    image_path = IMAGE_PATH + image_name

    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get filtered regions and negative examples
    filtered_regions, _ = iou_filter(image_path, df, thresh=iou_threshold)

    # Draw bounding boxes on the image
    for region in filtered_regions:
        bbox, label = region
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return img

# Streamlit app layout
st.title('Weed and Crop Detection')
st.write("Select an image and visualize the region proposals and bounding boxes.")

# Select an image from the dataset
image_list = df['filename'].unique()
selected_image = st.selectbox('Choose an image', image_list)

# Input for IoU threshold
iou_threshold = st.slider('Select IoU Threshold', 0.1, 1.0, 0.5)

# Display the selected image with bounding boxes
if st.button('Show Bounding Boxes'):
    with st.spinner('Processing...'):
        img_with_boxes = display_image_with_bboxes(selected_image, iou_threshold)
        
        # Convert OpenCV image to PIL Image for displaying in Streamlit
        img_with_boxes = Image.fromarray(img_with_boxes)
        
        st.image(img_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)
