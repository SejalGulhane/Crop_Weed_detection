import streamlit as st
import pandas as pd
import numpy as np
import cv2
from iou_calculation import iou_calc  # Ensure this is correctly imported

# Set up Selective Search
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Define your iou_filter function here (as you've written it)
def iou_filter(image_path, true_bb, thresh=0.5):
    img_name = image_path.split('/')[-1]
    img_bb = true_bb[true_bb['filename'] == img_name].reset_index(drop=True)
    img = cv2.imread(image_path)
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    ss_bb = rects[:2000]
    filtered_selective_search = []
    negative_examples = []
    maybe_negative = []

    for label in range(len(img_bb)):
        true_xmin, true_ymin, true_width, true_height = img_bb.loc[label, 'xmin'], img_bb.loc[label, 'ymin'], img_bb.loc[label, 'xmax'] - img_bb.loc[label, 'xmin'], img_bb.loc[label, 'ymax'] - img_bb.loc[label, 'ymin']
        class_of_label = img_bb.loc[label, 'class']

        for j, rect in enumerate(ss_bb):
            calculating_iou_for_selectivesearch = iou_calc([true_xmin, true_ymin, true_width, true_height], rect)

            if calculating_iou_for_selectivesearch > thresh:
                filtered_selective_search.append([list(rect), class_of_label])

            elif calculating_iou_for_selectivesearch < 0.2:
                maybe_negative.append(list(rect))

    def Remove(duplicate):
        final_list = []
        for num in duplicate:
            if num not in final_list:
                final_list.append(num)
        return final_list

    maybe_negative = Remove(maybe_negative)
    filtered_selective_search = Remove(filtered_selective_search)

    only_labels_of_filtered_selective_search = [x[0] for x in filtered_selective_search]

    for lab in maybe_negative:
        condition = []
        for true_lab in only_labels_of_filtered_selective_search:
            iou_for_negative_ex = iou_calc(true_lab, lab)
            condition.append(True) if iou_for_negative_ex <= 0.2 else condition.append(False)

        if False not in condition:
            negative_examples.append(lab)

    negative_examples = Remove(negative_examples)
    random_background_images_index = np.random.randint(low=0, high=len(negative_examples), size=2 * len(only_labels_of_filtered_selective_search))
    random_background_images = [negative_examples[x] for x in random_background_images_index]

    return filtered_selective_search, Remove(random_background_images)

# Streamlit UI
st.title("Weed and Crop Detection App")

# File uploader to upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# File uploader to upload the CSV file with bounding box data
csv_file = st.file_uploader("Upload a CSV file with bounding box data", type=["csv"])

if uploaded_file is not None and csv_file is not None:
    # Load the image
    image_bytes = uploaded_file.read()
    img = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_column_width=True)

        # Load the CSV file with bounding boxes
        true_bb = pd.read_csv(csv_file)

        # Save the uploaded image temporarily to run the iou_filter function
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)

        # Run the iou_filter function
        filtered_selective_search, negative_examples = iou_filter(temp_image_path, true_bb)

        # Draw bounding boxes on the image
        for rect, label in filtered_selective_search:
            x, y, w, h = rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            st.text(f"Detected {label} at [{x}, {y}, {x + w}, {y + h}]")

        # Display the processed image
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(processed_img, caption='Processed Image with Bounding Boxes', use_column_width=True)

    else:
        st.error("Error: Image could not be processed.")
else:
    st.info("Please upload both an image and a CSV file with bounding box data.")
