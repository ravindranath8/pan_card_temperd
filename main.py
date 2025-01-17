import streamlit as st
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


# Function to highlight differences between two images
def highlight_difference(original_image_cv, tampered_image_cv):
    # Compute absolute difference between the images
    diff = cv2.absdiff(original_image_cv, tampered_image_cv)
    
    # Convert difference image to grayscale to highlight areas of change
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to find significant differences (tampered areas)
    _, thresholded = cv2.threshold(diff_gray, 50, 255, cv2.THRESH_BINARY)
    
    # Highlight tampered areas in green
    tampered_image = tampered_image_cv.copy()
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small differences
            cv2.drawContours(tampered_image, [contour], -1, (0, 255, 0), 3)  # Green contours

    return tampered_image, contours, diff_gray

# Function to load model and detect tampering (if needed)
def detect_tampering(original_image: Image, tampered_image: Image):
    # Convert images to OpenCV format
    original_image_cv = np.array(original_image)
    original_image_cv = cv2.cvtColor(original_image_cv, cv2.COLOR_RGB2BGR)
    
    tampered_image_cv = np.array(tampered_image)
    tampered_image_cv = cv2.cvtColor(tampered_image_cv, cv2.COLOR_RGB2BGR)

    # Highlight differences (tampering) between the original and tampered images
    tampered_image_with_highlight, contours, diff_gray = highlight_difference(original_image_cv, tampered_image_cv)
    
    # Check if any significant tampering is detected
    tampering_detected = len(contours) > 0
    
    # Calculate percentage of tampered areas
    total_pixels = diff_gray.size  # Total pixels in the image
    differing_pixels = np.count_nonzero(diff_gray)  # Non-zero pixels indicate differences
    tampering_percentage = (differing_pixels / total_pixels) * 100  # Percentage of tampered areas

    # Calculate SSIM (Structural Similarity Index) between the original and tampered images
    original_gray = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered_image_cv, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(original_gray, tampered_gray)

    # Calculate the percentage match based on SSIM
    ssim_percentage = ssim_value * 100

    return tampering_detected, tampered_image_with_highlight, contours, tampering_percentage, ssim_percentage

# Streamlit UI Components
st.title('PAN Card Tampering Detection - Compare Original and Tampered Images')

# File uploader for original and tampered PAN card images
original_file = st.file_uploader("Upload Original PAN Card Image", type=["jpg", "jpeg", "png"])
tampered_file = st.file_uploader("Upload Tampered PAN Card Image", type=["jpg", "jpeg", "png"])

if original_file is not None and tampered_file is not None:
    # Load the images
    original_image = Image.open(original_file)
    tampered_image = Image.open(tampered_file)
    
    # Display the uploaded images
    st.image(original_image, caption="Original PAN Card Image", use_column_width=True)
    st.image(tampered_image, caption="Tampered PAN Card Image", use_column_width=True)
    
    # Process the images for tampering detection
    with st.spinner("Analyzing for tampering..."):
        tampering_detected, tampered_image_with_highlight, contours, tampering_percentage, ssim_percentage = detect_tampering(original_image, tampered_image)
    
    # Display Results
    if tampering_detected:
        st.subheader("Tampering Detected!")
        st.write(f"The tampered PAN card image has differences from the original image.")
    else:
        st.subheader("No Tampering Detected")
        st.write("No significant tampering was detected between the images.")
    
    # Display the tampered image with highlighted differences
    st.image(tampered_image_with_highlight, caption="Tampered Image with Highlighted Differences", use_column_width=True)
    
    # Display the detected tampered areas (if any)
    if tampering_detected:
        st.write(f"Found {len(contours)} tampered areas!")
        for idx, contour in enumerate(contours):
            st.write(f"Tampered Area {idx+1}: Contour with area {cv2.contourArea(contour)} pixels")
    
    # Display tampering percentage
    st.write(f"Tampering Percentage: {tampering_percentage:.2f}%")
    
    # Display SSIM percentage (how much the images match)
    st.write(f"Percentage Match between Original and Tampered Image (SSIM): {ssim_percentage:.2f}%")
    
    # Button to try again with new images
    st.button("Try Again")
