# ================================
# Predictive Silicon Intelligence
# Wafer Defect Detection Demo
# ================================

# Install required library
!pip install opencv-python

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

print("Upload a wafer image (JPEG/PNG)")
uploaded = files.upload()

# Load image
image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)

# Convert to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Adaptive thresholding for better defect isolation
thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

# Morphological operations to clean noise
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Detect contours (potential defects)
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on copy of image
output = image_rgb.copy()
cv2.drawContours(output, contours, -1, (0,255,0), 2)

# Count defect areas based on contour size threshold
defect_count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 50:  # Filter tiny noise
        defect_count += 1

# Display results
plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
plt.title("Original Wafer")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Thresholded Image")
plt.imshow(opening, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Detected Defects")
plt.imshow(output)
plt.axis("off")

plt.show()

print("Total Contours Detected:", len(contours))
print("Filtered Defect Regions (Area > 50):", defect_count)
