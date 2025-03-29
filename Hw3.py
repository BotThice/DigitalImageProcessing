import pywt
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2

# Load the image
# image = iio.imread("./inputPictures/WormHole_1h.tif")
image = iio.imread("./inputPictures/WormHole_2h.tif")

# Convert to grayscale
def toGrayscale(img):
    if len(img.shape) == 3:
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return img

image = toGrayscale(image)

# Wavelet Transform
coeffs2 = pywt.dwt2(image, 'haar')
LL, (LH, HL, HH) = coeffs2

# Construct edge
edgeImg = np.abs(HL) + np.abs(LH) + np.abs(HH)

# Normalize to 0-255 range
edgeImg = (edgeImg / np.max(edgeImg) * 255).astype(np.uint8)

# Create a circular template (20x20 pixels)
template = np.zeros((20, 20), dtype=np.uint8)
cv2.circle(template, (10, 10), 9, 255, 1)  # Draw circle in the center of the template
cv2.circle(template, (10, 10), 6, 0, -1)
file = open ("template.txt", "w")
file.write(str(template))
file.close()
# # Perform template matching with the circular template
result = cv2.matchTemplate(edgeImg, template, cv2.TM_CCOEFF_NORMED)

# # Define a threshold for detection
threshold = 0.2
locations = np.where(result >= threshold)


# # Count detected holes
hole_count = len(locations[0])  # locations[0] contains the row indices where matches occur
print(f"Detected wormholes: {hole_count}")

# # Draw rectangle detected wormholes
for pt in zip(*locations[::-1]):
    cv2.rectangle(edgeImg, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (255, 255, 255), 2)

# Display the result
plt.figure(figsize=(6, 6))
plt.imshow(edgeImg, cmap='gray')
plt.title("Worm Hole Detection by Circular Template")
plt.axis('off')
plt.show()
