import cv2
import numpy as np

def load_img(impath):
    """
    Loads an image from a specified location and returns it in RGB format.
    Input:
    - impath: a string specifying the target image location.
    Returns an RGB integer image.
    """
    img_loaded = cv2.imread(f"images/{impath}")
    # img = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB) 
    
    return img_loaded

image_name = "p2_solved"
img = load_img(f"{image_name}.png")

height, width, _ = img.shape
for row in range(height):
    for col in range(width):
        rgb_tuple = tuple(img[row, col])
        distance = np.linalg.norm(np.array(rgb_tuple))
        if distance < 130:
            img[row, col] = [255, 255, 255]

cv2.imwrite(f"processed_images/{image_name}_processed.png", img)
        