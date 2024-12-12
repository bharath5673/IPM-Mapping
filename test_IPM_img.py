import cv2
import numpy as np
import matplotlib.pyplot as plt

# # Load the image (Replace 'road_image.jpg' with your image file)
image = cv2.imread('test1.png')


def IPM(image):
    # Dimensions of the image
    height, width = image.shape[:2]

    param = 400
    # First perspective transform
    original_points1 = np.float32([
        [0, height // 2],          # Top-left of the lower half
        [width, height // 2],      # Top-right of the lower half
        [width, height],           # Bottom-right corner
        [0, height],               # Bottom-left corner
    ])

    destination_points1 = np.float32([
        [0, 0],                   # Top-left corner
        [width, 0],               # Top-right corner
        [width - param, height],    # Bottom-right corner
        [param, height],            # Bottom-left corner
    ])

    # Compute and apply the first transformation
    matrix1 = cv2.getPerspectiveTransform(original_points1, destination_points1)
    warped_image1 = cv2.warpPerspective(image, matrix1, (width, height))



    # Draw lines on the original image to visualize the selected points for better understanding
    for i in range(len(original_points1)):
    	start_point = tuple(map(int, original_points1[i]))  # Convert to integers
    	end_point = tuple(map(int, original_points1[(i + 1) % len(original_points1)]))  # Wrap around to the first point
    	cv2.line(image, start_point, end_point, (0, 0, 255), 5)  # Red line with thickness of 2
    # Draw lines between the destination points on the warped image
    for i in range(len(destination_points1)):
        # Map destination points to integers
        start_point_ = tuple(map(int, destination_points1[i]))
        end_point_ = tuple(map(int, destination_points1[(i + 1) % len(destination_points1)]))
        cv2.line(warped_image1, start_point_, end_point_, (255, 0, 0), 5)  # Blue line with thickness of 2



    # Second perspective transform
    original_points2 = np.float32([
        [0, height // 3],          # Top-left of the lower third
        [width, height // 3],      # Top-right of the lower third
        [width, height],           # Bottom-right corner
        [0, height],               # Bottom-left corner
    ])

    destination_points2 = np.float32([
        [0, 0],                   # Top-left corner
        [width, 0],               # Top-right corner
        [width - param, height],    # Bottom-right corner
        [param, height],            # Bottom-left corner
    ])

    # Compute and apply the second transformation
    matrix2 = cv2.getPerspectiveTransform(original_points2, destination_points2)
    warped_image2 = cv2.warpPerspective(warped_image1, matrix2, (width, height))



    # Third perspective transform (this seems redundant but kept for consistency)
    original_points3 = np.float32([
        [0, height],              # Top-left of the lower half
        [width, height],          # Top-right of the lower half
        [width, height],          # Bottom-right corner
        [0, height],              # Bottom-left corner
    ])

    destination_points3 = np.float32([
        [0, 0],                   # Top-left corner
        [width, 0],               # Top-right corner
        [width - param, height],    # Bottom-right corner
        [param, height],            # Bottom-left corner
    ])

    # Compute and apply the third transformation
    matrix3 = cv2.getPerspectiveTransform(original_points3, destination_points3)
    warped_image3 = cv2.warpPerspective(warped_image2, matrix3, (width, height))


def IPM(image):
    # Dimensions of the image
    height, width = image.shape[:2]

    # Forth perspective transform (this seems redundant but kept for consistency)
    param1 = 570
    param2 = 35
    # First perspective transform
    original_points4 = np.float32([
        [0, (height // 2)+param2],          # Top-left of the lower half
        [width, (height // 2)+param2],      # Top-right of the lower half
        [width, height],           # Bottom-right corner
        [0, height],               # Bottom-left corner
    ])

    destination_points4 = np.float32([
        [0, 0],                   # Top-left corner
        [width, 0],               # Top-right corner
        [width - param1, height*2],    # Bottom-right corner
        [param1, height*2],            # Bottom-left corner
    ])

    # Compute and apply the first transformation
    matrix4 = cv2.getPerspectiveTransform(original_points4, destination_points4)
    warped_image4 = cv2.warpPerspective(image, matrix4, (width, height*2))



    # final_warped_image = warped_image1.copy()
    # final_warped_image = warped_image2.copy()
    # final_warped_image = warped_image3.copy()
    final_warped_image = warped_image4.copy()
    #final_warped_image = cv2.resize(final_warped_image, (width, height))


    return image, final_warped_image




image, ipm_image = IPM(image)



# Display the results
plt.figure(figsize=(12, 6))
# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Warped image with destination points
plt.subplot(1, 2, 2)
plt.title("Warped Image")
plt.imshow(cv2.cvtColor(ipm_image, cv2.COLOR_BGR2RGB))

# # Warped image with destination points
# plt.subplot(1, 3, 3)
# plt.title("Warped Image 2")
# plt.imshow(cv2.cvtColor(warped_image2, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()
