import cv2
import numpy as np

# Input and output file paths
input_video_path =  'carla_test_vid_156.mp4' 
output_video_path = 'carla_BEV_IPM_output_.mp4' 



# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width, frame_height = 1280, 800
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 output
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))




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


    # final_warped_image = warped_image1.copy()
    final_warped_image = warped_image2.copy()
    # final_warped_image = warped_image3.copy()
    # final_warped_image = cv2.resize(warped_image2, (800, 800))

    return final_warped_image




def picture_in_picture(main_image, overlay_image, img_ratio=3, border_size=3, x_margin=30, y_offset_adjust=-100):
    """
    Overlay an image onto a main image with a white border.
    
    Args:
        main_image_path (str): Path to the main image.
        overlay_image_path (str): Path to the overlay image.
        img_ratio (int): The ratio to resize the overlay image height relative to the main image.
        border_size (int): Thickness of the white border around the overlay image.
        x_margin (int): Margin from the right edge of the main image.
        y_offset_adjust (int): Adjustment for vertical offset.

    Returns:
        np.ndarray: The resulting image with the overlay applied.
    """
    # Load images
    if main_image is None or overlay_image is None:
        raise FileNotFoundError("One or both images not found.")

    # Resize the overlay image to 1/img_ratio of the main image height
    new_height = main_image.shape[0] // img_ratio
    new_width = int(new_height * (overlay_image.shape[1] / overlay_image.shape[0]))
    overlay_resized = cv2.resize(overlay_image, (new_width, new_height))

    # Add a white border to the overlay image
    overlay_with_border = cv2.copyMakeBorder(
        overlay_resized,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    # Determine overlay position
    x_offset = main_image.shape[1] - overlay_with_border.shape[1] - x_margin
    y_offset = (main_image.shape[0] // 2) - overlay_with_border.shape[0] + y_offset_adjust

    # Overlay the image
    main_image[y_offset:y_offset + overlay_with_border.shape[0], x_offset:x_offset + overlay_with_border.shape[1]] = overlay_with_border

    return main_image





# Process the video
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    frame_imp = IPM(frame)

    frame = picture_in_picture(frame, frame_imp)

    # Display the frame
    cv2.imshow('Frame', frame)


    # Write the frame to the output video
    out.write(frame)

    # Press 'q' to exit the display window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects and close windows
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved as: {output_video_path}")

