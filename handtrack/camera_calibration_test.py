import cv2
import numpy as np

from calibration_io import load_fisheye_params

# Hard-coded camera index
CAMERA_INDEX = 0

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Capture one frame
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture image")
    exit()

# Print image size
h, w = frame.shape[:2]
print(f"Image size: {w} x {h}")

params = load_fisheye_params()
K = params.k
D = params.d

# Compute undistortion maps
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K,
    D,
    np.eye(3),
    K,
    (w, h),
    cv2.CV_16SC2
)

# Apply undistortion
undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

# Display image
cv2.imshow("Undistorted Image", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
