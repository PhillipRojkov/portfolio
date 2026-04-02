import cv2
import numpy as np

from calibration_io import DEFAULT_VALUES_PATH, FisheyeParams, save_fisheye_params

# ----------------------------
# Configuration
# ----------------------------
CAMERA_INDEX = 0

CHECKERBOARD = (10, 7)   # inner corners (columns, rows)
SQUARE_SIZE = 23.2       # mm

MIN_CALIBRATION_FRAMES = 15

# termination criteria
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ----------------------------
# Prepare object points
# ----------------------------
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

# ----------------------------
# Open camera
# ----------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

print("Press SPACE to capture calibration frame")
print("Press Q to finish calibration")

# ----------------------------
# Capture frames
# ----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    display = frame.copy()

    if found:
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (3,3),
            (-1,-1),
            criteria
        )

        cv2.drawChessboardCorners(display, CHECKERBOARD, corners2, found)

    cv2.imshow("Calibration", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        if found:
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"Captured frame {len(objpoints)}")
        else:
            print("Chessboard not detected")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nCaptured {len(objpoints)} valid frames")

if len(objpoints) < MIN_CALIBRATION_FRAMES:
    print("Not enough frames for calibration")
    exit()

# ----------------------------
# Fisheye calibration
# ----------------------------
h, w = gray.shape[:2]

K = np.zeros((3,3))
D = np.zeros((4,1))

rvecs = []
tvecs = []

flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
    cv2.fisheye.CALIB_CHECK_COND +
    cv2.fisheye.CALIB_FIX_SKEW
)

rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    (w, h),
    K,
    D,
    rvecs,
    tvecs,
    flags,
    criteria
)

print("\nCalibration RMS error:", rms)

print("\nCamera matrix (K):")
print(K)

print("\nDistortion coefficients (D):")
print(D)

# Save values for reuse in other scripts.
save_fisheye_params(FisheyeParams(k=K, d=D, image_size=(w, h)), DEFAULT_VALUES_PATH)
print(f"\nSaved calibration values to: {DEFAULT_VALUES_PATH}")

# ----------------------------
# Undistort example frame
# ----------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
ret, frame = cap.read()
cap.release()

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K,
    D,
    np.eye(3),
    K,
    (w, h),
    cv2.CV_16SC2
)

undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

cv2.imshow("Original", frame)
cv2.imshow("Undistorted", undistorted)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Captured 16 valid frames

# Calibration RMS error: 0.47355188792894787

# Camera matrix (K):
# [[825.62992016   0.         935.41128309]
#  [  0.         825.40561869 589.92733883]
#  [  0.           0.           1.        ]]

# Distortion coefficients (D):
# [[ 0.06673122]
#  [-0.03313482]
#  [ 0.07564529]
#  [-0.04242766]]
