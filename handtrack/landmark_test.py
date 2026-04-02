# Run pip install mediapipe
# May need to downgrade numpy version to <2.0

camera_index = 0

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import numpy as np

import matplotlib.pyplot as plt

cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)
landmarker = HandLandmarker.create_from_options(options)

def compute_camera_intrinsics(width, height, hfov_deg):
    hfov_rad = np.deg2rad(hfov_deg)

    fx = width / (2.0 * np.tan(hfov_rad / 2.0))
    fy = fx * (height / width)

    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])

    return K, fx, fy, cx, cy

# MediaPipe landmark indices
WRIST = 0
INDEX_MCP = 5
INDEX_TIP = 8
PINKY_MCP = 17

SCALE_PAIRS = [
    (WRIST, INDEX_MCP),
    (INDEX_MCP, INDEX_TIP),
    (WRIST, PINKY_MCP),
]

def pixel_distance(lm1, lm2, width, height):
    x1, y1 = lm1.x * width, lm1.y * height
    x2, y2 = lm2.x * width, lm2.y * height
    return np.hypot(x2 - x1, y2 - y1)

def world_distance(w1, w2):
    return np.linalg.norm([
        w1.x - w2.x,
        w1.y - w2.y,
        w1.z - w2.z
    ])

def compute_depth_scale(norm_landmarks, world_landmarks):
    scales = []

    for i, j in SCALE_PAIRS:
        # World-space distance (meters)
        wi = world_landmarks[i]
        wj = world_landmarks[j]

        d_world = np.linalg.norm([
            wi.x - wj.x,
            wi.y - wj.y,
            wi.z - wj.z
        ])

        # Normalized landmark distance (MediaPipe space)
        ni = norm_landmarks[i]
        nj = norm_landmarks[j]

        d_norm = np.linalg.norm([
            ni.x - nj.x,
            ni.y - nj.y,
            ni.z - nj.z
        ])

        if d_norm > 1e-6:
            scales.append(d_world / d_norm)

    return np.mean(scales)

def estimate_hand_depth(
    lm_norm, lm_world,
    fx, width, height,
    i, j
):
    L_pixel = pixel_distance(
        lm_norm[i], lm_norm[j],
        width, height
    )

    L_real = world_distance(
        lm_world[i], lm_world[j]
    )

    if L_pixel < 1e-6:
        return None

    Z = fx * L_real / L_pixel
    return Z


def wrist_to_camera_frame(
    wrist_norm,
    Z,
    fx, fy, cx, cy,
    width, height
):
    # Pixel coordinates
    x_pix = wrist_norm.x * width
    y_pix = wrist_norm.y * height

    # Camera-frame coordinates
    X = (x_pix - cx) / fx * Z
    Y = (y_pix - cy) / fy * Z

    return np.array([X, Y, Z])

print("Press 'q' to quit")

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        print("Frame grab failed")
        break

    # OpenCV captures in BGR; convert to RGB (sRGB color space)
    frame_srgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Ensure dtype is uint8 (standard for sRGB images)
    frame_srgb = frame_srgb.astype(np.uint8)

    # mp_image = mp.Image.create_from_file('image.jpg')
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_srgb)

    results = landmarker.detect(mp_image)
    # print(hand_landmarker_result)

    hand_landmarks_list = results.hand_landmarks
    hand_world_landmarks_list = results.hand_world_landmarks
    handedness_list = results.handedness

    for i in range(len(hand_landmarks_list)):
        if handedness_list[i][0].category_name != 'Right':
            break

        hand_landmarks = hand_landmarks_list[i]
        world_landmarks = hand_world_landmarks_list[i]

        # Camera params
        width, height = frame_srgb.shape[1], frame_srgb.shape[0]
        K, fx, fy, cx, cy = compute_camera_intrinsics(
            width, height, hfov_deg=120.0
        )

        # Depth scale (meters per MediaPipe unit)
        scale = compute_depth_scale(hand_landmarks, world_landmarks)
        Z = estimate_hand_depth(hand_landmarks, world_landmarks, fx, width, height, WRIST, INDEX_MCP) 

        # Wrist position in camera frame (meters)
        wrist_cam = wrist_to_camera_frame(
            hand_landmarks[WRIST],
            Z,
            fx, fy, cx, cy,
            width, height
        )

        print("Wrist position (camera frame, meters):", wrist_cam)

        cv2.putText(frame_bgr, f"({wrist_cam[0]:.3f}, {wrist_cam[1]:.3f}, {wrist_cam[2]:.3f})", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        
        # for landmark in hand_landmarker_result.hand_world_landmark:
        #     ax.scatter(landmark.x, landmark.y, landmark.z)

        # plt.show()


    cv2.imshow("RunCam Live", frame_bgr)
    # Important: waitKey is required for window events on macOS
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

