# Run pip install mediapipe
# May need to downgrade numpy version to <2.0

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import numpy as np

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

def intrinsics_from_k(k):
    # Return fx, fy, cx, cy
    return k, k[0, 0], k[1, 1], k[0, 2], k[1, 2]

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
    wrist_norm, # Wrist in pixel space
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


mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

class HandLandmarks:
    """Class which encapsulates mediapipe hand landmark extractor. Get world coordinates from an image, knowing the camera intrinsics"""
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a hand landmarker instance with the image mode:
        options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
                running_mode=VisionRunningMode.IMAGE)
        self.landmarker = HandLandmarker.create_from_options(options)

    def draw_landmarks_on_image(self, bgr_image, detection_result):
      hand_landmarks_list = detection_result.hand_landmarks
      handedness_list = detection_result.handedness
      annotated_image = np.copy(bgr_image)

      # Loop through the detected hands to visualize.
      for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

      return annotated_image

    def getCoords(self, image, intrinsics=None, grip_dist=0.05, hand='Left'):
        """Return world coordinates from cv2 BGR image"""
        # OpenCV captures in BGR; convert to RGB (sRGB color space)
        frame_srgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure dtype is uint8 (standard for sRGB images)
        frame_srgb = frame_srgb.astype(np.uint8)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_srgb)

        results = self.landmarker.detect(mp_image)

        hand_landmarks_list = results.hand_landmarks
        hand_world_landmarks_list = results.hand_world_landmarks
        handedness_list = results.handedness

        for i in range(len(hand_landmarks_list)):
            if handedness_list[i][0].category_name != hand:
                continue

            hand_landmarks = hand_landmarks_list[i]
            world_landmarks = hand_world_landmarks_list[i]

            # Camera params
            width, height = frame_srgb.shape[1], frame_srgb.shape[0]
            if intrinsics is not None:
                K, fx, fy, cx, cy = intrinsics_from_k(intrinsics)
            else:
                K, fx, fy, cx, cy = compute_camera_intrinsics(
                    width, height, hfov_deg=120.0
                )

            # Depth scale (meters per MediaPipe unit)
            # scale = compute_depth_scale(hand_landmarks, world_landmarks)
            Z = estimate_hand_depth(hand_landmarks, world_landmarks, fx, width, height, WRIST, INDEX_MCP) 

            # Wrist position in camera frame (meters)
            wrist_cam = wrist_to_camera_frame(
                hand_landmarks[WRIST],
                Z,
                fx, fy, cx, cy,
                width, height
            )

            # Distance between index and thumb for gripping
            grip = world_distance(world_landmarks[4], world_landmarks[8]) < grip_dist

            return wrist_cam, grip, results
        return None, None, None

