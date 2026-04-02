import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera index {i}: OK, shape={frame.shape}")
        cap.release()

