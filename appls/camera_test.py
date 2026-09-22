import cv2

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
try:
    if not cap.isOpened():
        raise SystemExit("Cannot open /dev/video0")

    frame = None
    for _ in range(30):
        ok, img = cap.read()
        if ok:
            frame = img

    if frame is None:
        raise SystemExit("No frames received")

    if not cv2.imwrite("camera_test.jpg", frame):
        raise SystemExit("Failed to save image")

    print("Success: camera_test.jpg")
    print("Resolution:", frame.shape[1], "x", frame.shape[0])
finally:
    cap.release()
