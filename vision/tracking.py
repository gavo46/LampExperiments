import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path="vision/blaze_face_short_range.tflite",
    delegate=python.BaseOptions.Delegate.CPU
)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)

detector = vision.FaceDetector.create_from_options(options)

capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    face_present = len(result.detections) > 0
    print(face_present)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()