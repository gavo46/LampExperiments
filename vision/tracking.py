import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision

from vision.camera import Camera

MODEL_PATH = "vision/blaze_face_short_range.tflite"


def _make_detector():
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def detect_face(detector, frame):
    """
    Run face detection on a single BGR frame.

    Returns (face_present, face_x): face_x is the normalized [0, 1]
    horizontal position of the first detected face's center, or 0.5
    when no face is present.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if not result.detections:
        return False, 0.5

    bbox = result.detections[0].bounding_box
    center_x = bbox.origin_x + bbox.width / 2
    return True, center_x / frame.shape[1]


def watch(state, camera_id=0):
    """
    Blocks forever. Continuously reads frames from the camera, runs
    face detection, and writes face presence/position into `state`
    (also stashing the raw frame there so vision/scene.py can describe
    whatever's currently in view on demand).

    Meant to be run on its own daemon thread.
    """
    detector = _make_detector()
    camera = Camera(camera_id)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            try:
                face_present, face_x = detect_face(detector, frame)
            except RuntimeError:
                detector = _make_detector()
                continue

            state.set_face(face_present, face_x)
            state.set_last_frame(frame)
    finally:
        camera.release()


if __name__ == "__main__":
    # Quick manual check: prints face presence/position to the terminal.
    import threading
    import time

    from character.state import SharedState

    demo_state = SharedState()
    threading.Thread(target=watch, args=(demo_state,), daemon=True).start()

    while True:
        print(demo_state.get_face())
        time.sleep(0.2)
