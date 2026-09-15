import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision

import config
from utils import metrics
from vision.camera import Camera

MODEL_PATH = "vision/blaze_face_short_range.tflite"

# How often to log/record engagement stats, and the window they cover
# (each report resets the counters, so it's the last REPORT_INTERVAL_SECONDS,
# not a sliding window).
REPORT_INTERVAL_SECONDS = 30.0


class _EngagementTracker:
    """
    Counts frames, frames-with-a-face, and face_present flips since the
    last report. update() is called every frame; maybe_report() checks
    whether REPORT_INTERVAL_SECONDS has elapsed and, if so, prints and
    records detection rate + flips/minute, then resets the window.
    """

    def __init__(self):
        self._window_start = time.monotonic()
        self._frame_count = 0
        self._face_frame_count = 0
        self._flip_count = 0
        self._last_present = None

    def update(self, face_present):
        self._frame_count += 1
        if face_present:
            self._face_frame_count += 1
        if self._last_present is not None and face_present != self._last_present:
            self._flip_count += 1
        self._last_present = face_present

    def maybe_report(self):
        elapsed = time.monotonic() - self._window_start
        if elapsed < REPORT_INTERVAL_SECONDS or self._frame_count == 0:
            return

        detection_rate = self._face_frame_count / self._frame_count
        flips_per_minute = self._flip_count * (60.0 / elapsed)

        print(
            "[metrics] engagement  "
            f"detection_rate={detection_rate:.1%}  "
            f"flips/min={flips_per_minute:.1f}  "
            f"frames={self._frame_count}  window={elapsed:.0f}s"
        )
        metrics.record_engagement(detection_rate, flips_per_minute)

        self._window_start = time.monotonic()
        self._frame_count = 0
        self._face_frame_count = 0
        self._flip_count = 0


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
    tracker = _EngagementTracker() if config.MEASURE else None

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

            if tracker:
                tracker.update(face_present)
                tracker.maybe_report()
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
