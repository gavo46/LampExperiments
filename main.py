import threading
import time
import cv2
import mujoco
import mujoco.viewer
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from control.poses import poses
from control.controller import move_toward_target

face_present = False
face_x = 0.5


def watch_camera():
    global face_present, face_x

    base_options = python.BaseOptions(
        model_asset_path="vision/blaze_face_short_range.tflite"
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

        if result.detections:
            face_present = True
            bbox = result.detections[0].bounding_box
            center_x = bbox.origin_x + bbox.width / 2
            face_x = center_x / frame.shape[1]
        else:
            face_present = False


def main():
    model = mujoco.MjModel.from_xml_path("robot/dummy_lamp_5dof.urdf")
    data = mujoco.MjData(model)

    threading.Thread(target=watch_camera, daemon=True).start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if face_present:
                target = list(poses["alert"])
                target[0] = (face_x - 0.5) * -2.0
            else:
                target = poses["neutral"]

            move_toward_target(data, target)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()