# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import argparse
import time
import numpy as np

import cv2

from src.utils_demo import (
    FaceDetectorCV2,
    FaceDetectorYUNET,
    FaceTracker,
    HGPredictor,
    MediapipePredictor,
    TrackHandler,
    Visualizer,
)

def bounding_box(face):
    bbox = face.loc
    return f"{bbox.x1},{bbox.y1},{bbox.x2},{bbox.y2}"

def face_landmarks(face):
    # Five facial landmarks
    if "ldmks" in face.prediction:
        landmarks = np.array(face.prediction["ldmks"]).reshape(-1, 3)
        s = ""
        for i, (x, y, z) in enumerate(landmarks):
            s = s + f",{x:.1f},{y:.1f},{z:.1f}"
        return s[1:]
    return "no_landmarks"

def main(args):
    gesture_labels = ["none", "nod", "shake", "tilt", "turn", "up/down"]

    # Instantiate
    if args.face_detector == "YUNET":
        face_detector = FaceDetectorYUNET()
    elif args.face_detector == "CV2":
        face_detector = FaceDetectorCV2()
    else:
        raise ValueError("Invalid face detector")
    face_tracker = FaceTracker()
    face_predictor = MediapipePredictor()
    hg_predictor = HGPredictor("cpu")
    track_handler = TrackHandler(face_tracker)
    visualizer = Visualizer(
        draw_bbox=args.draw_bbox,
        draw_landmarks=args.draw_landmarks,
        draw_head_gesture=args.draw_head_gesture,
    )
    face_ids = {}

    if args.file:
        vid = cv2.VideoCapture(args.file)
    else:
        vid = cv2.VideoCapture(0)

    FPS = args.fps
    i = -1

    if args.startat > 0:
        vid.set(cv2.CAP_PROP_POS_MSEC, args.startat)

    probability_headers = [f"prob_{label.replace('/', '_')}" for label in gesture_labels]
    print(
        ",".join(
            [
                "frame",
                "frame_time",
                "face_id",
                "head_gesture",
                "confidence",
                "prediction_timestamp",
                *probability_headers,
                "x_start,y_start,x_end,y_end",
            ]
        )
    )
    input_fps = FPS
    target_fps = 30

    while True:
        i = i + 1

        # Capture the video frame by frame
        start = time.time()
        ret, frame = vid.read()

        if not ret or frame is None:
            print("End of video.")
            break

        frame_time = vid.get(cv2.CAP_PROP_POS_MSEC)
        #frame_time = int(round((i / FPS) * 1000))

        should_process = (
            int(i * target_fps / input_fps)
            != int((i - 1) * target_fps / input_fps)
        )

        if not should_process:
            print(f"FPS Ignore {i}")
            continue

        if not args.file:
            frame_time = int(round(time.time() * 1000))

        # Detect faces
        detection = face_detector.process_image(frame)

        # Track the faces
        face_tracker.update(detection, frame_time)

        # Get the current track id
        track_id = face_tracker.get_tracks()

        # Detect the face landmarks in those faces
        for track in track_id:
            face_prediction = face_predictor.process_face(
                frame, face_tracker.tracks_store[track][-1]
            )
            face_tracker.tracks_store[track][-1].add_prediction(face_prediction)

        try:
            output_track = hg_predictor.process(face_tracker, track_id)
        except ValueError as e:
            print(e)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if str(e) == "Segement is too short":
                print(f"Segement too short, continue (at {i})")
                continue
            raise

        track_handler.add_track_prediction(output_track)

        # Draw the output
        frame = visualizer.process(frame, face_tracker, output_track)

        # print(output_track)
        for track in output_track:
            if not track in face_ids:
                face_ids[track] = len(face_ids)
            face = face_tracker.tracks_store[track][-1]

            output_data = output_track[track]
            if output_data["probabilities"] is None:
                probability_values = ["not_available"] * len(gesture_labels)
            else:
                probability_values = [
                    f"{output_data['probabilities'][label]:.6f}" for label in gesture_labels
                ]
            print(
                f"{i},{frame_time},{face_ids[track]},"
                f"{output_data['head_gesture']},{output_data['confidence']:.6f},"
                f"{output_data['timestamp']},{','.join(probability_values)},"
                f"{bounding_box(face)}"
            )

        if not output_track:
            unavailable_probabilities = ",".join(["not_available"] * len(gesture_labels))
            print(
                f"{i},{frame_time},-1,no_face,not_available,not_available,"
                f"{unavailable_probabilities}"
            )

        # Draw real time fps
        end = time.time()
        fps = 1 / (end - start)
        if args.view:
            cv2.putText(
                frame,
                f"FPS {int(fps)} ",
                (450, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            # Display the resulting frame
            cv2.imshow("frame", frame)

            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # After the loop release the cap object
    vid.release()
    if args.view:
        # Destroy all the windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run demo")
    parser.add_argument(
        "--file",
        type=str,
        help="File to process, leave empty to use webcam",
        required=False,
    )
    parser.add_argument(
        "--face_detector",
        type=str,
        default="CV2",
        help="Face detector CV2 is faster but less accurate if face more than 2m away",
        choices=["YUNET", "CV2"],
    )
    parser.add_argument(
        "--startat",
        type=int,
        default=0,
        help="Time in milliseconds to start in the video (only when using file)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (defaults to 30) (only used when file is set)",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        default=False,
        help="Show GUI",
    )
    parser.add_argument(
        "--draw_bbox",
        "--db",
        type=bool,
        default=True,
        help="Draw bounding box",
    )
    parser.add_argument(
        "--draw_landmarks",
        "--dl",
        type=bool,
        default=True,
        help="Draw landmarks",
    )
    parser.add_argument(
        "--draw_head_gesture",
        "--dhg",
        type=bool,
        default=True,
        help="Draw head gesture",
    )
    args = parser.parse_args()

    # run demo head gesture
    main(args)
