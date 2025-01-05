# series of operations that involve face detection, comparison with pre-existing facial encodings, and recording frames without matches into new video files.


#SECTION 1 : import the necessary packages

import imutils

#handles video streaming 
from imutils.video import VideoStream

#face_recognition libray is used for face recognition and comparison
import face_recognition

#argparse is used to parse command line arguments
#Command line arguments are flags given to a program/script at runtime.It is similar to a function parameter.
#They contain additional information for our program so that it can execute.
import argparse

#It allows you to save Python objects such as lists, dictionaries, and models to files, and load them back into memory.
import pickle

#It provides functions for creating and navigating directories, and manipulating files.
import os

#library for performing computer vision tasks like image or video processing
import cv2
from datetime import datetime

#NumPy is used to perform a wide variety of mathematical operations on arrays
import numpy as np

#SECTION 2 : construct the argument parse to parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", required=True,
    help="path to input folder containing video files")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
    help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


#SECTION 3 : Below code recognizes faces in videos using OpenCV, Python, and deep learning.

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

dataset_folder = "dataset"
os.makedirs(dataset_folder, exist_ok=True)

for video_file in os.listdir(args["input"]):
    video_path = os.path.join(args["input"], video_file)

    print(f"[INFO] processing video: {video_path}")
    vs = cv2.VideoCapture(video_path)

    original_format = int(vs.get(cv2.CAP_PROP_FOURCC))

    match_found = False
    frames_without_matches = []

    while True:
        if match_found:
            break
        
        frame = vs.read()[1]

        if frame is not None and frame.size > 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width=750)
            r = frame.shape[1] / float(rgb.shape[1])

            boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
            encodings = face_recognition.face_encodings(rgb, boxes)

            if not encodings:
                print("No faces detected.")
                break

            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                distances = face_recognition.face_distance(data["encodings"], encoding)

                if distances.size > 0:
                    min_distance = np.min(distances)
                    min_distance_index = np.argmin(distances)
                    name = "Unknown"

                    if True in matches:
                        threshold_distance = 0.5

                        if min_distance < threshold_distance:
                            name = data["names"][min_distance_index]
                            print(f"Match found with distance: {min_distance} for {name}")
                            match_found = True
                            break
                    else:
                        print("No matches found")
                        frames_without_matches.append({
                            "frame": frame.copy(),
                            "encoding": encoding
                        })

                else:
                    print("No matches found")
                    frames_without_matches.append({
                        "frame": frame.copy(),
                        "encoding": encoding
                    })
        else:
            break

    if not match_found and frames_without_matches:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_video_file = os.path.join(dataset_folder, f"{video_name}_no_match_{timestamp}.mp4")

        if frames_without_matches:
            frame_shape = frames_without_matches[0]["frame"].shape
            output_writer = cv2.VideoWriter(output_video_file, original_format, 20, (frame_shape[1], frame_shape[0]), True)

            for frame_info in frames_without_matches:
                data["encodings"].append(frame_info["encoding"])
                data["names"].append(output_video_file)
                output_writer.write(frame_info["frame"])

            output_writer.release()

    vs.release()

    with open(args["encodings"], "wb") as f:
        f.write(pickle.dumps(data))
