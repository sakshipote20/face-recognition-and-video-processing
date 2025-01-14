# SECTION 1: Import the necessary packages
import imutils  # For image processing
import face_recognition  # For face detection and recognition
import argparse  # For command-line argument parsing
import pickle  # For saving and loading serialized data
import os  # For file and directory operations
import cv2  # For video and image processing
from datetime import datetime  # For timestamping output files
import numpy as np  # For numerical computations
import shutil  # For file copying

# SECTION 2: Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", required=True, help="path to input folder containing video files")
ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# SECTION 3: Load facial encodings and process videos
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# Create a dataset folder if it doesn't already exist
dataset_folder = "dataset"
os.makedirs(dataset_folder, exist_ok=True)

# Loop through each video file in the input folder
for video_file in os.listdir(args["input"]):
    video_path = os.path.join(args["input"], video_file)

    print(f"[INFO] processing video: {video_path}")
    vs = cv2.VideoCapture(video_path)

    original_format = int(vs.get(cv2.CAP_PROP_FOURCC))  # Original video codec
    match_found = False  # Track if any face matches are found
    frames_without_matches = []  # Store frames without face matches

    while True:
        if match_found:  # Exit loop if a match is found
            break

        frame = vs.read()[1]

        if frame is not None and frame.size > 0:
            # Convert frame to RGB and resize for faster processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width=750)
            r = frame.shape[1] / float(rgb.shape[1])

            # Detect face locations and compute encodings
            boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
            encodings = face_recognition.face_encodings(rgb, boxes)

            if not encodings:  # No faces detected in the frame
                print("No faces detected.")
                break

            # Compare encodings with known faces
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

    # Save original video to dataset folder if no match is found
    if not match_found:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_video_file = os.path.join(dataset_folder, f"{video_name}_no_match_{timestamp}.mp4")

        shutil.copy(video_path, output_video_file)  # Copy original video
        print(f"[INFO] Original video saved to: {output_video_file}")

        # Update dataset with unmatched encodings
        for frame_info in frames_without_matches:
            data["encodings"].append(frame_info["encoding"])
            data["names"].append(output_video_file)

    vs.release()

    # Save updated encodings to the file
    with open(args["encodings"], "wb") as f:
        f.write(pickle.dumps(data))


os.remove("encodings.pickle")