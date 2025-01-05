# face-recognition-and-video-processing
Face recognition and matching using OpenCV, python and face_recognition


Project Description
Objective:

The project identifies unique faces from video files uploaded by users.
It processes videos to filter out redundant faces and retains only one instance of each unique face in the output dataset.
Key Features:

Video Upload: Users can upload multiple video files through a web interface.
Face Recognition: Uses pre-trained facial encodings to recognize and match faces across frames in the videos.
Unique Face Extraction: Processes videos to store frames (or videos) containing unique faces in a dataset folder.
Dynamic Results Display: After processing, a list of videos with unique faces is displayed on the web interface.
Workflow:

Frontend Interaction:
Users interact with the web interface built using Flask, HTML, CSS, and JavaScript.
A spinner is displayed while videos are processed, and results are shown dynamically.
Backend Processing:
The uploaded videos are sent to the server, where the match_faces.py script:
Loads pre-existing facial encodings from a file (encodings.pickle).
Processes each video to detect and compare faces using the face_recognition library.
Saves frames or new video files containing unique faces in the dataset folder.
Facial encodings of unmatched faces are updated in encodings.pickle.
Output Delivery:
The unique face videos are served to users dynamically via Flask endpoints.
Management Tools:

A "Clear" button lets users delete uploaded files, the dataset folder, and reset the system for fresh uploads.


![image](https://github.com/user-attachments/assets/9d29ed48-b314-4aed-87e8-2d816f53dbbe)

![image](https://github.com/user-attachments/assets/70d04f80-4ce9-4f6b-9b8d-6f1505f28e33)

![image](https://github.com/user-attachments/assets/8fd210bb-54cf-4eb2-ab98-71a6b31c0ddb)



