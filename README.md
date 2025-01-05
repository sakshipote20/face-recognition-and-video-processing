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


![image](https://github.com/user-attachments/assets/0af58489-410d-494c-8652-99658162f8e9)

![image](https://github.com/user-attachments/assets/58265256-1076-48f1-9297-1296316effdd)


Overview of the technologies, tools, algorithms & techniques used 
OpenCV, Python & Deep Learning: OpenCV provides functions for reading, processing & analyzing visual data. Deep learning based metric algorithm to perform face recognition. Python libraries like numpy, datetime, etc.
Dlib & face_recognition libraries: Used to perform face detection, vector calculation, comparison & matching. 
Pickle library is used to save python objects.
HTML, CSS & Java script for frontend development.
Flask, a web framework for interaction of front end with backend.

1. OpenCV: for face capture, enables our application to perform complex operations on video frames, such as resizing, converting colors, and writing output videos.

2. Face_Recognition: for face detection and recognition, is like the brain of our application, responsible for recognizing faces in videos.
- It uses advanced algorithms to detect and recognize faces by analyzing facial features and comparing them with known faces.
The library enables our application to identify individuals in videos by encoding facial characteristics and matching them with stored encodings.
The network architecture for face recognition is based on ResNet-34 but with fewer layers and the number of filters reduced by half.
On the Labeled Faces in the Wild (LFW) dataset the network compares to other state-of-the-art methods, reaching 99.38% accuracy.

3. Flask: for integration, helps in handling user requests and serving responses.
![image](https://github.com/user-attachments/assets/83648948-5873-4bbd-9ca4-6e9a71d0ff14)

