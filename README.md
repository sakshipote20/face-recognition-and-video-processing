# Face Recognition and Video Processing

## **Project Description**

### **Objective**

- The project identifies unique faces from video files uploaded by users.
- It processes videos to filter out redundant faces and retains only one instance of each unique face in the output dataset.

### **Key Features**

1. **Video Upload**:
   - Users can upload multiple video files through a web interface.
2. **Face Recognition**:
   - Uses pre-trained facial encodings to recognize and match faces across frames in the videos.
3. **Unique Face Extraction**:
   - Processes videos to store frames (or videos) containing unique faces in a dataset folder.
4. **Dynamic Results Display**:
   - After processing, a list of videos with unique faces is displayed on the web interface.

---

## **Workflow**

### **Frontend Interaction**

- Users interact with the web interface built using **Flask**, **HTML**, **CSS**, and **JavaScript**.
- A spinner is displayed while videos are processed, and results are shown dynamically.

### **Backend Processing**

- The uploaded videos are sent to the server, where the `match_faces.py` script:
  1. Loads pre-existing facial encodings from a file (`encodings.pickle`).
  2. Processes each video to detect and compare faces using the **face_recognition** library.
  3. Saves frames or new video files containing unique faces in the dataset folder.
  4. Updates facial encodings of unmatched faces in `encodings.pickle`.

### **Output Delivery**

- The unique face videos are served to users dynamically via Flask endpoints.

### **Management Tools**

- A **Clear** button allows users to delete uploaded files, the dataset folder, and reset the system for fresh uploads.

---

## **Technologies, Tools, Algorithms & Techniques Used**

1. **OpenCV**, **Python**, & **Deep Learning**:

   - **OpenCV** provides functions for reading, processing, and analyzing visual data.
   - Deep learning-based metric algorithms are used to perform face recognition.
   - Python libraries like `numpy`, `datetime`, etc., assist in data handling and operations.

2. **Dlib** & **face_recognition Libraries**:

   - Used to perform face detection, vector calculation, comparison, and matching.
   - The **pickle** library is used to save Python objects like encodings.

3. **HTML**, **CSS**, & **JavaScript**:

   - Frontend development to create a user-friendly web interface.

4. **Flask**:

   - A web framework that integrates the front end with the backend, enabling interaction between the user and the server.

---

### **Core Components**

#### **1. OpenCV**

- Enables the application to perform operations on video frames, such as resizing, color conversion, and writing output videos.

#### **2. face_recognition Library**

- Detects and recognizes faces using advanced algorithms.
- Encodes facial features and compares them with stored encodings to identify individuals in videos.
- Network architecture is based on **ResNet-34** with fewer layers and reduced filters.
- Achieves **99.38% accuracy** on the **Labeled Faces in the Wild (LFW)** dataset.

#### **3. Flask**

- Handles user requests and serves responses, enabling seamless integration between the frontend and backend.

---

## **Screenshots**

![Image 1](https://github.com/user-attachments/assets/0af58489-410d-494c-8652-99658162f8e9)

![Image 2](https://github.com/user-attachments/assets/58265256-1076-48f1-9297-1296316effdd)


