# Object Detection with Tracking

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat-square&logo=opencv&logoColor=white)
![YOLOv5](https://img.shields.io/badge/YOLOv5-FFA500?style=flat-square&logo=PyTorch&logoColor=white)
![DeepSort](https://img.shields.io/badge/DeepSort-FF4500?style=flat-square&logoColor=white)

## Description
Object Detection with Tracking is a project that utilizes Python, OpenCV, YOLOv5, and DeepSort for real-time object detection and tracking. The code takes inputs in the form of a video or direct input from a webcam, detecting objects and drawing bounding boxes around them along with their names and confidence scores. For object tracking, it assigns unique IDs to the detected objects and tracks them over consecutive frames. The YOLOv5 model is used with default weights trained on the COCO dataset, and for feature extraction in DeepSort, 'osnet_x0_25' is used.

![object detection](https://learnopencv.com/wp-content/uploads/2024/01/object_detection.gif)

## Technologies Used
- Python
- OpenCV
- YOLOv5
- YOLOv8
- DeepSort

## Usage
To run the project, make sure you have Python and the required libraries installed. Then, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-github-username/object-detection-with-tracking.git

2. **Navigate to the project directory:**
	  ```bash
      cd object-detection-with-tracking
      
3. **Unzip the deep_sort file:**
	  ```bash
      unzip deep_sort.zip 
4. **Install the requirements :**
   	```bash
    pip install requirement.txt
5. **Run the script for detecting:**
	```bash
    python detect.py

6. **Run the script for tracking :**
	```bash 
    python track.py

For training on custom dataset :
1. **Clone the repository:**
   ```bash
   python train.py
   #set the data_path to the path of your yaml file     
## Usage of own weights

If you would want the model to detect things that you have trained your model on , you can simply change the weight file of the yolov5n.pt inside the track.py and give your own weight files .
