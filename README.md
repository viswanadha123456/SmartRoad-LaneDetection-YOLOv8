# SmartRoad-LaneDetection-YOLOv8
A Comprehensive YOLOv8-Based Framework for Smart Road Lane Segmentation and Lane Detection to Enhance Autonomous Driving in Electric Vehicles
**Introduction**
The project aims to develop an intelligent system for road lane segmentation and damage detection using deep learning techniques.
It is to support Electric Vehicles (EVs) by enhancing road awareness and assisting in autonomous navigation.
The system utilizes YOLOv8, a state-of-the-art model known for real-time object detection and segmentation.
The trained model is deployed on a Raspberry with a camera module, enabling real-time lane and damage detection on the edge.
This integration provides a cost-effective, compact solution for smart EV systems, focusing on safety and automation.
**Dataset Preparation**
Images resized to 1088x1920 resolution.
Total Images: 2020
Annotation Tool: Computer Vision Annotation Tool (CVAT)
Annotation Format: YOLO format (bounding boxes for road lanes)
Classes:
Single class: "road_lane"
Dataset Split:
1616 images for training (80%)
404 images for validation (20%)
Preprocessing:
Data augmentation (horizontal flip, brightness adjustment).
**Deployment of YOLOv8 for Real-Time Lane Detection in Video Streams**
rom ultralytics import YOLO
import cv2
# Load the YOLOv8 segmentation model
model = YOLO("/home/zuber/work/runs/segment/train/weights/best.pt")
# Load the video
video_path = "/home/zuber/work/1.mp4"
cap = cv2.VideoCapture(video_path)
# Get original frame size and FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Check if video is in portrait mode
is_portrait = height > width
# Adjust output size
if is_portrait:
    out_width, out_height = height, width  # swap dimensions
else:
    out_width, out_height = width, height
# Save output video
output_path = "/home/zuber/work/o1.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Rotate if portrait
    if is_portrait:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Inference
    results = model.predict(source=frame, task='segment', save=False, stream=False, verbose=False)
    # Annotate and display the frame
    annotated_frame = results[0].plot()
    out.write(annotated_frame)
    cv2.imshow("Lane Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()**
**Video Link Road  Detection ****
 < href a :https://www.canva.com/design/DAG5rrsZYHI/K-qxpWgVsSQVqIKzWmXUvQ/view?presentation >
**  Conclusiom**
 This project demonstrates an effective and reliable approach to Smart Road Lane Segmentation and Lane Detection for Electric Vehicles using YOLOv8. By leveraging advanced deep learning techniques, the system achieves accurate lane recognition, robust segmentation, and real-time performance—key requirements for modern autonomous and intelligent driving technologies.
The implemented model showcases strong adaptability across diverse road conditions, including varying lighting, weather, and complex lane structures. With its modular design, this repository provides a solid foundation for researchers, developers, and EV manufacturers to further enhance autonomous driving features such as lane-keeping, driver assistance, and road safety monitoring.
Overall, this work contributes to the development of next-generation AI-driven electric vehicle navigation systems, offering a scalable and efficient solution for future mobility innovations.

