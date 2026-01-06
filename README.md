# 3D Object Detection and Multi-Object Tracking for Autonomous Vehicles

This project implements a complete **3D object detection and multi-object tracking pipeline** designed for autonomous driving applications. The system processes sensor data to detect, track, and visualize multiple objects in real time, enabling robust perception for self-driving vehicles.

---

## 🚀 Project Features

- 3D object detection from point cloud data
- Multi-object tracking across consecutive frames
- Bird’s-eye-view (BEV) visualization
- Model inference and integration pipeline
- Performance evaluation and analysis tools
- Real-time detection loop implementation

---

## 📂 Project Structure

### 🔹 Core Pipeline

- **Detection_loop.py**  
  Main execution loop for continuous 3D object detection and tracking across frames.

- **model_integ.py**  
  Handles model loading, inference, and integration with the detection pipeline.

---

### 🔹 Data Processing & Utilities

- **pcl.py**  
  Point cloud processing utilities including filtering and transformation operations.

- **birdeyeview.py**  
  Generates bird’s-eye-view (BEV) representations for visualization and analysis.

- **load_save_funcs.py**  
  Helper functions for loading and saving models, configurations, and outputs.

- **obj_labels.py**  
  Defines object class labels used during detection and tracking.

---

### 🔹 Visualization & Evaluation

- **visualization.py**  
  Visualization tools for detected and tracked objects in both 3D and BEV formats.

- **evaluation.py**  
  Performance evaluation metrics and benchmarking for detection and tracking accuracy.

- **data_analysis.py**  
  Analytical scripts for post-processing results and generating insights.

---

### 🔹 Additional Files

- **__pycache__/**  
  Cached Python files generated during execution.

- **drive-download-*.zip**  
  Archived datasets or pretrained resources used for experimentation.

---

## 🧠 Technologies Used

- Python
- 3D Computer Vision
- Point Cloud Processing (LiDAR)
- Deep Learning Models
- NumPy, OpenCV, Matplotlib
- Autonomous Driving Perception Concepts

---

## ▶️ How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
