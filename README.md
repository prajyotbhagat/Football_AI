# Football AI: Player, Referee, & Ball Detection

This repository contains an object detection pipeline powered by **YOLOv8 (Ultralytics)**, designed to detect players, referees, goalkeepers, and footballs in match footage. It includes advanced training strategies optimized for stable gradient steps, high recall (>95%), and VRAM efficiency.

---

## 📂 Repository Structure

*   **`train_advanced.py` / `train_advanced.ipynb`**: The advanced training script and notebook. Configured with a large **YOLOv8x** model and optimized hyperparameters (resolution, scale augmentations, loss weights) designed to achieve >95% recall on a 6 GB GPU.
*   **`train.py` / `train_football_detection.ipynb`**: Baseline training configurations utilizing the medium **YOLOv8m** and small **YOLOv8s** models.
*   **`predict.py`**: A general-purpose command-line prediction script that runs inference on custom sources (images, videos, or directories) using the trained model weights.
*   **`test.py`**: A quick script to verify predictions on a small subset of test images.
*   **`convert_dataset.py`**: Helper script for dataset conversion.
*   **`requirements.txt`**: Standard python package requirements.

---

## ⚙️ Installation & Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Running Training

### 1. Baseline Training (YOLOv8m)
To run the baseline training pipeline:
```bash
python train.py
```

### 2. Advanced Training (YOLOv8x - High Recall)
Optimized for 6 GB VRAM with a large model footprint, high batch stability, and advanced augmentations (mosaic, copy-paste) while preventing OOM crashes:
```bash
python train_advanced.py
```
*Alternatively, you can run and visualize each epoch within **`train_advanced.ipynb`**.*

---

## 🔮 Inference & Prediction

Once training is complete, the best model weights will be automatically saved under:
`runs/detect/football_detection/yolov8x_high_recall_v2/weights/best.pt`

Use **`predict.py`** to run predictions.

### Examples:

1.  **Run on the default test dataset folder:**
    ```bash
    python predict.py
    ```

2.  **Run on a specific image or video:**
    ```bash
    python predict.py --source path/to/image.jpg
    ```

3.  **Tune prediction threshold (e.g., lower confidence to `0.05` for maximum recall):**
    ```bash
    python predict.py --source path/to/image.jpg --conf 0.05
    ```

### Output Location
All predictions will draw bounding boxes around players, referees, goalkeepers, and balls, saving the results in:
`runs/detect/football_detection/predictions/`