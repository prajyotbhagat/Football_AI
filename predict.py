import os
import glob
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8x Football Player Detection predictions using best.pt.")
    parser.add_argument(
        "--source", 
        type=str, 
        default="yolo_dataset/images/test", 
        help="Path to an image, directory of images, or video file for prediction."
    )
    parser.add_argument(
        "--weights", 
        type=str, 
        default="runs/detect/football_detection/yolov8x_high_recall_v2/weights/best.pt",
        help="Path to the model weights file."
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.10, 
        help="Confidence threshold for predictions (default: 0.10)."
    )
    parser.add_argument(
        "--iou", 
        type=float, 
        default=0.6, 
        help="IOU threshold for NMS (default: 0.6)."
    )
    parser.add_argument(
        "--project", 
        type=str, 
        default="football_detection", 
        help="Project directory name for saving results."
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="predictions", 
        help="Run name/folder for saving prediction results."
    )
    
    args = parser.parse_args()
    
    # ── Resolve Model Path ─────────────────────────────────────────────────
    # Try the default path, then fallbacks based on different workspace root viewpoints
    model_path = args.weights
    if not os.path.exists(model_path):
        fallbacks = [
            os.path.join("football_detection", "yolov8x_high_recall_v2", "weights", "best.pt"),
            os.path.join("yolov8x_high_recall_v2", "weights", "best.pt"),
            os.path.join("weights", "best.pt")
        ]
        
        found = False
        for path in fallbacks:
            if os.path.exists(path):
                model_path = path
                found = True
                break
                
        if not found:
            print(f"Error: Could not locate 'best.pt' weights.")
            print(f"Tried paths:")
            print(f"  - {os.path.abspath(args.weights)}")
            for path in fallbacks:
                print(f"  - {os.path.abspath(path)}")
            return

    print(f"Loading YOLO model from: {os.path.abspath(model_path)}")
    model = YOLO(model_path)

    # ── Resolve Source Path ────────────────────────────────────────────────
    source = args.source
    if not os.path.exists(source):
        print(f"Error: Source path '{source}' does not exist.")
        return

    print(f"Running inference on source: {source}")
    
    results = model.predict(
        source=source,
        save=True,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        exist_ok=True
    )
    
    # Display output path dynamically from the results object
    save_dir = results[0].save_dir if results else os.path.join(args.project, args.name)
    print("\nInference completed successfully!")
    print(f"You can find the saved predictions with bounding boxes drawn in:")
    print(f"  {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()
