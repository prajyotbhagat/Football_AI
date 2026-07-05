import os
import glob
from ultralytics import YOLO

def main():
    # Path to your best trained model
    model_path = "football_detection/yolov8l_high_recall/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please ensure training has completed successfully and generated the best.pt file.")
        return

    print(f"Loading trained model from {model_path}...")
    model = YOLO(model_path)
    
    # Get a few images from the test set
    test_images_dir = "yolo_dataset/images/test"
    test_images = glob.glob(os.path.join(test_images_dir, "*.jpg"))
    
    if not test_images:
        print(f"Error: No test images found in {test_images_dir}")
        return
        
    # Select the first 5 images for testing
    num_images_to_test = 5
    images_to_test = test_images[:num_images_to_test]
    
    print(f"Running inference on {len(images_to_test)} images...")
    
    # Run prediction and save the results
    # The results will be saved in a new folder automatically
    results = model.predict(
        source=images_to_test,
        save=True,      # Save images with bounding boxes drawn
        conf=0.10,      # VERY LOW confidence threshold to guarantee high recall
        iou=0.6,        # Intersection over union threshold for Non-Max Suppression
        project="football_detection",
        name="test_predictions_high_recall",
        exist_ok=True
    )
    
    print("\nInference completed successfully!")
    print("You can view the saved images with the predicted bounding boxes in the 'football_detection/test_predictions_high_recall/' folder.")

if __name__ == "__main__":
    main()
