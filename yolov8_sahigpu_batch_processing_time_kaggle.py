import argparse
import time
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

import torch
import pandas as pd
import numpy as np

def run(weights="yolov8n.pt", source="images", view_img=False, save_img=False, exist_ok=False):
    start_time = time.time()

    # Check source path
    source_path = Path(source)
    if not source_path.exists() or not source_path.is_dir():
        raise FileNotFoundError(f"Source path '{source}' does not exist or is not a directory.")

    # Model setup
    yolov8_model_path = f"models/{weights}"
    download_yolov8s_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=yolov8_model_path, confidence_threshold=0.4, device="cuda:0"
    )

    # Get images from directory
    images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    if len(images) < 2:
        raise ValueError(f"Source directory '{source}' should contain at least two images.")

    # Process the first image once
    performance_data = []
    performance_data.extend(process_image(images[0], detection_model, view_img, save_img, source, 1, save_dir))

    # Process the second image ten times
    for i in range(10):
        performance_data.extend(process_image(images[1], detection_model, view_img, save_img, source, i + 2, save_dir))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.2f} seconds")

    # Convert performance data to DataFrame and print
    df = pd.DataFrame(performance_data)
    print(df)

def process_image(image_path, detection_model, view_img, save_img, source, iteration, save_dir):
    performance_data = []
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        return

    # Resize frame
    frame = cv2.resize(frame, (1280, 720))

    # Preprocessing
    preprocess_start_time = time.time()
    frame_tensor = torch.from_numpy(frame).to(torch.device("cuda:0")).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    preprocess_end_time = time.time()
    preprocess_time = preprocess_end_time - preprocess_start_time

    # Object detection
    od_start_time = time.time()
    results = get_sliced_prediction(
        frame, detection_model,
        slice_height=384, slice_width=384,
        overlap_height_ratio=0.05, overlap_width_ratio=0.05
    )
    od_end_time = time.time()
    od_time = od_end_time - od_start_time

    # Postprocessing
    postprocess_start_time = time.time()
    for object_prediction in results.object_prediction_list:
        bbox = object_prediction.bbox
        category_name = object_prediction.category.name
        score = object_prediction.score.value

        # Draw bounding box
        cv2.rectangle(frame, 
                      (int(bbox.minx), int(bbox.miny)), 
                      (int(bbox.maxx), int(bbox.maxy)), 
                      (255, 0, 0), 2)
        
        # Draw label
        label = f"{category_name}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, 
                      (int(bbox.minx), int(bbox.miny) - label_height - 5),
                      (int(bbox.minx) + label_width, int(bbox.miny)), 
                      (255, 0, 0), -1)
        cv2.putText(frame, label, 
                    (int(bbox.minx), int(bbox.miny) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if view_img:
        cv2.imshow(image_path.stem, frame)
    if save_img:
        cv2.imwrite(str(save_dir / f"{image_path.stem}_result.jpg"), frame)
    
    postprocess_end_time = time.time()
    postprocess_time = postprocess_end_time - postprocess_start_time

    # Store performance metrics
    performance_data.append({
        'Image': str(image_path),
        'Iteration': iteration,
        'Preprocess Time (s)': preprocess_time,
        'Object Detection Time (s)': od_time,
        'Postprocess Time (s)': postprocess_time,
        'Total Time (s)': preprocess_time + od_time + postprocess_time
    })

    return performance_data

def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    parser.add_argument("--source", type=str, required=True, help="directory with images")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    return parser.parse_args()

def main(opt):
    """Main function."""
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
