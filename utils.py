from typing import List
import numpy as np
import cv2
import os
import json


def draw_bounding_box(image, bbox, color=(255, 0, 0), thickness=2):
    h, w, _ = image.shape
    x_min = int(bbox[0] * w)
    y_min = int(bbox[1] * h)
    x_max = int(bbox[2] * w)
    y_max = int(bbox[3] * h)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

def only_jpg_files(files: List) -> List[str]:
    return [item for item in files if item.endswith('jpg')]

def get_bbox(filename: str, dirname: str) -> List[float]:
    json_file = os.path.join(dirname, f'{filename}.json')
    with open(json_file, 'r') as f:
        bbox = json.load(f)['bbox']
    return bbox

def create_mask_from_bbox(image_path, bounding_box):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    height, width = image.shape[:2]
    
    mask = np.zeros((height, width), dtype=np.float32)
    
    x_min, y_min, x_max, y_max = bounding_box
    
    x_min = int(x_min * width)
    y_min = int(y_min * height)
    x_max = int(x_max * width)
    y_max = int(y_max * height)
    
    mask[y_min:y_max, x_min:x_max] = 255
        
    return mask
