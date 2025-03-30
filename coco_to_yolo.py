import json
import os
from pathlib import Path

def convert_coco_to_yolo(coco_annotation_path, output_dir):
    # Read COCO annotation file
    with open(coco_annotation_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create category id to number mapping (0-based index for YOLO)
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    # Get image dimensions for normalization
    image_dims = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)

    # Process each image
    for img in coco_data['images']:
        image_id = img['id']
        img_width = img['width']
        img_height = img['height']
        filename = Path(img['file_name']).stem  # Get filename without extension
        
        # Create txt file for this image
        txt_path = os.path.join(output_dir, f"{filename}.txt")
        
        with open(txt_path, 'w') as f:
            # If image has annotations
            if image_id in image_annotations:
                for ann in image_annotations[image_id]:
                    # Get category id and convert to YOLO class index
                    category_id = categories[ann['category_id']]
                    
                    # Get bbox coordinates (COCO format: x, y, width, height)
                    x, y, w, h = ann['bbox']
                    
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    # Write to file
                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == "__main__":
    coco_annotation_path = "train_dataset/_annotations.coco.json"
    output_dir = "labels/train_dataset/"  # Directory where txt files will be saved
    convert_coco_to_yolo(coco_annotation_path, output_dir)
    print("Conversion completed!")