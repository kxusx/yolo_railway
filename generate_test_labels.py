import json
import os
from pathlib import Path

def generate_test_labels(coco_test_annotation_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read COCO test annotation file
    with open(coco_test_annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create category id to number mapping (assuming same categories as training)
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    # Create mapping of image id to dimensions
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
        filename = Path(img['file_name']).stem
        
        # Create txt file for this image
        txt_path = os.path.join(output_dir, f"{filename}.txt")
        
        with open(txt_path, 'w') as f:
            if image_id in image_annotations:
                for ann in image_annotations[image_id]:
                    # Convert category_id to YOLO class index
                    category_id = categories[ann['category_id']]
                    
                    # Get bbox in COCO format
                    x, y, w, h = ann['bbox']
                    
                    # Convert to YOLO format
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    # Write YOLO format line
                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def compare_predictions(ground_truth_dir, predictions_dir, iou_threshold=0.5):
    """
    Compare model predictions with ground truth labels
    """
    import numpy as np
    
    def calculate_iou(box1, box2):
        # Convert from center format to corner format
        def get_corners(box):
            x_center, y_center, width, height = box
            x1 = x_center - width/2
            y1 = y_center - height/2
            x2 = x_center + width/2
            y2 = y_center + height/2
            return np.array([x1, y1, x2, y2])
        
        box1_corners = get_corners(box1)
        box2_corners = get_corners(box2)
        
        # Calculate intersection
        x1 = max(box1_corners[0], box2_corners[0])
        y1 = max(box1_corners[1], box2_corners[1])
        x2 = min(box1_corners[2], box2_corners[2])
        y2 = min(box1_corners[3], box2_corners[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union
        box1_area = (box1_corners[2] - box1_corners[0]) * (box1_corners[3] - box1_corners[1])
        box2_area = (box2_corners[2] - box2_corners[0]) * (box2_corners[3] - box2_corners[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    metrics = {
        'total_gt': 0,
        'total_pred': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0
    }
    
    # Process each ground truth file
    for gt_file in os.listdir(ground_truth_dir):
        if not gt_file.endswith('.txt'):
            continue
            
        pred_file = os.path.join(predictions_dir, gt_file)
        gt_file = os.path.join(ground_truth_dir, gt_file)
        
        # Read ground truth boxes
        gt_boxes = []
        with open(gt_file, 'r') as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.strip().split())
                gt_boxes.append((class_id, (x, y, w, h)))
        
        # Read prediction boxes
        pred_boxes = []
        if os.path.exists(pred_file):
            with open(pred_file, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    pred_boxes.append((class_id, (x, y, w, h)))
        
        metrics['total_gt'] += len(gt_boxes)
        metrics['total_pred'] += len(pred_boxes)
        
        # Match predictions to ground truth
        matched_gt = set()
        for pred_class, pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for idx, (gt_class, gt_box) in enumerate(gt_boxes):
                if idx in matched_gt or gt_class != pred_class:
                    continue
                    
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_gt_idx >= 0:
                metrics['true_positives'] += 1
                matched_gt.add(best_gt_idx)
            else:
                metrics['false_positives'] += 1
        
        metrics['false_negatives'] += len(gt_boxes) - len(matched_gt)
    
    # Calculate final metrics
    precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
    recall = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives']) if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'metrics': metrics,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

if __name__ == "__main__":
    # Generate ground truth labels for test set
    coco_test_annotation_path = "images/test/_annotations.coco.json"
    test_labels_dir = "labels/test"
    generate_test_labels(coco_test_annotation_path, test_labels_dir)
    print(f"Generated test labels in: {test_labels_dir}")
    
    # Optional: Compare with model predictions
    # Uncomment and use after you have model predictions
    """
    predictions_dir = "path/to/model/predictions"
    results = compare_predictions(test_labels_dir, predictions_dir)
    print("\nEvaluation Results:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print("\nDetailed Metrics:")
    print(f"Total Ground Truth Boxes: {results['metrics']['total_gt']}")
    print(f"Total Predicted Boxes: {results['metrics']['total_pred']}")
    print(f"True Positives: {results['metrics']['true_positives']}")
    print(f"False Positives: {results['metrics']['false_positives']}")
    print(f"False Negatives: {results['metrics']['false_negatives']}")
    """