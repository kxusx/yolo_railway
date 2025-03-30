from ultralytics import YOLO
import yaml
import os

def fix_label_files():
    """Fix label files by converting class index 1 to 0"""
    import glob
    from pathlib import Path
    
    # Process both train and test labels
    for split in ['train', 'test']:
        label_files = glob.glob(f'labels/{split}/*.txt')
        print(f"Processing {len(label_files)} {split} label files...")
        
        for label_path in label_files:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Convert class index from 1 to 0
            fixed_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == '1':  # if class index is 1
                    parts[0] = '0'  # change to 0
                    fixed_lines.append(' '.join(parts) + '\n')
            
            # Write back the fixed labels
            with open(label_path, 'w') as f:
                f.writelines(fixed_lines)

def create_data_yaml():
    current_dir = os.getcwd()  # Get current working directory
    data = {
        'path': current_dir,  # dataset root dir
        'train': os.path.join(current_dir, 'images/train'),  # train images
        'val': os.path.join(current_dir, 'images/train'),    # validation images
        'test': os.path.join(current_dir, 'images/test'),    # test images
        
        # Classes (using 0-based indexing)
        'names': ['object'],  # single class, index 0
        'nc': 1  # number of classes (just one class)
    }
    
    os.makedirs('dataset', exist_ok=True)
    with open('dataset/data.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def train_yolo():
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')  # n is for nano, you can use s/m/l/x for larger models
    
    # Training arguments
    args = {
        'data': 'dataset/data.yaml',  # path to data.yaml
        'epochs': 100,                # number of epochs
        'imgsz': 640,                # image size
        'batch': 16,                 # batch size
        'device': 0,                 # cuda device, i.e. 0 or 0,1,2,3 or cpu
        'workers': 8,                # number of worker threads
        'patience': 50,              # early stopping patience
        'project': 'runs/train',     # save results to project/name
        'name': 'exp1',              # experiment name
        'pretrained': True,          # use pretrained model
        'optimizer': 'Adam',         # optimizer to use (SGD, Adam)
        'lr0': 0.001,               # initial learning rate
        'weight_decay': 0.0005,     # weight decay
        'warmup_epochs': 3,         # warmup epochs
        'warmup_momentum': 0.8,     # warmup momentum
        'warmup_bias_lr': 0.1,      # warmup initial bias lr
        'box': 7.5,                 # box loss gain
        'cls': 0.5,                 # cls loss gain
        'dfl': 1.5,                 # dfl loss gain
        'save': True,               # save train checkpoints
        'save_period': -1,          # Save checkpoint every x epochs (disabled if < 1)
        'cache': False,             # cache images for faster training
        'val': False,               # disable validation
        'amp': False,               # disable automatic mixed precision
        'half': False,              # disable half precision
    }
    
    try:
        # Train the model
        results = model.train(**args)
        
        # Export the model
        model.export(format='onnx')  # Export to ONNX format
        
        # Save the model
        model.save('best_model.pt')
        
        return results
    except RuntimeError as e:
        print(f"CUDA error encountered. Trying CPU training instead...")
        args['device'] = 'cpu'
        results = model.train(**args)
        model.export(format='onnx')
        model.save('best_model.pt')
        return results

def evaluate_on_test_set(model_path='best_model.pt'):
    """
    Evaluate the model on the test set and calculate metrics
    """
    import glob
    from pathlib import Path
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Get all test images
    test_images = glob.glob('images/test/*.jpg') + glob.glob('images/test/*.jpeg') + glob.glob('images/test/*.png')
    
    # Create directory for predictions
    os.makedirs('predictions/test', exist_ok=True)
    
    # Initialize counters for accuracy calculation
    total_predictions = 0
    correct_predictions = 0
    
    # Run predictions on test images and save results
    for img_path in test_images:
        results = model(img_path)
        
        # Get the corresponding prediction file path
        img_name = Path(img_path).stem
        pred_path = f'predictions/test/{img_name}.txt'
        
        # Save predictions in YOLO format
        with open(pred_path, 'w') as f:
            for r in results:
                for box in r.boxes:
                    # Get class, confidence and normalized box coordinates
                    cls = int(box.cls[0])  # This will now be 0 instead of 1
                    conf = float(box.conf[0])
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    
                    # Write in YOLO format (converting class 0 back to 1 for compatibility with existing labels)
                    f.write(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    total_predictions += 1
                    if conf > 0.5:  # Consider predictions with confidence > 0.5 as correct
                        correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Compare predictions with ground truth
    from generate_test_labels import compare_predictions
    results = compare_predictions('labels/test', 'predictions/test')
    
    print("\nTest Set Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print("\nDetailed Metrics:")
    print(f"Total Predictions: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Ground Truth Boxes: {results['metrics']['total_gt']}")
    print(f"Total Predicted Boxes: {results['metrics']['total_pred']}")
    print(f"True Positives: {results['metrics']['true_positives']}")
    print(f"False Positives: {results['metrics']['false_positives']}")
    print(f"False Negatives: {results['metrics']['false_negatives']}")
    
    return {**results, 'accuracy': accuracy}

def verify_dataset():
    import glob
    from pathlib import Path
    
    # Get all training images
    image_files = glob.glob('images/train/*.jpg') + glob.glob('images/train/*.jpeg') + glob.glob('images/train/*.png')
    
    print(f"Found {len(image_files)} images in training directory")
    
    valid_pairs = 0
    for img_path in image_files:
        img_name = Path(img_path).stem
        label_path = f'labels/train/{img_name}.txt'
        
        if os.path.exists(label_path):
            valid_pairs += 1
        else:
            print(f"Missing label file for {img_path}")
    
    print(f"Found {valid_pairs} valid image-label pairs")
    
    return valid_pairs > 0

if __name__ == "__main__":
    # Fix label files first
    fix_label_files()
    
    # Create data.yaml file
    create_data_yaml()
    
    # Clear the cache if it exists
    cache_file = 'labels/train.cache'
    if os.path.exists(cache_file):
        os.remove(cache_file)
    
    # Verify dataset before training
    if not verify_dataset():
        print("No valid image-label pairs found. Please check your dataset structure.")
        exit(1)
    
    # Train the model
    val_results = train_yolo()
    print("\nValidation Results:")
    print(val_results)
    
    # Evaluate on test set
    test_results = evaluate_on_test_set()
