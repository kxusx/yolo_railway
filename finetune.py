from ultralytics import YOLO
import yaml
import os

# First, create a data.yaml file
def create_data_yaml():
    data = {
        'path': '.',  # dataset root dir (modified to use current directory)
        'train': 'images/train',  # train images
        'val': 'images/val',      # val images
        'test': 'images/test',    # test images
        
        # Classes (modified to match your label files which use class 1)
        'names': {
            1: 'object'  # class index 1 based on your label files
        },
        'nc': 1  # number of classes
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
    }
    
    # Train the model
    results = model.train(**args)
    
    # Export the model
    model.export(format='onnx')  # Export to ONNX format
    
    # Save the model
    model.save('best_model.pt')
    
 

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
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    
                    # Write in YOLO format
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Compare predictions with ground truth
    from generate_test_labels import compare_predictions
    results = compare_predictions('labels/test', 'predictions/test')
    
    print("\nTest Set Evaluation Results:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print("\nDetailed Metrics:")
    print(f"Total Ground Truth Boxes: {results['metrics']['total_gt']}")
    print(f"Total Predicted Boxes: {results['metrics']['total_pred']}")
    print(f"True Positives: {results['metrics']['true_positives']}")
    print(f"False Positives: {results['metrics']['false_positives']}")
    print(f"False Negatives: {results['metrics']['false_negatives']}")
    
    return results

if __name__ == "__main__":
    # Create data.yaml file
    create_data_yaml()
    
    # Train the model
    train_yolo()
    # print("\nValidation Results:")
    # print(val_results)
    
    # Evaluate on test set
    test_results = evaluate_on_test_set()
