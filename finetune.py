from ultralytics import YOLO
import yaml

# First, create a data.yaml file
def create_data_yaml():
    data = {
        'path': 'train_dataset',  # dataset root dir
        'train': 'images/train',  # train images (relative to 'path')
        'val': 'images/val',      # val images (relative to 'path')
        'test': 'images/test',    # test images (optional)
        
        # Classes
        'names': {
            0: 'object'  # replace with your class name
        },
        'nc': 1  # number of classes
    }
    
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
    
    # Evaluate model performance on validation set
    results = model.val()
    
    # Export the model
    model.export(format='onnx')  # Export to ONNX format
    
    # Save the model
    model.save('best_model.pt')

def predict_test_images():
    # Load the trained model
    model = YOLO('best_model.pt')
    
    # Run inference on test images
    results = model('path/to/test/image.jpg')
    
    # Print results
    for r in results:
        print(f"Detected {len(r.boxes)} objects")
        for box in r.boxes:
            print(f"Class: {box.cls}, Confidence: {box.conf:.2f}, Box: {box.xyxy}")

if __name__ == "__main__":
    # Create data.yaml file
    create_data_yaml()
    
    # Train the model
    train_yolo()
    
    # Optional: test the model
    predict_test_images()
