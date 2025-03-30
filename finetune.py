from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' or 'yolov8m.pt' for larger models

# Train on your dataset
model.train(
    data="dataset/data.yaml",  # Path to your dataset YAML
    epochs=50,  # Number of epochs
    imgsz=640,  # Image size
    batch=16,   # Batch size
    device="cuda",  # Use GPU if available
    workers=4,  # Number of workers
    project="yolo_training",
    name="exp1",
)

# Evaluate model
model.val()

# Export model for deployment
model.export(format="onnx")  # Export to ONNX, TensorRT, etc.

# save the model
model.save("yolo_model.pt")

# calculate accuracy after training



