from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")  # Load the YOLO model
    data_path = "/Users/pushpakreddy/yoga poses rohit/YOLO YOGA Dataset.v1i.yolov11/data.yaml"  # Path to your YAML file

    # Train the model using CPU
    train_results = model.train(
        data=data_path,  # Path to the dataset YAML
        epochs=10,        # Number of training epochs
        imgsz=640,        # Training image size
        device='cpu',     # Use CPU instead of GPU
    )
