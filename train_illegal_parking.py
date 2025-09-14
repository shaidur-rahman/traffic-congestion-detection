from ultralytics import YOLO

# Define your dataset YAML
# Example: dataset.yaml should look like this:
# path: datasets/parking
# train: images/train
# val: images/val
# names:
#   0: illegal_car
#   1: illegal_bus
#   2: moving_car
#   3: moving_bus

def main():
    # Load a YOLOv11 base model (can use yolov11s.pt, yolov11m.pt etc.)
    model = YOLO("yolov11s.pt")

    # Train on your dataset
    results = model.train(
        data="datasets/parking/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="illegal_parking_yolo",
        workers=4,
        device=0  # set to "cpu" if no GPU
    )

if __name__ == "__main__":
    main()
