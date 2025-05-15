from ultralytics import YOLO

# Load a pre-trained YOLOv8 segmentation model (nano)
model = YOLO("yolo11n-seg.pt")  # options: n, s, m, l, x

# Train on your dataset
model.train(
    data="C://OCT//dataset//dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    name="oct-rpe-pr2-model"
)

results = model.predict(source="data/images/11.PNG", save=True, imgsz=640)
