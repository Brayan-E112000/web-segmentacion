from roboflow import Roboflow
from ultralytics import YOLO

# Conectar con tu proyecto en Roboflow
rf = Roboflow(api_key="nzv483Zw1qv967g0jGFo")
project = rf.workspace("investigacin-31").project("deteccion-de-danos-nzpy5")
dataset = project.version(1).download("coco-segmentation")  # ← esta es la forma correcta

# Cargar modelo base de segmentación
model = YOLO("yolov8n-seg.pt")

# Entrenar con segmentación
model.train(
    data=dataset.location + "/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    project="entrenamiento_danos_seg",
    name="modelo_segmentacion",
    task="segment"
)

