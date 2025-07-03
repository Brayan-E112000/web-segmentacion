from ultralytics import YOLO
import cv2

# Cargar el modelo segmentador entrenado
model = YOLO("entrenamiento_danos_seg/modelo_segmentacion/weights/best.pt")

# Ruta de la imagen a probar
imagen = "prueba.01.jpg"

# Ejecutar predicción y guardar el resultado con máscaras
resultados = model.predict(source=imagen, save=True, conf=0.3, task="segment")

# Mostrar visualmente (opcional)
img = cv2.imread(imagen)
cv2.imshow("Segmentación del modelo", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
