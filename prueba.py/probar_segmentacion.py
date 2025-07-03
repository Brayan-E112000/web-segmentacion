from ultralytics import YOLO
import cv2

# Cargar tu modelo entrenado con segmentación
model = YOLO("entrenamiento_danos_seg/modelo_segmentacion/weights/best.pt")

# Imagen que quieres probar
imagen = "prueba.01.jpg"

# Ejecutar la predicción
resultados = model.predict(source=imagen, save=True, conf=0.3, task="segment")

# Mostrar visualmente (opcional)
img = cv2.imread(imagen)
cv2.imshow("Resultado con Segmentación", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
