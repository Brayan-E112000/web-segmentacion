
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import os, uuid
import numpy as np

app = Flask(__name__)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results/prediccion", exist_ok=True)
model = YOLO("best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    resumen = {}
    if request.method == "POST":
        imagen = request.files["imagen"]
        nombre = str(uuid.uuid4()) + ".jpg"
        ruta_guardada = os.path.join("static/uploads", nombre)
        imagen.save(ruta_guardada)
        resultados = model.predict(source=ruta_guardada, save=True, conf=0.3, task="segment",
                                   project="static/results", name="prediccion", exist_ok=True, verbose=False)
        masks = resultados[0].masks
        classes = resultados[0].boxes.cls.cpu().numpy()
        if masks is not None:
            total_pixels = 0
            area_por_clase = {}
            for i, cls_id in enumerate(classes):
                mask = masks.data[i].cpu().numpy()
                area = np.sum(mask)
                total_pixels += area
                class_name = model.names[int(cls_id)]
                area_por_clase[class_name] = area_por_clase.get(class_name, 0) + area
            for clase in ["Crack", "Moisture", "Material Loss"]:
                area = area_por_clase.get(clase, 0)
                porcentaje = (area / total_pixels) * 100 if total_pixels > 0 else 0
                resumen[clase] = {"porcentaje": round(porcentaje, 1), "color": "gray", "gravedad": "-"}
        resultado = os.path.join("results/prediccion", nombre)
        return render_template("index.html", resultado=resultado, resumen=resumen)
    return render_template("index.html", resultado=None, resumen=None)

@app.route('/static/results/prediccion/<filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'static/results/prediccion'), filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
