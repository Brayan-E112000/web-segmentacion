<<<<<<< HEAD
from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
from collections import Counter
import urllib.request

app = Flask(__name__)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# ========== DESCARGA DEL MODELO DESDE GOOGLE DRIVE ==========
MODEL_PATH = "best.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1h08AfLSOvvLEDMCPNIHWWZJSKqpy2CM4"

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo desde Google Drive...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ========== CARGA DEL MODELO ==========
model = YOLO(MODEL_PATH)

# Traducción de clases
CLASS_TRANSLATIONS = {
    "crack": "Grietas",
    "humidity": "Humedad",
    "detachment": "Desprendimiento"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    analysis = {}
    if request.method == 'POST':
        images = request.files.getlist('images')
        total_counts = Counter()
        filenames = []

        for image in images:
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join("static/uploads", filename)
            image.save(filepath)

            results = model.predict(filepath, task="segment")
            names = model.names
            result = results[0]

            masks = result.masks
            classes = result.boxes.cls.tolist() if result.boxes else []

            class_counts = Counter()
            for c in classes:
                class_name = names[int(c)]
                class_counts[class_name] += 1

            total_counts += class_counts
            filenames.append(filename)

        total = sum(total_counts.values())
        for key in ["crack", "humidity", "detachment"]:
            count = total_counts.get(key, 0)
            percent = (count / total * 100) if total > 0 else 0
            analysis[key] = {
                "translated": CLASS_TRANSLATIONS.get(key, key),
                "percent": round(percent, 1)
            }

    return render_template("index.html", analysis=analysis)

# ========== CONFIGURACIÓN PARA RAILWAY ==========
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
=======

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
>>>>>>> 536bd878816bff700b5419a08d3aaebbf30e985e
