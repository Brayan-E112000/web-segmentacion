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