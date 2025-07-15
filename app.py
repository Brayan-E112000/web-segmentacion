from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)
model = YOLO("best.pt")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Traducción de clases según tu modelo
clase_traduccion = {
    'grieta': '🪨 Grietas',
    'humedad': '💧 Humedad',
    'desprendimiento de material': '🧱 Desprendimiento'
}

@app.route("/", methods=["GET", "POST"])
def index():
    resumen_daños = {k: 0 for k in clase_traduccion.keys()}
    imagenes_resultado = []

    if request.method == "POST":
        files = request.files.getlist("images")

        for file in files:
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            results = model(filepath)

            for result in results:
                # Guardar imagen con segmentación marcada
                result_path = os.path.join(RESULT_FOLDER, filename)
                result.save(filename=result_path)

                if result.boxes and result.boxes.cls is not None:
                    clases_detectadas = result.boxes.cls.tolist()
                    nombres = result.names

                    for c in clases_detectadas:
                        nombre_clase = nombres[int(c)]
                        if nombre_clase in resumen_daños:
                            resumen_daños[nombre_clase] += 1

            imagenes_resultado.append(filename)

    total_daños = sum(resumen_daños.values())
    porcentajes = {}

    for clase in resumen_daños:
        cantidad = resumen_daños[clase]
        porcentaje = (cantidad / total_daños) * 100 if total_daños > 0 else 0
        porcentajes[clase_traduccion[clase]] = round(porcentaje, 1)

    return render_template("index.html",
                           imagenes=imagenes_resultado,
                           porcentajes=porcentajes)

if __name__ == "__main__":
    app.run(debug=True)
