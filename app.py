from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
from collections import Counter
from PIL import Image
import pytesseract

app = Flask(__name__)
model = YOLO("modelo/best.pt")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Traducción de clases (según tu modelo)
CLASSES_TRADUCIDAS = {
    "grieta": "Grietas",
    "humedad": "Humedad",
    "desprendimiento de material": "Desprendimiento"
}

@app.route("/", methods=["GET", "POST"])
def index():
    resultados = []
    resumen_final = {"Grietas": 0, "Humedad": 0, "Desprendimiento": 0}
    imagenes_subidas = []

    if request.method == "POST":
        archivos = request.files.getlist("images")

        for archivo in archivos:
            if archivo:
                filename = str(uuid.uuid4()) + os.path.splitext(archivo.filename)[1]
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                archivo.save(filepath)

                imagenes_subidas.append(filename)

                # Ejecutar predicción
                results = model.predict(filepath)[0]

                # Contar clases por OCR sobre los labels en la imagen (ya que las etiquetas aparecen dibujadas)
                image = Image.open(filepath)
                text = pytesseract.image_to_string(image).lower()

                contador = Counter()
                for clase in CLASSES_TRADUCIDAS.keys():
                    if clase in text:
                        contador[clase] += text.count(clase)

                resultado_imagen = {}

                for clase_en, clase_es in CLASSES_TRADUCIDAS.items():
                    cantidad = contador[clase_en]
                    resultado_imagen[clase_es] = cantidad
                    resumen_final[clase_es] += cantidad

                resultados.append({
                    "archivo": filename,
                    "resultado": resultado_imagen
                })

        # Calcular porcentajes totales
        total = sum(resumen_final.values())
        porcentajes = {}
        for clase, cantidad in resumen_final.items():
            if total == 0:
                porcentajes[clase] = 0
            else:
                porcentajes[clase] = round((cantidad / total) * 100, 1)

        return render_template("index.html", resultados=resultados, resumen=porcentajes, imagenes=imagenes_subidas)

    return render_template("index.html")

# 👇 ESTA PARTE ES LA QUE CAMBIAMOS PARA RENDER 👇
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
