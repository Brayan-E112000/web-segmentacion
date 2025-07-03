from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)
# Crear carpetas si no existen
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results/prediccion", exist_ok=True)

# Cargar el modelo segmentado
model = YOLO("best.pt")  # Asegúrate de tener este archivo en esta misma carpeta

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        imagen = request.files["imagen"]
        nombre = str(uuid.uuid4()) + ".jpg"
        ruta_guardada = os.path.join("static/uploads", nombre)
        imagen.save(ruta_guardada)

        model.predict(
            source=ruta_guardada,
            save=True,
            conf=0.3,
            task="segment",
            project="static/results",
            name="prediccion",
            exist_ok=True
        )

        resultado = os.path.join("results/prediccion", nombre)  # ruta para el HTML
        return render_template("index.html", resultado=resultado)

    return render_template("index.html", resultado=None)

# ✅ Esta parte DEBE IR FUERA del if __name__ == "__main__":
@app.route("/galeria")
def galeria():
    archivos = os.listdir("static/results/prediccion")
    archivos = sorted(archivos, reverse=True)  # muestra primero los más recientes
    return render_template("galeria.html", imagenes=archivos)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


