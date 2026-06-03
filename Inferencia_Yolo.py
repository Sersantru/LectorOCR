#!/usr/bin/env python3
from pathlib import Path
import cv2
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO
import threading
import time

import easyocr
import torch
import gc

app = Flask(__name__)

CAMERA_URL = "http://192.168.150.244:8123/video"
MODEL_PATH = Path(__file__).resolve().parent / "modelo-detector-matricula.pt"

# Inicialización de YOLOv8
model = YOLO(str(MODEL_PATH))

# Variables globales compartidas de forma segura
lock = threading.Lock()
frame_actual = None
coordenadas_guardadas = []
ultima_vez_ia = 0

def bucle_lectura_camara():
    """Hilo secundario que limpia la cola de red constantemente a máxima velocidad"""
    global frame_actual
    print("[CÁMARA] Hilo de captura en tiempo real iniciado.")
    
    # Forzamos los flags nativos y el buffer mínimo
    cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        if not cap.isOpened():
            print("[CÁMARA] Error de conexión. Reintentando...")
            time.sleep(1)
            cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
            
        ok, frame = cap.read()
        if not ok:
            print("[CÁMARA] Stream caído, reiniciando captura...")
            cap.release()
            time.sleep(0.3)
            cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
            
        # Pisamos el frame viejo en la RAM global de forma segura
        with lock:
            # COPIA OBLIGATORIA para evitar colisiones de memoria y el core dumped
            frame_actual = frame.copy()

# ---- RUTAS DE FLASK ----

@app.route('/')
def index():
    # Flask busca automáticamente index.html en la carpeta "templates"
    # y le inyecta la variable CAMERA_URL
    return render_template('index.html', url_camara=CAMERA_URL)

@app.route('/coordenadas')
def coordenadas():
    global coordenadas_guardadas, ultima_vez_ia
    ahora = time.time()
    
    # Restringimos que YOLOv8 procese solo cada 150 milisegundos (evita saturar la GPU)
    if ahora - ultima_vez_ia > 0.15:
        with lock:
            if time.time() - ultima_vez_ia > 0.15:
                if frame_actual is None:
                    return jsonify(coordenadas_guardadas)
                
                # Segunda zona crítica contra el Segmentation Fault: leemos de la copia
                frame_ia = frame_actual.copy()
                
                ultima_vez_ia = ahora
                
                # Inferencia con umbral 0.15 para cazar matrículas en movimiento
                results = model.predict(frame_ia, verbose=False, conf=0.15)
                
                nuevas_cajas = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        nuevas_cajas.append([int(x1), int(y1), int(x2), int(y2), conf])
                
                coordenadas_guardadas = nuevas_cajas

                # LIMPIEZA DE MEMORIA POST-INFERENCIA
                del results
                torch.cuda.empty_cache()
                gc.collect()

    # Retorno sin latencia
    return jsonify(coordenadas_guardadas)

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Arrancamos el demonio de la cámara
    hilo_cam = threading.Thread(target=bucle_lectura_camara, daemon=True)
    hilo_cam.start()
    
    # Levantamos servicio de inferencia/datos
    app.run(host='0.0.0.0', port=8181, threaded=True)
    