import easyocr
import torch
import gc

class OcrProcesador:
    def __init__(self):
        # Cargamos el motor una vez al iniciar el contenedor
        print("[OCR] Cargando EasyOCR en GPU...")
        self.reader = easyocr.Reader(['es'], gpu=True)
        print("[OCR] Motor OCR cargado.")

    def leer_matricula(self, recorte):
        """
        Procesa el recorte y devuelve el texto.
        """
        try:
            # EasyOCR detecta y lee
            resultados = self.reader.readtext(recorte, detail=0)
            texto = "".join(resultados) if resultados else ""
            
            # Limpieza post-lectura (fundamental para la memoria de la Xavier)
            torch.cuda.empty_cache()
            
            return texto
        except Exception as e:
            print(f"[OCR] Error en lectura: {e}")
            return ""