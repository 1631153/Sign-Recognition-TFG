import cv2
import numpy as np
import tensorflow as tf
from time import sleep

# Cargar el modelo entrenado
model_path = 'hand_model.h5'
model = tf.keras.models.load_model(model_path)

# Etiquetas de las letras (excluyendo J y Z)
abc = 'ABCDEFGHIKLMNOPQRSTUVWXY'

def predict_letter(roi):
    """ Redimensiona y normaliza la imagen antes de hacer la predicción. """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)            # Convertir a escala de grises
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    input_img = normalized.reshape(-1, 28, 28, 1)            # Añadir dimensiones necesarias
    pred = model.predict(input_img)
    index = np.argsort(pred[0])[-3:]                        # Top 3 predicciones
    return abc[index[-1]], abc[index[-2]], abc[index[-3]]

# Inicializar cámara
camera = cv2.VideoCapture(0)

top, right, bottom, left = 50, 50, 200, 200  # Región de interés (ROI)
num_frames = 0

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    roi = frame[top:bottom, right:left]  # Extraer ROI
    
    # Predecir la letra
    letra, l2, l3 = predict_letter(roi)
    
    # Dibujar la región de interés y mostrar predicción
    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(clone, letra, (left - 90, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(clone, l2, (left - 150, top + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(clone, l3, (left - 10, top + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    cv2.imshow("Video Feed", clone)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
camera.release()
cv2.destroyAllWindows()
