#Ya que este codigo está modificado con chat, ya de paso hago los comentarios con chat. En la versión final lo cambio todo.
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Cargar el modelo de reconocimiento de lenguaje de señas previamente entrenado
model = load_model("hand_model.h5")

# Diccionario que mapea las clases del modelo a las letras del alfabeto
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# Inicializar el modelo de Mediapipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Tamaño de imagen requerido para la entrada del modelo
IMG_SIZE = 28  

# Capturar video desde la webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()  # Leer un fotograma de la cámara
    if not ret:
        break  # Si no se obtiene un fotograma, salir del bucle

    # Convertir el fotograma de BGR a RGB para su procesamiento con Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)  # Procesar la imagen para detectar manos

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Inicializar coordenadas para delimitar la mano detectada
            x_min, y_min = int(frame.shape[1]), int(frame.shape[0])
            x_max, y_max = 0, 0
            
            # Recorrer los puntos clave de la mano detectada para encontrar sus límites
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Agregar un margen para mejorar la detección
            margin = 20
            x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
            x_max, y_max = min(frame.shape[1], x_max + margin), min(frame.shape[0], y_max + margin)

            # Recortar la región de la imagen donde está la mano
            hand_img = frame[y_min:y_max, x_min:x_max]

            # Verificar que el recorte no esté vacío antes de procesarlo
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                # Convertir la imagen de la mano a escala de grises
                hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                
                # Redimensionar la imagen para que coincida con el tamaño de entrada del modelo
                hand_gray = cv2.resize(hand_gray, (IMG_SIZE, IMG_SIZE))
                
                # Normalizar los valores de la imagen (convertirlos a rango 0-1)
                hand_gray = hand_gray / 255.0  
                
                # Ajustar la forma de la imagen para que sea compatible con el modelo
                hand_gray = np.reshape(hand_gray, (1, IMG_SIZE, IMG_SIZE, 1))  
                
                # Realizar la predicción con el modelo entrenado
                pred = model.predict(hand_gray)
                
                # Obtener la clase con mayor probabilidad
                pred_label = np.argmax(pred)  
                
                # Obtener la letra correspondiente a la predicción
                label_text = labels_dict.get(pred_label, "Desconocido")
                
                # Mostrar la predicción en la pantalla
                cv2.putText(frame, f"Letra: {label_text}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Dibujar las conexiones de la mano en la imagen para visualización
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar la imagen con la detección y la predicción
    cv2.imshow("Sign Language Detection", frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
