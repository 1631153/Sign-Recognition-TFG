import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("hand_model.h5")

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir imagen a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar imagen con MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obtener coordenadas de los 21 puntos clave de la mano
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')

            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)

                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Expande un poco el área para asegurarse de incluir la mano completa
            margen = 20
            x_min, y_min = max(x_min - margen, 0), max(y_min - margen, 0)
            x_max, y_max = min(x_max + margen, w), min(y_max + margen, h)

            # Recortar la región de la mano
            mano_recortada = frame[y_min:y_max, x_min:x_max]

            if mano_recortada.size == 0:
                continue

            # Redimensionar a 200x200
            mano_redimensionada = cv2.resize(mano_recortada, (200, 200))

            # Normalizar valores de píxeles
            mano_redimensionada = mano_redimensionada / 255.0

            # Asegurar el shape correcto
            entrada = np.expand_dims(mano_redimensionada, axis=0)  # Shape (1, 200, 200, 3)

            # Hacer la predicción
            prediccion = modelo.predict(entrada)[0][0]
            mano_detectada = "Derecha" if prediccion > 0.5 else "Izquierda"

            # Dibujar la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Mostrar resultado en pantalla
            cv2.putText(frame, f"Mano: {mano_detectada}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el video
    cv2.imshow("Detección de Mano", frame)

    # Salir con la tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
