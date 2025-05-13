import cv2
import os
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model

model = load_model('models/name_model.h5')
DATA_PATH = 'extracted_keypoints'
TARGET_WORDS = ['my', 'name', 'j', 'a', 'v', 'i']

# Obtener las acciones/clases
available_actions = [a for a in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, a)) and not a.startswith('.')]
actions = np.array(sorted([a for a in TARGET_WORDS if a in available_actions]))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if handedness.classification[0].label == 'Left':
                lh = coords
            else:
                rh = coords
    return np.concatenate([lh, rh])

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Variables para la inferencia
sequence = []
sentence = []
threshold = 0.25
last_prediction_time = time.time()
prediction_delay = 2  # segundos entre palabras
frame_count = 0
frame_interval = 30  # reiniciar cada 30 frames

# Iniciar cámara
cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count = (frame_count + 1) % frame_interval

        image, results = mediapipe_detection(frame, hands)
        draw_styled_landmarks(image, results)

        # Contador en pantalla
        cv2.putText(image, f'Frame: {frame_count}', (500, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        # Solo continuar si se detecta al menos una mano
        #if not results.multi_hand_landmarks:
         #   cv2.imshow('Hand Gesture Recognition', image)
          #  if cv2.waitKey(10) & 0xFF == ord('q'):
           #     break
           # continue

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            current_time = time.time()

            # Mostrar los porcentajes de cada acción
            for action, prob in zip(actions, res):
                print(f'{action}: {prob:.2f}')

            if res[np.argmax(res)] > threshold:
                predicted_action = actions[np.argmax(res)]
                time_since_last = current_time - last_prediction_time

                if (len(sentence) == 0 or predicted_action != sentence[-1]) and time_since_last > prediction_delay:
                    sentence.append(predicted_action)
                    last_prediction_time = current_time

        # Mostrar resultados en pantalla
        cv2.rectangle(image, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(image, ' '.join(sentence), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Hand Gesture Recognition', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
