import cv2
import numpy as np
import tensorflow as tf 
import mediapipe as mp
import os

model_path='hand_model5.h5'
model=tf.keras.models.load_model(model_path)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

EXPAND_RATIO = 1.5 

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    lista = os.listdir('Signos ASL')
    
    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            break
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        height, width, _ = frame.shape
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_min, y_min, x_max, y_max = width, height, 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * width), int(lm.y * height)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)
                
                # Expandir el área de detección
                hand_width = x_max - x_min
                hand_height = y_max - y_min
                x_min = max(0, x_min - int(hand_width * (EXPAND_RATIO - 1) / 2))
                y_min = max(0, y_min - int(hand_height * (EXPAND_RATIO - 1) / 2))
                x_max = min(width, x_max + int(hand_width * (EXPAND_RATIO - 1) / 2))
                y_max = min(height, y_max + int(hand_height * (EXPAND_RATIO - 1) / 2))
                
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue
                
                handedness = result.multi_handedness[0].classification[0].label
                if handedness == "Right":
                    roi = cv2.flip(roi, 1)
                    print("right")
                
                k = 2
                resized = cv2.resize(roi, (28*k, 28*k), interpolation=cv2.INTER_AREA) / 255
                pred = model.predict(resized.reshape(-1, 28*k, 28*k, 3))
                abc = 'ABCDEFGHIKLMNOPQRSTUVWXY'
                idx = np.argmax(pred)
                letra = abc[idx]
                confianza = pred[0][idx] * 100  # Probabilidad en porcentaje

                cv2.rectangle(clone, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(clone, f"{letra} ({confianza:.1f}%)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
        cv2.imshow("Video Feed", clone)
        
        keypress = cv2.waitKey(1)
        if keypress == ord("q"):
            break
        elif keypress == ord(" "):
            letrica = lista[np.random.randint(len(lista))]
            letraimagen = cv2.imread('Signos ASL/' + letrica)
            letraimagen = cv2.putText(letraimagen, str(letrica[0]), (10, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 2, (226, 43, 138), 2, cv2.LINE_AA)
            cv2.imshow("Letra", letraimagen)
    
    camera.release()
    cv2.destroyAllWindows()