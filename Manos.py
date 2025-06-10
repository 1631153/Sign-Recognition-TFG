#ESTO ES UNA VERSIÓN ARREGLADA DEL TUTORIAL DE YT https://www.youtube.com/watch?v=RMI1vjFvV14&t=0s

import cv2
import mediapipe as mp
import os

nombre = 'Letra A'
direccion = '/Fotos/Train' #Editado sin probar. Usaba un absolute path.
carpeta = os.path.join(direccion, nombre)
if not os.path.exists(carpeta):
    print('Carpeta creada: ', carpeta)
    os.makedirs(carpeta)

cont = 0  # contador de fotos
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

clase_manos = mp.solutions.hands
manos = clase_manos.Hands()
dibujo = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame de la cámara.")
        break
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []

    if resultado.multi_hand_landmarks:
        print("Manos detectadas.")
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape 
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
        
        if posiciones:  # comprobación de que hay posiciones
            pto_i5 = posiciones[9]  # Punto central de la mano
            x1, y1 = max(pto_i5[1] - 100, 0), max(pto_i5[2] - 100, 0) 
            x2, y2 = min(x1 + 200, ancho), min(y1 + 200, alto)
            dedos_reg = copia[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            if dedos_reg.size > 0:  # asegurarse de que hay contenido en dedos_reg
                print(f"Dimensiones de dedos_reg: {dedos_reg.shape}")
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                try:
                    guardado = cv2.imwrite(os.path.join(carpeta, f'Dedos_{cont}.jpg'), dedos_reg)
                    if guardado:
                        print(f"Imagen guardada: Dedos_{cont}.jpg")
                    else:
                        print("Imagen NO guardada")
                    cont += 1
                except Exception as e:
                    print(f"Error al guardar la imagen: {e}")

    else:
        print("No se detectaron manos.")

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27 or cont >= 300:
        break

cap.release()
cv2.destroyAllWindows()
