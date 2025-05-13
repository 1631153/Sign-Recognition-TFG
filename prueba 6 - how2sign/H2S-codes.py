import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# Configuración
CSV_PATH = 'how2sign_train.csv'
VIDEO_DIR = 'raw_videos'
OUTPUT_DIR = 'extracted_keypoints_holistic_full'
TARGET_KEYWORDS = ['hello', 'my name is', 'javi']  # ← opcional, puedes comentar esto si quieres todos

# Crear carpeta de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.zeros(33 * 4)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if results.pose_landmarks:
        pose = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]).flatten()
    if results.left_hand_landmarks:
        lh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.left_hand_landmarks.landmark]).flatten()
    if results.right_hand_landmarks:
        rh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([pose, lh, rh])  # Total: 258

# Cargar CSV y filtrar frases
df = pd.read_csv(CSV_PATH, delimiter='\t')
if TARGET_KEYWORDS:
    filtered_df = df[df['SENTENCE'].str.contains('|'.join(TARGET_KEYWORDS), case=False, na=False)]
else:
    filtered_df = df.copy()

# Procesar vídeos
with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for idx, row in filtered_df.iterrows():
        sentence_name = row['SENTENCE_NAME']
        sentence = row['SENTENCE'].strip()
        filename = f"{sentence_name}.mp4"
        video_path = os.path.join(VIDEO_DIR, filename)

        if not os.path.exists(video_path):
            print(f"⛔ Video no encontrado: {filename}")
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{sentence_name}.npy")
        if os.path.exists(out_path):
            print(f"✅ Ya existe: {sentence_name}")
            continue

        print(f"📹 Procesando: {filename}")
        cap = cv2.VideoCapture(video_path)
        sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

        cap.release()

        if len(sequence) == 0:
            print(f"❌ No se detectaron manos/cuerpo en {filename}")
            continue

        # Guardar secuencia completa
        np.save(out_path, np.array(sequence))
