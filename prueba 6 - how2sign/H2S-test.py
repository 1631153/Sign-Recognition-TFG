import cv2
import numpy as np
import mediapipe as mp
import os
import json
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# Config
ENCODER_PATH = 'gesture_encoder.h5'
KEYPOINTS_DIR = 'extracted_keypoints_holistic'
CSV_PATH = 'how2sign_train.csv'
TOP_K = 3  # número de frases más parecidas a mostrar

# Cargar modelo encoder
encoder = load_model(ENCODER_PATH)

# Cargar CSV para mapear frases
import pandas as pd
df = pd.read_csv(CSV_PATH, delimiter='\t')
sentence_map = dict(zip(df['SENTENCE_NAME'], df['SENTENCE']))

# MediaPipe
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

    return np.concatenate([pose, lh, rh])

# Crear base de embeddings del dataset
known_embeddings = []
known_labels = []

for file in os.listdir(KEYPOINTS_DIR):
    if not file.endswith('.npy'):
        continue
    data = np.load(os.path.join(KEYPOINTS_DIR, file))
    if data.shape == (30, 258):
        emb = encoder.predict(np.expand_dims(data, axis=0), verbose=0)[0]
        known_embeddings.append(emb)
        label = sentence_map.get(file.replace('.npy', ''), file.replace('.npy', ''))
        known_labels.append(label)

known_embeddings = np.array(known_embeddings)

# Capturar desde webcam
cap = cv2.VideoCapture(0)
sequence = []

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    print("🟢 Comienza a hacer tu seña... (30 frames)")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Mostrar cámara
        cv2.putText(frame, f"Frames capturados: {len(sequence)}/30", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Sign Capture", frame)

        if len(sequence) == 30:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Si no hay suficientes frames
if len(sequence) < 30:
    print("❌ No se capturaron suficientes frames.")
    exit()

# Generar embedding del gesto capturado
sequence = np.array(sequence)
embedding = encoder.predict(np.expand_dims(sequence, axis=0), verbose=0)

# Comparar con embeddings del dataset
similarities = cosine_similarity(embedding, known_embeddings)[0]
top_indices = similarities.argsort()[-TOP_K:][::-1]

# Mostrar frases más similares
print("\n🔍 Frases más parecidas a tu seña:")
for idx in top_indices:
    print(f"→ \"{known_labels[idx]}\" (similaridad: {similarities[idx]:.2f})")
