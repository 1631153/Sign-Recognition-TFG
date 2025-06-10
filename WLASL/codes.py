import os
import cv2
import json
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = 'extracted_keypoints_h2s'
VIDEO_PATH = 'videos'
JSON_PATH = 'WLASL_v0.3.json'
TARGET_WORDS = ['my', 'name', 'j', 'a', 'v', 'i']

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

video_to_word = {}
for entry in data:
    word = entry['gloss']
    if word in TARGET_WORDS:
        for instance in entry['instances']:
            video_id = instance['video_id']
            video_to_word[video_id] = word

for word in TARGET_WORDS:
    os.makedirs(os.path.join(DATA_PATH, word), exist_ok=True)

# Crear carpetas para cada palabra
#all_words = list(set(video_to_word.values()))
#for word in all_words:
    #os.makedirs(os.path.join(DATA_PATH, word), exist_ok=True)

# Procesar vídeos
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    for video_file in os.listdir(VIDEO_PATH):
        video_id = os.path.splitext(video_file)[0]
        if video_id not in video_to_word:
            continue  # Saltar si no está en el JSON

        word = video_to_word[video_id]
        cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, video_file))
        sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            lh = np.zeros(21 * 3)
            rh = np.zeros(21 * 3)

            if results.multi_hand_landmarks and results.multi_handedness:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[i].classification[0].label
                    keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                    if handedness == 'Left':
                        lh = keypoints
                    else:
                        rh = keypoints

            keypoints = np.concatenate([lh, rh])
            sequence.append(keypoints)

        cap.release()

        # Solo se guarda si hay suficientes frames
        if len(sequence) >= 30:
            np.save(os.path.join(DATA_PATH, word, f'{video_id}.npy'), np.array(sequence[:30]))
