import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau

DATA_PATH = 'extracted_keypoints'
LOG_DIR = 'logs'
MODEL_DIR = 'models'
SEQUENCE_LENGTH = 30
FEATURES = 126  # 21 puntos * 3 coordenadas * 2 manos
TARGET_WORDS = ['my', 'name', 'j', 'a', 'v', 'i']

os.makedirs(MODEL_DIR, exist_ok=True)
available_actions = [a for a in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, a))]
actions = np.array(sorted([a for a in TARGET_WORDS if a in available_actions]))
label_map = {label: num for num, label in enumerate(actions)}

# Cargar datos
sequences, labels = [], []
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    for file in os.listdir(action_path):
        file_path = os.path.join(action_path, file)
        try:
            data = np.load(file_path)
            if data.shape == (SEQUENCE_LENGTH, FEATURES):  # Validar forma
                sequences.append(data)
                labels.append(label_map[action])
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Crear modelo LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, FEATURES)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Entrenamiento
tb_callback = TensorBoard(log_dir=LOG_DIR)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
model.fit(X_train, y_train, validation_split=0.2, epochs=1000, callbacks=[tb_callback, reduce_lr])

# Guardar modelo
model.save(os.path.join(MODEL_DIR, 'wlasl_model.h5'))
