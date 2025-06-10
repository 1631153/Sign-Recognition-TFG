import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping

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
            if data.shape == (SEQUENCE_LENGTH, FEATURES):
                sequences.append(data)
                labels.append(label_map[action])
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

X = np.array(sequences)
y = to_categorical(labels).astype(int)

test_size = max(len(actions), int(0.2 * len(X)))
for seed in [123, 456, 789]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

# Modelo LSTM más pequeño para menos datos
model = Sequential()
model.add(LSTM(48, return_sequences=False, activation='relu', input_shape=(SEQUENCE_LENGTH, FEATURES)))
model.add(Dropout(0.3))
model.add(Dense(48, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Callbacks
tb_callback = TensorBoard(log_dir=LOG_DIR)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=15, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[tb_callback, reduce_lr, early_stopping])
(ls,acc)=model.evaluate(x=X_test,y=y_test)
model.save(os.path.join(MODEL_DIR, 'name_model.h5'))
print('MODEL ACCURACY = {}%'.format(acc*100))